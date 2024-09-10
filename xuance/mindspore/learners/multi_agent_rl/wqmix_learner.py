"""
Weighted QMIX
Paper link:
https://proceedings.neurips.cc/paper/2020/file/73a427badebe0e32caa2e1fc7530b7f3-Paper.pdf
Implementation: MindSpore
"""
from mindspore.nn import MSELoss
from xuance.mindspore import ms, Module, Tensor, optim, ops
from xuance.mindspore.learners import LearnerMAS
from xuance.mindspore.utils import clip_grads
from xuance.common import List
from argparse import Namespace
from operator import itemgetter


class WQMIX_Learner(LearnerMAS):
    def __init__(self,
                 config: Namespace,
                 model_keys: List[str],
                 agent_keys: List[str],
                 policy: Module):
        super(WQMIX_Learner, self).__init__(config, model_keys, agent_keys, policy)
        self.optimizer = optim.Adam(params=self.policy.trainable_params(), lr=config.learning_rate, eps=1e-5)
        self.scheduler = optim.lr_scheduler.LinearLR(self.optimizer, start_factor=1.0, end_factor=0.5,
                                                     total_iters=self.config.running_steps)
        self.alpha = config.alpha
        self.gamma = config.gamma
        self.sync_frequency = config.sync_frequency
        self.mse_loss = MSELoss()
        self.n_actions = {k: self.policy.action_space[k].n for k in self.model_keys}
        # Get gradient function
        self.grad_fn = ms.value_and_grad(self.forward_fn, None, self.optimizer.parameters, has_aux=True)
        self.policy.set_train()

    def forward_fn(self, state, obs, actions, agt_mask, avail_actions, ids, target_value):
        # calculate Q_tot
        _, action_max, q_eval = self.policy(observation=obs, agent_ids=ids, avail_actions=avail_actions)
        _, q_eval_centralized = self.policy.q_centralized(observation=obs, agent_ids=ids)

        q_eval_a, q_eval_centralized_a, q_eval_next_centralized_a, act_next = {}, {}, {}, {}
        for key in self.model_keys:
            action_max[key] = action_max[key].unsqueeze(-1)
            q_eval_a[key] = q_eval[key].gather(actions[key].unsqueeze(-1).astype(ms.int32), -1, -1).reshape(-1)
            q_eval_centralized_a[key] = q_eval_centralized[key].gather(action_max[key].astype(ms.int32),
                                                                       -1, -1).reshape(-1)
            q_eval_a[key] *= agt_mask[key]
            q_eval_centralized_a[key] *= agt_mask[key]

        q_tot_eval = self.policy.Q_tot(q_eval_a, state)  # calculate Q_tot
        q_tot_centralized = self.policy.q_feedforward(q_eval_centralized_a, state)  # calculate centralized Q
        td_error = q_tot_eval - target_value

        # calculate weights
        ones = ops.ones_like(td_error)
        w = ones * self.alpha
        if self.config.agent == "CWQMIX":
            condition_1 = ((action_max == actions.reshape([-1, self.n_agents, 1])) * agt_mask).all(axis=1)
            condition_2 = target_value > q_tot_centralized
            conditions = condition_1 | condition_2
            w = ops.where(conditions, ones, w)
        elif self.config.agent == "OWQMIX":
            condition = td_error < 0
            w = ops.where(condition, ones, w)
        else:
            raise AttributeError(f"The agent named is {self.config.agent} is currently not supported.")

        # calculate losses and train
        loss_central = self.mse_loss(logits=q_tot_centralized, labels=ops.stop_gradient(target_value))
        loss_qmix = (ops.stop_gradient(w) * (td_error ** 2)).mean()
        loss = loss_qmix + loss_central
        return loss, loss_qmix, loss_central, q_tot_eval

    def update(self, sample):
        self.iterations += 1
        info = {}

        # prepare training data
        sample_Tensor = self.build_training_data(sample=sample,
                                                 use_parameter_sharing=self.use_parameter_sharing,
                                                 use_actions_mask=self.use_actions_mask,
                                                 use_global_state=True)
        batch_size = sample_Tensor['batch_size']
        state = sample_Tensor['state']
        state_next = sample_Tensor['state_next']
        obs = sample_Tensor['obs']
        actions = sample_Tensor['actions']
        obs_next = sample_Tensor['obs_next']
        rewards = sample_Tensor['rewards']
        terminals = sample_Tensor['terminals']
        agent_mask = sample_Tensor['agent_mask']
        avail_actions = sample_Tensor['avail_actions']
        avail_actions_next = sample_Tensor['avail_actions_next']
        IDs = sample_Tensor['agent_ids']

        if self.use_parameter_sharing:
            key = self.model_keys[0]
            bs = batch_size * self.n_agents
            rewards_tot = rewards[key].mean(axis=1).reshape(batch_size, 1)
            terminals_tot = terminals[key].all(axis=1).astype(ms.float32).reshape(batch_size, 1)
        else:
            bs = batch_size
            rewards_tot = ops.stack(itemgetter(*self.agent_keys)(rewards), axis=1).mean(axis=-1).reshape(batch_size, 1)
            terminals_tot = ops.stack(itemgetter(*self.agent_keys)(terminals),
                                      axis=1).all(axis=1).astype(ms.float32).reshape(batch_size, 1)

        # calculate Q_tot
        _, q_eval_next_centralized = self.policy.target_q_centralized(observation=obs_next, agent_ids=IDs)

        q_eval_next_centralized_a, act_next = {}, {}
        for key in self.model_keys:
            if self.config.double_q:
                _, a_next_greedy, _ = self.policy(observation=obs_next, agent_ids=IDs,
                                                  avail_actions=avail_actions_next, agent_key=key)
                act_next[key] = a_next_greedy[key].unsqueeze(-1)
            else:
                _, q_next_eval = self.policy.Qtarget(observation=obs_next, agent_ids=IDs, agent_key=key)
                if self.use_actions_mask:
                    q_next_eval[key][avail_actions_next[key] == 0] = -9999999
                act_next[key] = q_next_eval[key].argmax(dim=-1, keepdim=True)
            q_eval_next_centralized_a[key] = q_eval_next_centralized[key].gather(act_next[key], -1, -1).reshape(bs)
            q_eval_next_centralized_a[key] *= agent_mask[key]

        q_tot_next_centralized = self.policy.target_q_feedforward(q_eval_next_centralized_a, state_next)  # y_i

        target_value = rewards_tot + (1 - terminals_tot) * self.gamma * q_tot_next_centralized

        (loss, loss_qmix, loss_central, q_tot_eval), grads = self.grad_fn(state, obs, actions, agent_mask,
                                                                          avail_actions, IDs, target_value)
        if self.use_grad_clip:
            grads = clip_grads(grads, Tensor(-self.grad_clip_norm), Tensor(self.grad_clip_norm))
        self.optimizer(grads)

        if self.iterations % self.sync_frequency == 0:
            self.policy.copy_target()
        self.scheduler.step()
        lr = self.scheduler.get_last_lr()[0]

        info.update({
            "learning_rate": lr.asnumpy(),
            "loss_Qmix": loss_qmix.asnumpy(),
            "loss_central": loss_central.asnumpy(),
            "loss": loss.asnumpy(),
            "predictQ": q_tot_eval.mean().asnumpy()
        })

        return info
