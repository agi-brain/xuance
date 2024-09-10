"""
Qmix: Monotonic value function factorisation for deep multi-agent reinforcement learning
Paper link:
http://proceedings.mlr.press/v80/rashid18a/rashid18a.pdf
Implementation: MindSpore
"""
from mindspore.nn import MSELoss
from xuance.mindspore import ms, Module, Tensor, optim, ops
from xuance.mindspore.learners import LearnerMAS
from xuance.mindspore.utils import clip_grads
from xuance.common import List
from argparse import Namespace
from operator import itemgetter


class QMIX_Learner(LearnerMAS):
    def __init__(self,
                 config: Namespace,
                 model_keys: List[str],
                 agent_keys: List[str],
                 policy: Module):
        self.gamma = config.gamma
        self.sync_frequency = config.sync_frequency
        self.mse_loss = MSELoss()
        super(QMIX_Learner, self).__init__(config, model_keys, agent_keys, policy)
        self.optimizer = optim.Adam(params=self.policy.trainable_params(), lr=config.learning_rate, eps=1e-5)
        self.scheduler = optim.lr_scheduler.LinearLR(self.optimizer, start_factor=1.0, end_factor=0.5,
                                                     total_iters=self.config.running_steps)
        self.n_actions = {k: self.policy.action_space[k].n for k in self.model_keys}
        # Get gradient function
        self.grad_fn = ms.value_and_grad(self.forward_fn, None, self.optimizer.parameters, has_aux=True)
        self.policy.set_train()

    def forward_fn(self, state, obs, actions, agt_mask, avail_actions, ids, q_tot_target):
        _, _, q_eval = self.policy(observation=obs, agent_ids=ids, avail_actions=avail_actions)
        q_eval_a = {}
        for key in self.model_keys:
            q_eval_a[key] = q_eval[key].gather(actions[key].unsqueeze(-1).astype(ms.int32), -1, -1).reshape(-1)
            q_eval_a[key] *= agt_mask[key]
        q_tot_eval = self.policy.Q_tot(q_eval_a, state)
        loss = self.mse_loss(logits=q_tot_eval, labels=q_tot_target)
        return loss, q_tot_eval

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

        _, q_next = self.policy.Qtarget(observation=obs_next, agent_ids=IDs)
        q_next_a = {}
        for key in self.model_keys:
            if self.use_actions_mask:
                q_next[key][avail_actions_next[key] == 0] = -9999999

            if self.config.double_q:
                _, act_next, _ = self.policy(observation=obs_next, agent_ids=IDs,
                                             avail_actions=avail_actions, agent_key=key)
                q_next_a[key] = q_next[key].gather(act_next[key].astype(ms.int32).unsqueeze(-1), -1, -1).reshape(bs)
            else:
                q_next_a[key] = q_next[key].max(axis=-1, keepdim=True).values.reshape(bs)
            q_next_a[key] *= agent_mask[key]

        q_tot_next = self.policy.Qtarget_tot(q_next_a, state_next)
        q_tot_target = rewards_tot + (1 - terminals_tot) * self.gamma * q_tot_next

        # calculate the loss function
        (loss, q_tot_eval), grads = self.grad_fn(state, obs, actions, agent_mask, avail_actions, IDs, q_tot_target)
        if self.use_grad_clip:
            grads = clip_grads(grads, Tensor(-self.grad_clip_norm), Tensor(self.grad_clip_norm))
        self.optimizer(grads)

        self.scheduler.step()
        lr = self.scheduler.get_last_lr()[0]

        info.update({
            "learning_rate": lr.asnumpy(),
            "loss_Q": loss.asnumpy(),
            "predictQ": q_tot_eval.mean().asnumpy()
        })

        if self.iterations % self.sync_frequency == 0:
            self.policy.copy_target()
        return info
