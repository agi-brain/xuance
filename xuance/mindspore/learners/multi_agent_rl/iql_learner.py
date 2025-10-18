"""
Independent Q-learning (IQL)
Implementation: MindSpore
"""
from xuance.mindspore import ms, Module, Tensor, optim, ops
from xuance.mindspore.learners import LearnerMAS
from xuance.mindspore.utils import clip_grads
from xuance.common import List
from argparse import Namespace


class IQL_Learner(LearnerMAS):
    def __init__(self,
                 config: Namespace,
                 model_keys: List[str],
                 agent_keys: List[str],
                 policy: Module,
                 callback):
        super(IQL_Learner, self).__init__(config, model_keys, agent_keys, policy, callback)
        self.optimizer = {key: optim.Adam(params=self.policy.parameters_model[key], lr=self.config.learning_rate,
                                          eps=1e-5) for key in self.model_keys}
        self.scheduler = {key: optim.lr_scheduler.LinearLR(self.optimizer[key], start_factor=1.0, end_factor=self.end_factor_lr_decay,
                                                           total_iters=self.config.running_steps)
                          for key in self.model_keys}
        self.gamma = config.gamma
        self.sync_frequency = config.sync_frequency
        self.n_actions = {k: self.policy.action_space[k].n for k in self.model_keys}
        # Get gradient function
        self.grad_fn = {key: ms.value_and_grad(self.forward_fn, None, self.optimizer[key].parameters, has_aux=True)
                        for key in self.model_keys}
        self.policy.set_train()

    def forward_fn(self, obs, actions, agt_mask, avail_actions, ids, q_target, agent_key):
        rnn_hidden = None
        _, _, q_eval = self.policy(observation=obs, agent_ids=ids, avail_actions=avail_actions,
                                   agent_key=agent_key, rnn_hidden=rnn_hidden)
        q_eval_a = q_eval[agent_key].gather(actions[agent_key].astype(ms.int32).unsqueeze(-1), axis=-1, batch_dims=-1)
        td_error = (q_eval_a.reshape(-1) - q_target) * agt_mask
        loss = (td_error ** 2).sum() / agt_mask.sum()
        return loss, q_eval_a, td_error

    def update(self, sample):
        self.iterations += 1

        # prepare training data
        sample_Tensor = self.build_training_data(sample=sample,
                                                 use_parameter_sharing=self.use_parameter_sharing,
                                                 use_actions_mask=self.use_actions_mask)
        batch_size = sample_Tensor['batch_size']
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
            rewards[key] = rewards[key].reshape(batch_size * self.n_agents)
            terminals[key] = terminals[key].reshape(batch_size * self.n_agents)
        else:
            bs = batch_size

        info = self.callback.on_update_start(self.iterations, method="update",
                                             policy=self.policy, sample_Tensor=sample_Tensor, bs=bs)

        _, q_next = self.policy.Qtarget(observation=obs_next, agent_ids=IDs)

        for key in self.model_keys:
            mask_values = agent_mask[key]

            if self.use_actions_mask:
                q_next[key][avail_actions_next[key] == 0] = -1e10

            if self.config.double_q:
                _, actions_next_greedy, _ = self.policy(obs_next, IDs, agent_key=key, avail_actions=avail_actions)
                q_next_a = q_next[key].gather(actions_next_greedy[key].unsqueeze(-1).long(), -1, -1).reshape(bs)
            else:
                q_next_a = q_next[key].max(dim=-1, keepdim=True).values.reshape(bs)

            q_target = rewards[key] + (1 - terminals[key]) * self.gamma * q_next_a

            (loss, q_eval_a, td_error), grads = self.grad_fn[key](obs, actions, mask_values,
                                                                  avail_actions, IDs, q_target, key)
            if self.use_grad_clip:
                grads = clip_grads(grads, Tensor(-self.grad_clip_norm), Tensor(self.grad_clip_norm))
            self.optimizer[key](grads)

            self.scheduler[key].step()
            lr = self.scheduler[key].get_last_lr()[0]

            info.update({
                f"{key}/learning_rate": lr.asnumpy(),
                f"{key}/loss_Q": loss.asnumpy(),
                f"{key}/predictQ": q_eval_a.mean().asnumpy()
            })

            info.update(self.callback.on_update_agent_wise(self.iterations, key, info=info, method="update",
                                                           mask_values=mask_values, q_eval_a=q_eval_a,
                                                           q_next_a=q_next_a, q_target=q_target,
                                                           td_error=td_error, loss=loss))

        if self.iterations % self.sync_frequency == 0:
            self.policy.copy_target()

        info.update(self.callback.on_update_end(self.iterations, method="update", policy=self.policy, info=info))

        return info
