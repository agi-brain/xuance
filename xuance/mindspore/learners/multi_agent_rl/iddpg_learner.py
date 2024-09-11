"""
Independent Deep Deterministic Policy Gradient (IDDPG)
Implementation: MindSpore
"""
from mindspore.nn import MSELoss
from xuance.mindspore import ms, Module, Tensor, optim, ops
from xuance.mindspore.learners import LearnerMAS
from xuance.mindspore.utils import clip_grads
from xuance.common import List
from argparse import Namespace


class IDDPG_Learner(LearnerMAS):
    def __init__(self,
                 config: Namespace,
                 model_keys: List[str],
                 agent_keys: List[str],
                 policy: Module):
        super().__init__(config, model_keys, agent_keys, policy)
        self.optimizer = {
            key: {
                'actor': optim.Adam(params=self.policy.parameters_actor[key], lr=self.config.learning_rate_actor,
                                    eps=1e-5),
                'critic': optim.Adam(params=self.policy.parameters_critic[key], lr=self.config.learning_rate_critic,
                                     eps=1e-5)}
            for key in self.model_keys}
        self.scheduler = {
            key: {'actor': optim.lr_scheduler.LinearLR(self.optimizer[key]['actor'], start_factor=1.0,
                                                       end_factor=0.5, total_iters=self.config.running_steps),
                  'critic': optim.lr_scheduler.LinearLR(self.optimizer[key]['critic'], start_factor=1.0,
                                                        end_factor=0.5, total_iters=self.config.running_steps)}
            for key in self.model_keys}
        self.gamma = config.gamma
        self.tau = config.tau
        self.mse_loss = MSELoss()
        # Get gradient function
        self.grad_fn_actor = {key: ms.value_and_grad(self.forward_fn_actor, None,
                                                     self.optimizer[key]['actor'].parameters, has_aux=True)
                              for key in self.model_keys}
        self.grad_fn_critic = {key: ms.value_and_grad(self.forward_fn_critic, None,
                                                      self.optimizer[key]['critic'].parameters, has_aux=True)
                              for key in self.model_keys}
        self.policy.set_train()

    def forward_fn_actor(self, obs, ids, mask_values, agent_key):
        _, actions_eval = self.policy(observation=obs, agent_ids=ids, agent_key=agent_key)
        _, q_policy = self.policy.Qpolicy(observation=obs, actions=actions_eval, agent_ids=ids, agent_key=agent_key)
        q_policy_i = q_policy[agent_key].reshape(-1)
        loss_a = -(q_policy_i * mask_values).sum() / mask_values.sum()
        return loss_a, q_policy_i

    def forward_fn_critic(self, obs, actions, ids, mask_values, q_target, agent_key):
        _, q_eval = self.policy.Qpolicy(observation=obs, actions=actions, agent_ids=ids, agent_key=agent_key)
        q_eval_a = q_eval[agent_key].reshape(-1)
        td_error = (q_eval_a - ops.stop_gradient(q_target)) * mask_values
        loss_c = (td_error ** 2).sum() / mask_values.sum()
        return loss_c, q_eval_a

    def update(self, sample):
        self.iterations += 1
        info = {}

        # prepare training data.
        sample_Tensor = self.build_training_data(sample,
                                                 use_parameter_sharing=self.use_parameter_sharing,
                                                 use_actions_mask=False)
        batch_size = sample_Tensor['batch_size']
        obs = sample_Tensor['obs']
        actions = sample_Tensor['actions']
        obs_next = sample_Tensor['obs_next']
        rewards = sample_Tensor['rewards']
        terminals = sample_Tensor['terminals']
        agent_mask = sample_Tensor['agent_mask']
        IDs = sample_Tensor['agent_ids']
        if self.use_parameter_sharing:
            key = self.model_keys[0]
            bs = batch_size * self.n_agents
            rewards[key] = rewards[key].reshape(batch_size * self.n_agents)
            terminals[key] = terminals[key].reshape(batch_size * self.n_agents)
        else:
            bs = batch_size

        # feedforward
        _, next_actions = self.policy.Atarget(next_observation=obs_next, agent_ids=IDs)
        _, q_next = self.policy.Qtarget(next_observation=obs_next, next_actions=next_actions, agent_ids=IDs)

        for key in self.model_keys:
            mask_values = agent_mask[key]

            # updata critic
            q_next_i = q_next[key].reshape(bs)
            q_target = rewards[key] + (1 - terminals[key]) * self.gamma * q_next_i
            (loss_c, q_eval_a), grads_critic = self.grad_fn_critic[key](obs, actions, IDs, mask_values, q_target, key)
            if self.use_grad_clip:
                grads_critic = clip_grads(grads_critic, Tensor(-self.grad_clip_norm), Tensor(self.grad_clip_norm))
            self.optimizer[key]['critic'](grads_critic)

            # update actor
            (loss_a, _), grads_actor = self.grad_fn_actor[key](obs, IDs, mask_values, key)
            if self.use_grad_clip:
                grads_actor = clip_grads(grads_actor, Tensor(-self.grad_clip_norm), Tensor(self.grad_clip_norm))
            self.optimizer[key]['actor'](grads_actor)

            self.scheduler[key]['actor'].step()
            self.scheduler[key]['critic'].step()
            learning_rate_actor = self.scheduler[key]['actor'].get_last_lr()[0]
            learning_rate_critic = self.scheduler[key]['critic'].get_last_lr()[0]

            info.update({
                f"{key}/learning_rate_actor": learning_rate_actor.asnumpy(),
                f"{key}/learning_rate_critic": learning_rate_critic.asnumpy(),
                f"{key}/loss_actor": loss_a.asnumpy(),
                f"{key}/loss_critic": loss_c.asnumpy(),
                f"{key}/predictQ": q_eval_a.mean().asnumpy()
            })

        self.policy.soft_update(self.tau)
        return info
