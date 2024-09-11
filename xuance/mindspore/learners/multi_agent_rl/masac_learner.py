"""
Multi-agent Soft Actor-critic (MASAC)
Implementation: MindSpore
"""
from mindspore.nn import MSELoss
from xuance.mindspore import ms, Module, Tensor, optim, ops
from xuance.mindspore.learners import LearnerMAS
from xuance.mindspore.utils import clip_grads
from xuance.common import List
from argparse import Namespace
from operator import itemgetter


class MASAC_Learner(LearnerMAS):
    def __init__(self,
                 config: Namespace,
                 model_keys: List[str],
                 agent_keys: List[str],
                 policy: Module):
        super(MASAC_Learner, self).__init__(config, model_keys, agent_keys, policy)
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
        self.alpha = {key: config.alpha for key in self.model_keys}
        self.mse_loss = MSELoss()
        self._ones = ops.Ones()
        self.use_automatic_entropy_tuning = config.use_automatic_entropy_tuning
        if self.use_automatic_entropy_tuning:
            self.target_entropy = {key: -policy.action_space[key].shape[-1] for key in self.model_keys}
            self.log_alpha = {key: ms.Parameter(self._ones(1, ms.float32)) for key in self.model_keys}
            self.alpha = {key: ops.exp(self.log_alpha[key]) for key in self.model_keys}
            self.alpha_optimizer = {key: optim.Adam(params=[self.log_alpha[key]], lr=config.learning_rate_actor)
                                    for key in self.model_keys}
            # Get gradient function
            self.grad_fn_alpha = {key: ms.value_and_grad(self.forward_fn_alpha, None,
                                                         self.alpha_optimizer[key].parameters, has_aux=True)
                                  for key in self.model_keys}
        # Get gradient function
        self.grad_fn_actor = {key: ms.value_and_grad(self.forward_fn_actor, None,
                                                     self.optimizer[key]['actor'].parameters, has_aux=True)
                              for key in self.model_keys}
        self.grad_fn_critic = {key: ms.value_and_grad(self.forward_fn_critic, None,
                                                      self.optimizer[key]['critic'].parameters, has_aux=True)
                               for key in self.model_keys}

    def forward_fn_alpha(self, log_pi_eval_i, key):
        alpha_loss = -(self.log_alpha[key] * ops.stop_gradient((log_pi_eval_i + self.target_entropy[key]))).mean()
        return alpha_loss, self.log_alpha[key]

    def forward_fn_actor(self, batch_size, obs, obs_joint, ids, mask_values, agent_key):
        _, actions_eval, log_pi_eval = self.policy(observation=obs, agent_ids=ids)
        if self.use_parameter_sharing:
            actions_eval_joint = actions_eval[agent_key].reshape(batch_size, self.n_agents, -1).reshape(batch_size, -1)
        else:
            actions_eval_detach_others = {k: actions_eval[k] if k == agent_key else ops.stop_gradient(actions_eval[k])
                                          for k in self.model_keys}
            actions_eval_joint = ops.cat(itemgetter(*self.model_keys)(actions_eval_detach_others),
                                         axis=-1).reshape(batch_size, -1)
        _, _, policy_q_1, policy_q_2 = self.policy.Qpolicy(joint_observation=obs_joint,
                                                           joint_actions=actions_eval_joint,
                                                           agent_ids=ids, agent_key=agent_key)
        log_pi_eval_i = log_pi_eval[agent_key].reshape(-1)
        policy_q = ops.minimum(policy_q_1[agent_key], policy_q_2[agent_key]).reshape(-1)
        loss_a = ((self.alpha[agent_key] * log_pi_eval_i - policy_q) * mask_values).sum() / mask_values.sum()
        return loss_a, log_pi_eval[agent_key], policy_q

    def forward_fn_critic(self, obs_joint, actions_joint, ids, mask_values, backup, agent_key):
        _, _, action_q_1, action_q_2 = self.policy.Qaction(joint_observation=obs_joint, joint_actions=actions_joint,
                                                           agent_ids=ids)
        action_q_1_i = action_q_1[agent_key].reshape(-1)
        action_q_2_i = action_q_2[agent_key].reshape(-1)
        td_error_1, td_error_2 = action_q_1_i - ops.stop_gradient(backup), action_q_2_i - ops.stop_gradient(backup)
        td_error_1 *= mask_values
        td_error_2 *= mask_values
        loss_c = ((td_error_1 ** 2).sum() + (td_error_2 ** 2).sum()) / mask_values.sum()
        return loss_c, action_q_1_i, action_q_2_i

    def update(self, sample):
        self.iterations += 1
        info = {}

        # Prepare training data.
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
            obs_joint = obs[key].reshape(batch_size, -1)
            next_obs_joint = obs_next[key].reshape(batch_size, -1)
            actions_joint = actions[key].reshape(batch_size, -1)
            rewards[key] = rewards[key].reshape(batch_size * self.n_agents)
            terminals[key] = terminals[key].reshape(batch_size * self.n_agents)
        else:
            bs = batch_size
            obs_joint = ops.cat(itemgetter(*self.agent_keys)(obs), axis=-1).reshape(batch_size, -1)
            next_obs_joint = ops.cat(itemgetter(*self.agent_keys)(obs_next), axis=-1).reshape(batch_size, -1)
            actions_joint = ops.cat(itemgetter(*self.agent_keys)(actions), axis=-1).reshape(batch_size, -1)

        # train the model
        _, actions_next, log_pi_next = self.policy(observation=obs_next, agent_ids=IDs)
        if self.use_parameter_sharing:
            key = self.model_keys[0]
            actions_next_joint = actions_next[key].reshape(batch_size, self.n_agents, -1).reshape(batch_size, -1)
        else:
            actions_next_joint = ops.cat(itemgetter(*self.model_keys)(actions_next), -1).reshape(batch_size, -1)

        _, _, target_q = self.policy.Qtarget(joint_observation=next_obs_joint, joint_actions=actions_next_joint,
                                             agent_ids=IDs)
        for key in self.model_keys:
            mask_values = agent_mask[key]
            # critic update
            log_pi_next_eval = log_pi_next[key].reshape(bs)
            target_value = target_q[key].reshape(bs) - self.alpha[key] * log_pi_next_eval
            backup = rewards[key] + (1 - terminals[key]) * self.gamma * target_value
            (loss_c, _, _), grads_critic = self.grad_fn_critic[key](obs_joint, actions_joint, IDs, mask_values, backup,
                                                                    key)
            if self.use_grad_clip:
                grads_critic = clip_grads(grads_critic, Tensor(-self.grad_clip_norm), Tensor(self.grad_clip_norm))
            self.optimizer[key]['critic'](grads_critic)

            # update actor
            (loss_a, log_pi_eval_i, policy_q), grads_actor = self.grad_fn_actor[key](batch_size, obs, obs_joint, IDs,
                                                                                     mask_values, key)
            if self.use_grad_clip:
                grads_actor = clip_grads(grads_actor, Tensor(-self.grad_clip_norm), Tensor(self.grad_clip_norm))
            self.optimizer[key]['actor'](grads_actor)

            # automatic entropy tuning
            if self.use_automatic_entropy_tuning:
                (alpha_loss, _), grads_alpha = self.grad_fn_alpha[key](log_pi_eval_i, key)
                self.alpha_optimizer[key](grads_alpha)
                self.alpha[key] = ops.exp(self.log_alpha[key])
            else:
                alpha_loss = 0

            learning_rate_actor = self.scheduler[key]['actor'].get_last_lr()[0]
            learning_rate_critic = self.scheduler[key]['critic'].get_last_lr()[0]

            info.update({
                f"{key}/learning_rate_actor": learning_rate_actor.asnumpy(),
                f"{key}/learning_rate_critic": learning_rate_critic.asnumpy(),
                f"{key}/loss_actor": loss_a.asnumpy(),
                f"{key}/loss_critic": loss_c.asnumpy(),
                f"{key}/predictQ": policy_q.mean().asnumpy(),
                f"{key}/alpha_loss": alpha_loss.asnumpy(),
                f"{key}/alpha": self.alpha[key].asnumpy(),
            })

        self.policy.soft_update(self.tau)
        return info
