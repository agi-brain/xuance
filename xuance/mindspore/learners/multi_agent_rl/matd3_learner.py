"""
Multi-Agent TD3
"""
from mindspore.nn import MSELoss
from xuance.mindspore import ms, Module, Tensor, optim, ops
from xuance.mindspore.learners import LearnerMAS
from xuance.mindspore.utils import clip_grads
from xuance.common import List
from argparse import Namespace
from operator import itemgetter


class MATD3_Learner(LearnerMAS):
    def __init__(self,
                 config: Namespace,
                 model_keys: List[str],
                 agent_keys: List[str],
                 policy: Module):
        super(MATD3_Learner, self).__init__(config, model_keys, agent_keys, policy)
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
        self.actor_update_delay = config.actor_update_delay
        # Get gradient function
        self.grad_fn_actor = {key: ms.value_and_grad(self.forward_fn_actor, None,
                                                     self.optimizer[key]['actor'].parameters, has_aux=True)
                              for key in self.model_keys}
        self.grad_fn_critic = {key: ms.value_and_grad(self.forward_fn_critic, None,
                                                      self.optimizer[key]['critic'].parameters, has_aux=True)
                               for key in self.model_keys}
        self.policy.set_train()

    def forward_fn_actor(self, batch_size, obs, obs_joint, actions, ids, mask_values, agent_key):
        _, actions_eval = self.policy(observation=obs, agent_ids=ids)
        if self.use_parameter_sharing:
            act_eval = actions_eval[agent_key].reshape(batch_size, self.n_agents, -1).reshape(batch_size, -1)
        else:
            a_joint = {k: actions_eval[k] if k == agent_key else actions[k] for k in self.agent_keys}
            act_eval = ops.cat(itemgetter(*self.agent_keys)(a_joint), axis=-1).reshape(batch_size, -1)
        _, _, q_policy = self.policy.Qpolicy(joint_observation=obs_joint, joint_actions=act_eval, agent_ids=ids,
                                             agent_key=agent_key)
        q_policy_i = q_policy[agent_key].reshape(-1)
        loss_a = -(q_policy_i * mask_values).sum() / mask_values.sum()
        return loss_a, q_policy_i

    def forward_fn_critic(self, obs_joint, actions_joint, ids, mask_values, q_target, agent_key):
        q_eval_A, q_eval_B, _ = self.policy.Qpolicy(joint_observation=obs_joint, joint_actions=actions_joint,
                                                    agent_ids=ids)
        q_eval_A_i, q_eval_B_i = q_eval_A[agent_key].reshape(-1), q_eval_B[agent_key].reshape(-1)
        td_error_A = (q_eval_A_i - ops.stop_gradient(q_target)) * mask_values
        td_error_B = (q_eval_B_i - ops.stop_gradient(q_target)) * mask_values
        loss_c = ((td_error_A ** 2).sum() + (td_error_B ** 2).sum()) / mask_values.sum()
        return loss_c, q_eval_A_i, q_eval_B_i

    def update(self, sample):
        self.iterations += 1
        info = {}

        # prepare training data
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

        # get values
        _, actions_next = self.policy.Atarget(next_observation=obs_next, agent_ids=IDs)
        if self.use_parameter_sharing:
            key = self.model_keys[0]
            actions_next_joint = actions_next[key].reshape(batch_size, self.n_agents, -1).reshape(batch_size, -1)
        else:
            actions_next_joint = ops.cat(itemgetter(*self.model_keys)(actions_next), axis=-1).reshape(batch_size, -1)

        q_next = self.policy.Qtarget(joint_observation=next_obs_joint, joint_actions=actions_next_joint, agent_ids=IDs)

        # update critic(s)
        for key in self.model_keys:
            mask_values = agent_mask[key]
            q_next_i = q_next[key].reshape(bs)
            q_target = rewards[key] + (1 - terminals[key]) * self.gamma * q_next_i
            (loss_c, q_eval_A_i, q_eval_B_i), grads_critic = self.grad_fn_critic[key](obs_joint, actions_joint, IDs,
                                                                                      mask_values, q_target, key)
            if self.use_grad_clip:
                grads_critic = clip_grads(grads_critic, Tensor(-self.grad_clip_norm), Tensor(self.grad_clip_norm))
            self.optimizer[key]['critic'](grads_critic)

            self.scheduler[key]['critic'].step()
            learning_rate_critic = self.scheduler[key]['critic'].get_last_lr()[0]

            info.update({
                f"{key}/learning_rate_critic": learning_rate_critic.asnumpy(),
                f"{key}/loss_critic": loss_c.asnumpy(),
                f"{key}/predictQ_A": q_eval_A_i.mean().asnumpy(),
                f"{key}/predictQ_B": q_eval_B_i.mean().asnumpy()
            })

        # update actor(s)
        if self.iterations % self.actor_update_delay == 0:
            for key in self.model_keys:
                mask_values = agent_mask[key]
                # update actor
                (loss_a, q_policy_i), grads_actor = self.grad_fn_actor[key](batch_size, obs, obs_joint, actions,
                                                                            IDs, mask_values, key)
                if self.use_grad_clip:
                    grads_actor = clip_grads(grads_actor, Tensor(-self.grad_clip_norm), Tensor(self.grad_clip_norm))
                self.optimizer[key]['actor'](grads_actor)

                self.scheduler[key]['actor'].step()
                learning_rate_actor = self.scheduler[key]['actor'].get_last_lr()[0]

                info.update({
                    f"{key}/learning_rate_actor": learning_rate_actor.asnumpy(),
                    f"{key}/loss_actor": loss_a.asnumpy(),
                    f"{key}/q_policy": q_policy_i.mean().asnumpy(),
                })
            self.policy.soft_update(self.tau)

        return info
