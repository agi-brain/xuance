"""
Multi-Agent Deep Deterministic Policy Gradient
Paper link:
https://proceedings.neurips.cc/paper/2017/file/68a9750337a418a86fe06c1991a1d64c-Paper.pdf
Implementation: Pytorch
Trick: Parameter sharing for all agents, with agents' one-hot IDs as actor-critic's inputs.
"""
import torch
from torch import nn
from xuance.torch.learners import LearnerMAS
from xuance.common import AgentGrouping
from argparse import Namespace


class MADDPG_Learner(LearnerMAS):
    def __init__(self,
                 config: Namespace,
                 agent_grouping: AgentGrouping,
                 model: nn.Module,
                 callback):
        super(MADDPG_Learner, self).__init__(config, agent_grouping, model, callback)
        self.optimizer = {
            key: {
                'actor': torch.optim.Adam(self.model.actors[key].parameters(),
                                          self.config.learning_rate_actor, eps=1e-5),
                'critic': torch.optim.Adam(self.model.critics[key].parameters(),
                                           self.config.learning_rate_critic, eps=1e-5)}
            for key in self.group_keys}
        self.scheduler = {
            key: {'actor': torch.optim.lr_scheduler.LinearLR(self.optimizer[key]['actor'],
                                                             start_factor=1.0,
                                                             end_factor=self.end_factor_lr_decay,
                                                             total_iters=self.total_iters),
                  'critic': torch.optim.lr_scheduler.LinearLR(self.optimizer[key]['critic'],
                                                              start_factor=1.0,
                                                              end_factor=self.end_factor_lr_decay,
                                                              total_iters=self.total_iters)}
            for key in self.group_keys}
        self.gamma = config.gamma
        self.tau = config.tau
        self.mse_loss = nn.MSELoss()

    def update(self, sample):
        self.iterations += 1

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
        agent_indices = sample_Tensor['agent_indices']

        obs_joint = self.get_joint_input(obs, (batch_size, -1))
        next_obs_joint = self.get_joint_input(obs_next, (batch_size, -1))
        actions_joint = self.get_joint_input(actions, (batch_size, -1))

        if not self.agent_grouping.full_independent:
            obs = self.packed_tensor(obs)
            obs_next = self.packed_tensor(obs_next)

        info = self.callback.on_update_start(self.iterations, method="update",
                                             model=self.model, sample_Tensor=sample_Tensor,
                                             obs_joint=obs_joint, next_obs_joint=next_obs_joint,
                                             actions_joint=actions_joint)

        # get actions
        actions_eval = self.model(observations=obs, agent_indices=agent_indices).actions
        actions_next = self.model.Atarget(observations=obs_next, agent_indices=agent_indices)
        # get values
        actions_next_joint = self.get_joint_input(actions_next, (batch_size, -1))
        q_eval = self.model.Qpolicy(joint_observations=obs_joint, joint_actions=actions_joint,
                                    agent_indices=agent_indices)
        q_next = self.model.Qtarget(joint_observations=next_obs_joint, joint_actions=actions_next_joint,
                                    agent_indices=agent_indices)

        for group, agent_keys in self.groups.items():
            # update critic
            loss_c = []
            for key in agent_keys:
                mask_values = agent_mask[key]
                q_eval_a = q_eval[key].reshape(batch_size)
                q_next_i = q_next[key].reshape(batch_size)
                q_target = rewards[key] + (1 - terminals[key]) * self.gamma * q_next_i
                td_error = (q_eval_a - q_target.detach()) * mask_values
                loss_i = (td_error ** 2).sum() / mask_values.sum()
                loss_c.append(loss_i)
            loss_critic = sum(loss_c)
            self.optimizer[group]['critic'].zero_grad()
            loss_critic.backward()
            if self.use_grad_clip:
                torch.nn.utils.clip_grad_norm_(self.model.critics[group].parameters(), self.grad_clip_norm)
            self.optimizer[group]['critic'].step()
            if self.scheduler[group]['critic'] is not None:
                self.scheduler[group]['critic'].step()

            # update actor
            loss_a = []
            a_joint = actions.copy()
            for key in agent_keys:
                mask_values = agent_mask[key]
                original_action = a_joint[key]
                a_joint[key] = actions_eval[key]
                act_eval = self.get_joint_input(a_joint, (batch_size, -1))
                q_model = self.model.Qpolicy(joint_observations=obs_joint, joint_actions=act_eval,
                                             agent_indices=agent_indices, group_key=group)
                q_model_i = q_model[key].reshape(batch_size)
                loss_i = -(q_model_i * mask_values).sum() / mask_values.sum()
                loss_a.append(loss_i)
                a_joint[key] = original_action
            loss_actor = sum(loss_a)
            self.optimizer[group]['actor'].zero_grad()
            loss_actor.backward()
            if self.use_grad_clip:
                torch.nn.utils.clip_grad_norm_(self.model.actors[group].parameters(), self.grad_clip_norm)
            self.optimizer[group]['actor'].step()
            if self.scheduler[group]['actor'] is not None:
                self.scheduler[group]['actor'].step()

            learning_rate_actor = self.optimizer[group]['actor'].state_dict()['param_groups'][0]['lr']
            learning_rate_critic = self.optimizer[group]['critic'].state_dict()['param_groups'][0]['lr']

            info.update({
                f"{group}/learning_rate_actor": learning_rate_actor,
                f"{group}/learning_rate_critic": learning_rate_critic,
                f"{group}/loss_actor": loss_actor.item(),
                f"{group}/loss_critic": loss_critic.item(),
                f"{group}/predictQ": q_eval[agent_keys[0]].mean().item()
            })

            info.update(self.callback.on_update_agent_wise(self.iterations, group, info=info, method="update",
                                                           mask_values=mask_values, q_model_i=q_model_i,
                                                           act_eval=act_eval, q_eval_a=q_eval_a, q_next_i=q_next_i,
                                                           q_target=q_target, td_error=td_error))

        self.model.soft_update(self.tau)
        info.update(self.callback.on_update_end(self.iterations, method="update", model=self.model, info=info))
        return info

    def update_rnn(self, sample):
        self.iterations += 1

        # prepare training data
        sample_Tensor = self.build_training_data(sample=sample,
                                                 use_parameter_sharing=self.use_parameter_sharing,
                                                 use_actions_mask=self.use_actions_mask)
        batch_size = sample_Tensor['batch_size']
        seq_len = sample_Tensor['seq_length']
        obs = sample_Tensor['obs']
        actions = sample_Tensor['actions']
        rewards = sample_Tensor['rewards']
        terminals = sample_Tensor['terminals']
        agent_mask = sample_Tensor['agent_mask']
        filled = sample_Tensor['filled']
        agent_indices = sample_Tensor['agent_indices']

        obs_joint = self.get_joint_input(obs, (batch_size, seq_len + 1, -1))
        actions_joint = self.get_joint_input(actions, (batch_size, seq_len, -1))

        if not self.agent_grouping.full_independent:
            obs = self.packed_tensor(obs)

        info = self.callback.on_update_start(self.iterations, method="update_rnn",
                                             model=self.model, sample_Tensor=sample_Tensor,
                                             obs_joint=obs_joint, actions_joint=actions_joint)

        # initial hidden states for rnn
        rnn_states_actor = self.model.init_actor_rnn_states(batch_size)
        rnn_states_critic = self.model.init_critic_rnn_states(batch_size)

        # get actions
        actions_eval = self.model(observations=obs, agent_indices=agent_indices, rnn_states=rnn_states_actor).actions
        actions_next = self.model.Atarget(observations=obs, agent_indices=agent_indices, rnn_states=rnn_states_actor)

        # get q values
        actions_next_joint = self.get_joint_input(actions_next, (batch_size, seq_len + 1, -1))
        agent_indices_t = {k: v[:, :-1] for k, v in agent_indices.items()}
        q_eval = self.model.Qpolicy(joint_observations=obs_joint[:, :-1], joint_actions=actions_joint,
                                    agent_indices=agent_indices_t, rnn_states=rnn_states_critic)
        q_next = self.model.Qtarget(joint_observations=obs_joint, joint_actions=actions_next_joint,
                                    agent_indices=agent_indices, rnn_states=rnn_states_critic)

        for group, agent_keys in self.groups.items():
            # update critic
            loss_c = []
            for key in agent_keys:
                mask_values = agent_mask[key] * filled
                q_eval_a = q_eval[key].reshape(batch_size, seq_len)
                q_next_i = q_next[key][:, 1:].reshape(batch_size, seq_len)
                q_target = rewards[key] + (1 - terminals[key]) * self.gamma * q_next_i
                td_error = (q_eval_a - q_target.detach()) * mask_values
                loss_i = (td_error ** 2).sum() / mask_values.sum()
                loss_c.append(loss_i)
            loss_critic = sum(loss_c)
            self.optimizer[group]['critic'].zero_grad()
            loss_critic.backward()
            if self.use_grad_clip:
                torch.nn.utils.clip_grad_norm_(self.model.critics[group].parameters(), self.grad_clip_norm)
            self.optimizer[group]['critic'].step()
            if self.scheduler[group]['critic'] is not None:
                self.scheduler[group]['critic'].step()

            # update actor
            loss_a = []
            a_joint = actions.copy()
            for key in agent_keys:
                mask_values = agent_mask[key] * filled
                original_action = a_joint[key]
                a_joint[key] = actions_eval[key][:, :-1]
                act_eval = self.get_joint_input(a_joint, (batch_size, seq_len, -1))
                q_model = self.model.Qpolicy(joint_observations=obs_joint[:, :-1], joint_actions=act_eval,
                                             agent_indices=agent_indices_t, group_key=group,
                                             rnn_states=rnn_states_critic)
                q_model_i = q_model[key].reshape(batch_size, seq_len)
                loss_i = -(q_model_i * mask_values).sum() / mask_values.sum()
                loss_a.append(loss_i)
                a_joint[key] = original_action
            loss_actor = sum(loss_a)
            self.optimizer[group]['actor'].zero_grad()
            loss_actor.backward()
            if self.use_grad_clip:
                torch.nn.utils.clip_grad_norm_(self.model.actors[group].parameters(), self.grad_clip_norm)
            self.optimizer[group]['actor'].step()
            if self.scheduler[group]['actor'] is not None:
                self.scheduler[group]['actor'].step()

            learning_rate_actor = self.optimizer[group]['actor'].state_dict()['param_groups'][0]['lr']
            learning_rate_critic = self.optimizer[group]['critic'].state_dict()['param_groups'][0]['lr']

            info.update({
                f"{group}/learning_rate_actor": learning_rate_actor,
                f"{group}/learning_rate_critic": learning_rate_critic,
                f"{group}/loss_actor": loss_actor.item(),
                f"{group}/loss_critic": loss_critic.item(),
                f"{group}/predictQ": q_eval[agent_keys[0]].mean().item()
            })

            info.update(self.callback.on_update_agent_wise(self.iterations, group, info=info, method="update_rnn",
                                                           mask_values=mask_values, q_model_i=q_model_i,
                                                           q_eval_a=q_eval_a, q_next_i=q_next_i,
                                                           q_target=q_target, td_error=td_error))

        self.model.soft_update(self.tau)
        info.update(self.callback.on_update_end(self.iterations, method="update_rnn", model=self.model, info=info))
        return info
