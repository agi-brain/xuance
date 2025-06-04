"""
Multi-Agent TD3
"""
import torch
from torch import nn
from xuance.common import List
from argparse import Namespace
from xuance.torch.learners import LearnerMAS


class MATD3_Learner(LearnerMAS):
    def __init__(self,
                 config: Namespace,
                 model_keys: List[str],
                 agent_keys: List[str],
                 policy: nn.Module,
                 callback):
        super(MATD3_Learner, self).__init__(config, model_keys, agent_keys, policy, callback)
        self.optimizer = {
            key: {'actor': torch.optim.Adam(self.policy.parameters_actor[key], self.config.learning_rate_actor, eps=1e-5),
                  'critic': torch.optim.Adam(self.policy.parameters_critic[key], self.config.learning_rate_critic, eps=1e-5)}
            for key in self.model_keys}
        self.scheduler = {
            key: {'actor': torch.optim.lr_scheduler.LinearLR(self.optimizer[key]['actor'],
                                                             start_factor=1.0,
                                                             end_factor=self.end_factor_lr_decay,
                                                             total_iters=self.config.running_steps),
                  'critic': torch.optim.lr_scheduler.LinearLR(self.optimizer[key]['critic'],
                                                              start_factor=1.0,
                                                              end_factor=self.end_factor_lr_decay,
                                                              total_iters=self.config.running_steps)}
            for key in self.model_keys}
        self.gamma = config.gamma
        self.tau = config.tau
        self.mse_loss = nn.MSELoss()
        self.actor_update_delay = config.actor_update_delay

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
            obs_joint = self.get_joint_input(obs, (batch_size, -1))
            next_obs_joint = self.get_joint_input(obs_next, (batch_size, -1))
            actions_joint = self.get_joint_input(actions, (batch_size, -1))

        info = self.callback.on_update_start(self.iterations, method="update", policy=self.policy,
                                             sample_Tensor=sample_Tensor, bs=bs, obs_joint=obs_joint,
                                             next_obs_joint=next_obs_joint, actions_joint=actions_joint)

        # get values
        _, actions_next = self.policy.Atarget(next_observation=obs_next, agent_ids=IDs)
        if self.use_parameter_sharing:
            key = self.model_keys[0]
            actions_next_joint = actions_next[key].reshape(batch_size, self.n_agents, -1).reshape(batch_size, -1)
        else:
            actions_next_joint = self.get_joint_input(actions_next, (batch_size, -1))
        q_eval_A, q_eval_B, _ = self.policy.Qpolicy(joint_observation=obs_joint, joint_actions=actions_joint,
                                                    agent_ids=IDs)
        q_next = self.policy.Qtarget(joint_observation=next_obs_joint, joint_actions=actions_next_joint, agent_ids=IDs)

        # update critic(s)
        for key in self.model_keys:
            mask_values = agent_mask[key]
            q_eval_A_i, q_eval_B_i = q_eval_A[key].reshape(bs), q_eval_B[key].reshape(bs)
            q_next_i = q_next[key].reshape(bs)
            q_target = rewards[key] + (1 - terminals[key]) * self.gamma * q_next_i
            td_error_A = (q_eval_A_i - q_target.detach()) * mask_values
            td_error_B = (q_eval_B_i - q_target.detach()) * mask_values
            loss_c = ((td_error_A ** 2).sum() + (td_error_B ** 2).sum()) / mask_values.sum()
            self.optimizer[key]['critic'].zero_grad()
            loss_c.backward()
            if self.use_grad_clip:
                torch.nn.utils.clip_grad_norm_(self.policy.parameters_critic[key], self.grad_clip_norm)
            self.optimizer[key]['critic'].step()
            if self.scheduler[key]['critic'] is not None:
                self.scheduler[key]['critic'].step()

            learning_rate_critic = self.optimizer[key]['critic'].state_dict()['param_groups'][0]['lr']

            info.update({
                f"{key}/learning_rate_critic": learning_rate_critic,
                f"{key}/loss_critic": loss_c.item(),
                f"{key}/predictQ_A": q_eval_A[key].mean().item(),
                f"{key}/predictQ_B": q_eval_B[key].mean().item()
            })

            info.update(self.callback.on_update_agent_wise(self.iterations, key, info=info, method="update_critic",
                                                           mask_values=mask_values,
                                                           q_eval_A_i=q_eval_A_i, q_eval_B_i=q_eval_B_i,
                                                           q_target=q_target, q_next_i=q_next_i,
                                                           td_error_A=td_error_A, td_error_B=td_error_B))

        # update actor(s)
        if self.iterations % self.actor_update_delay == 0:
            _, actions_eval = self.policy(observation=obs, agent_ids=IDs)
            for key in self.model_keys:
                mask_values = agent_mask[key]
                if self.use_parameter_sharing:
                    act_eval = actions_eval[key].reshape(batch_size, self.n_agents, -1).reshape(batch_size, -1)
                else:
                    a_joint = {k: actions_eval[k] if k == key else actions[k] for k in self.agent_keys}
                    act_eval = self.get_joint_input(a_joint, (batch_size, -1))
                _, _, q_policy = self.policy.Qpolicy(joint_observation=obs_joint, joint_actions=act_eval, agent_ids=IDs,
                                                     agent_key=key)
                q_policy_i = q_policy[key].reshape(bs)
                loss_a = -(q_policy_i * mask_values).sum() / mask_values.sum()
                self.optimizer[key]['actor'].zero_grad()
                loss_a.backward()
                if self.use_grad_clip:
                    torch.nn.utils.clip_grad_norm_(self.policy.parameters_actor[key], self.grad_clip_norm)
                self.optimizer[key]['actor'].step()
                if self.scheduler[key]['actor'] is not None:
                    self.scheduler[key]['actor'].step()

                learning_rate_actor = self.optimizer[key]['actor'].state_dict()['param_groups'][0]['lr']

                info.update({
                    f"{key}/learning_rate_actor": learning_rate_actor,
                    f"{key}/loss_actor": loss_a.item(),
                    f"{key}/q_policy": q_policy_i.mean().item(),
                })
                info.update(self.callback.on_update_agent_wise(self.iterations, key, info=info, method="update_actor",
                                                               mask_values=mask_values, q_policy_i=q_policy_i))
            self.policy.soft_update(self.tau)

        info.update(self.callback.on_update_end(self.iterations, method="update", policy=self.policy, info=info))
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
        IDs = sample_Tensor['agent_ids']

        if self.use_parameter_sharing:
            key = self.model_keys[0]
            bs_rnn = batch_size * self.n_agents
            filled = filled.unsqueeze(1).expand(-1, self.n_agents, -1).reshape(bs_rnn, seq_len)
            obs_joint = obs[key].reshape(batch_size, self.n_agents, seq_len + 1, -1).transpose(
                1, 2).reshape(batch_size, seq_len + 1, -1)
            actions_joint = actions[key].reshape(batch_size, self.n_agents, seq_len, -1).transpose(
                1, 2).reshape(batch_size, seq_len, -1)
            rewards[key] = rewards[key].reshape(bs_rnn, seq_len)
            terminals[key] = terminals[key].reshape(bs_rnn, seq_len)
            IDs_t = IDs[:, :-1]
        else:
            bs_rnn, IDs_t = batch_size, None
            obs_joint = self.get_joint_input(obs, (batch_size, seq_len + 1, -1))
            actions_joint = self.get_joint_input(actions, (batch_size, seq_len, -1))

        info = self.callback.on_update_start(self.iterations, method="update_rnn", policy=self.policy,
                                             sample_Tensor=sample_Tensor, bs_rnn=bs_rnn,
                                             obs_joint=obs_joint, actions_joint=actions_joint)

        # initial hidden states for rnn
        rnn_hidden_actor = {k: self.policy.actor_representation[k].init_hidden(bs_rnn) for k in self.model_keys}
        rnn_hidden_critic = {k: self.policy.critic_A_representation[k].init_hidden(batch_size) for k in self.model_keys}

        # get q values
        _, actions_next = self.policy.Atarget(next_observation=obs, agent_ids=IDs, rnn_hidden=rnn_hidden_actor)
        if self.use_parameter_sharing:
            next_actions_joint = actions_next[self.model_keys[0]].reshape(
                batch_size, self.n_agents, seq_len + 1, -1).transpose(1, 2).reshape(batch_size, seq_len + 1, -1)
        else:
            next_actions_joint = self.get_joint_input(actions_next, (batch_size, seq_len + 1, -1))
        q_eval_A, q_eval_B, _ = self.policy.Qpolicy(joint_observation=obs_joint[:, :-1], joint_actions=actions_joint,
                                                    agent_ids=IDs_t, rnn_hidden=rnn_hidden_critic)
        q_next = self.policy.Qtarget(joint_observation=obs_joint, joint_actions=next_actions_joint, agent_ids=IDs,
                                     rnn_hidden=rnn_hidden_critic)

        # update critic(s)
        for key in self.model_keys:
            mask_values = agent_mask[key] * filled
            q_eval_A_i, q_eval_B_i = q_eval_A[key].reshape(bs_rnn, seq_len), q_eval_B[key].reshape(bs_rnn, seq_len)
            q_next_i = q_next[key][:, 1:].reshape(bs_rnn, seq_len)
            q_target = rewards[key] + (1 - terminals[key]) * self.gamma * q_next_i
            td_error_A = (q_eval_A_i - q_target.detach()) * mask_values
            td_error_B = (q_eval_B_i - q_target.detach()) * mask_values
            loss_c = ((td_error_A ** 2).sum() + (td_error_B ** 2).sum()) / mask_values.sum()
            self.optimizer[key]['critic'].zero_grad()
            loss_c.backward()
            if self.use_grad_clip:
                torch.nn.utils.clip_grad_norm_(self.policy.parameters_critic[key], self.grad_clip_norm)
            self.optimizer[key]['critic'].step()
            if self.scheduler[key]['critic'] is not None:
                self.scheduler[key]['critic'].step()

            learning_rate_critic = self.optimizer[key]['critic'].state_dict()['param_groups'][0]['lr']

            info.update({
                f"{key}/learning_rate_critic": learning_rate_critic,
                f"{key}/loss_critic": loss_c.item(),
                f"{key}/predictQ_A": q_eval_A[key].mean().item(),
                f"{key}/predictQ_B": q_eval_B[key].mean().item()
            })

            info.update(self.callback.on_update_agent_wise(self.iterations, key, info=info, method="update_rnn_critic",
                                                           mask_values=mask_values,
                                                           q_eval_A_i=q_eval_A_i, q_eval_B_i=q_eval_B_i,
                                                           q_target=q_target, q_next_i=q_next_i,
                                                           td_error_A=td_error_A, td_error_B=td_error_B))

        # update actor(s)
        if self.iterations % self.actor_update_delay == 0:
            _, actions_eval = self.policy(observation=obs, agent_ids=IDs, rnn_hidden=rnn_hidden_actor)
            for key in self.model_keys:
                mask_values = agent_mask[key] * filled
                if self.use_parameter_sharing:
                    act_eval = actions_eval[key][:, :-1].reshape(
                        batch_size, self.n_agents, seq_len, -1).transpose(1, 2).reshape(batch_size, seq_len, -1)
                else:
                    a_joint = {k: actions_eval[k][:, :-1] if k == key else actions[k] for k in self.agent_keys}
                    act_eval = self.get_joint_input(a_joint, (batch_size, seq_len, -1))
                _, _, q_policy = self.policy.Qpolicy(joint_observation=obs_joint[:, :-1], joint_actions=act_eval,
                                                     agent_key=key, agent_ids=IDs_t, rnn_hidden=rnn_hidden_critic)
                q_policy_i = q_policy[key].reshape(bs_rnn, seq_len)
                loss_a = -(q_policy_i * mask_values).sum() / mask_values.sum()
                self.optimizer[key]['actor'].zero_grad()
                loss_a.backward()
                if self.use_grad_clip:
                    torch.nn.utils.clip_grad_norm_(self.policy.parameters_actor[key], self.grad_clip_norm)
                self.optimizer[key]['actor'].step()
                if self.scheduler[key]['actor'] is not None:
                    self.scheduler[key]['actor'].step()

                learning_rate_actor = self.optimizer[key]['actor'].state_dict()['param_groups'][0]['lr']

                info.update({
                    f"{key}/learning_rate_actor": learning_rate_actor,
                    f"{key}/loss_actor": loss_a.item(),
                    f"{key}/q_policy": q_policy_i.mean().item(),
                })
                info.update(self.callback.on_update_agent_wise(self.iterations, key, info=info, method="update_rnn_actor",
                                                               mask_values=mask_values, q_policy_i=q_policy_i))
            self.policy.soft_update(self.tau)
        info.update(self.callback.on_update_end(self.iterations, method="update_rnn", policy=self.policy, info=info))
        return info
