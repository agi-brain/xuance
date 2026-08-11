"""
Independent TD3 for multi-agent cooperative task
"""
import torch
from torch import nn
from xuance.common import AgentGrouping
from argparse import Namespace
from xuance.torch.learners import LearnerMAS


class ITD3_Learner(LearnerMAS):
    def __init__(self,
                 config: Namespace,
                 agent_grouping: AgentGrouping,
                 model: nn.Module,
                 callback):
        super(ITD3_Learner, self).__init__(config, agent_grouping, model, callback)
        self.optimizer = {
            key: {'actor': torch.optim.Adam(self.model.actors[key].parameters(), self.config.learning_rate_actor,
                                            eps=1e-5),
                  'critic': torch.optim.Adam(self.model.critics[key].parameters(), self.config.learning_rate_critic,
                                             eps=1e-5)}
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
        agent_indices = sample_Tensor['agent_indices']

        if not self.agent_grouping.full_independent:
            obs = self.packed_tensor(obs)
            actions = self.packed_tensor(actions)
            obs_next = self.packed_tensor(obs_next)
            rewards = self.packed_tensor(rewards)
            terminals = self.packed_tensor(terminals)
            agent_mask = self.packed_tensor(agent_mask)

        info = self.callback.on_update_start(self.iterations, method="update",
                                             model=self.model, sample_Tensor=sample_Tensor)

        # get values
        q_eval_A, q_eval_B, _ = self.model.Qpolicy(observations=obs, actions=actions, agent_indices=agent_indices)
        next_actions = self.model.Atarget(observations=obs_next, agent_indices=agent_indices)
        if not self.agent_grouping.full_independent:
            next_actions = self.packed_tensor(next_actions)

        q_next = self.model.Qtarget(observations=obs_next, actions=next_actions, agent_indices=agent_indices)
        if not self.agent_grouping.full_independent:
            q_eval_A = self.packed_tensor(q_eval_A)
            q_eval_B = self.packed_tensor(q_eval_B)
            q_next = self.packed_tensor(q_next)

        # update critic(s)
        for group, agent_keys in self.groups.items():
            bs = batch_size * self.n_group_agents[group]
            mask_values = agent_mask[group]
            q_eval_A_i, q_eval_B_i = q_eval_A[group].reshape(bs), q_eval_B[group].reshape(bs)
            q_next_i = q_next[group].reshape(bs)
            q_target = rewards[group] + (1 - terminals[group]) * self.gamma * q_next_i
            td_error_A = (q_eval_A_i - q_target.detach()) * mask_values
            td_error_B = (q_eval_B_i - q_target.detach()) * mask_values
            loss_c = ((td_error_A ** 2).sum() + (td_error_B ** 2).sum()) / mask_values.sum()
            self.optimizer[group]['critic'].zero_grad()
            loss_c.backward()
            if self.use_grad_clip:
                torch.nn.utils.clip_grad_norm_(self.model.critics[group].parameters(), self.grad_clip_norm)
            self.optimizer[group]['critic'].step()
            if self.scheduler[group]['critic'] is not None:
                self.scheduler[group]['critic'].step()

            learning_rate_critic = self.optimizer[group]['critic'].state_dict()['param_groups'][0]['lr']

            info.update({
                f"{group}/learning_rate_critic": learning_rate_critic,
                f"{group}/loss_critic": loss_c.item(),
                f"{group}/predictQ_A": q_eval_A[group].mean().item(),
                f"{group}/predictQ_B": q_eval_B[group].mean().item()
            })

            info.update(self.callback.on_update_agent_wise(self.iterations, group, info=info, method="update_critic",
                                                           mask_values=mask_values,
                                                           q_eval_A_i=q_eval_A_i, q_eval_B_i=q_eval_B_i,
                                                           q_target=q_target, q_next_i=q_next_i,
                                                           td_error_A=td_error_A, td_error_B=td_error_B))

        # update actor(s)
        if self.iterations % self.actor_update_delay == 0:
            actions_eval = self.model(observations=obs, agent_indices=agent_indices).actions
            if not self.agent_grouping.full_independent:
                actions_eval = self.packed_tensor(actions_eval)
            for group, agent_keys in self.groups.items():
                bs = batch_size * self.n_group_agents[group]
                mask_values = agent_mask[group]
                _, _, q_policy = self.model.Qpolicy(observations=obs, actions=actions_eval, agent_indices=agent_indices,
                                                    grou_key=group)
                if not self.agent_grouping.full_independent:
                    q_policy = self.packed_tensor(q_policy)

                q_policy_i = q_policy[group].reshape(bs)
                loss_a = -(q_policy_i * mask_values).sum() / mask_values.sum()
                self.optimizer[group]['actor'].zero_grad()
                loss_a.backward()
                if self.use_grad_clip:
                    torch.nn.utils.clip_grad_norm_(self.model.actors[group].parameters(), self.grad_clip_norm)
                self.optimizer[group]['actor'].step()
                if self.scheduler[group]['actor'] is not None:
                    self.scheduler[group]['actor'].step()

                learning_rate_actor = self.optimizer[group]['actor'].state_dict()['param_groups'][0]['lr']

                info.update({
                    f"{group}/learning_rate_actor": learning_rate_actor,
                    f"{group}/loss_actor": loss_a.item(),
                    f"{group}/q_policy": q_policy_i.mean().item(),
                })
                info.update(self.callback.on_update_agent_wise(self.iterations, group, info=info, method="update_actor",
                                                               mask_values=mask_values, q_policy_i=q_policy_i))
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

        if not self.agent_grouping.full_independent:
            obs = self.packed_tensor(obs)
            actions = self.packed_tensor(actions)
            rewards = self.packed_tensor(rewards)
            terminals = self.packed_tensor(terminals)
            agent_mask = self.packed_tensor(agent_mask)

        info = self.callback.on_update_start(self.iterations, method="update_rnn",
                                             model=self.model, sample_Tensor=sample_Tensor)

        # feedforward
        rnn_states_actor = self.model.init_actor_rnn_states(batch_size)
        rnn_states_critic_1, rnn_states_critic_2 = self.model.init_critic_rnn_states(batch_size)

        # get q values
        next_actions = self.model.Atarget(observations=obs, agent_indices=agent_indices,
                                          rnn_states=rnn_states_actor)
        if not self.agent_grouping.full_independent:
            next_actions = self.packed_tensor(next_actions)

        obs_t = {k: v[:, :-1] for k, v in obs.items()}
        agent_indices_t = {k: v[:, :-1] for k, v in agent_indices.items()}
        q_eval_A, q_eval_B, _ = self.model.Qpolicy(observations=obs_t, actions=actions, agent_indices=agent_indices_t,
                                                   rnn_states_1=rnn_states_critic_1,
                                                   rnn_states_2=rnn_states_critic_2)
        q_next = self.model.Qtarget(observations=obs, actions=next_actions, agent_indices=agent_indices,
                                    rnn_states_1=rnn_states_critic_1,
                                    rnn_states_2=rnn_states_critic_2)

        if not self.agent_grouping.full_independent:
            q_eval_A = self.packed_tensor(q_eval_A)
            q_eval_B = self.packed_tensor(q_eval_B)
            q_next = self.packed_tensor(q_next)

        # update critic(s)
        for group, agent_keys in self.groups.items():
            n_agents = len(agent_keys)
            bs = batch_size * n_agents
            group_filled = filled.unsqueeze(1).expand(batch_size, n_agents, seq_len).reshape([bs, seq_len])
            mask_values = agent_mask[group] * group_filled
            q_eval_A_i, q_eval_B_i = q_eval_A[group].reshape(bs, seq_len), q_eval_B[group].reshape(bs, seq_len)
            q_next_i = q_next[group][:, 1:].reshape(bs, seq_len)
            q_target = rewards[group] + (1 - terminals[group]) * self.gamma * q_next_i
            td_error_A = (q_eval_A_i - q_target.detach()) * mask_values
            td_error_B = (q_eval_B_i - q_target.detach()) * mask_values
            loss_c = ((td_error_A ** 2).sum() + (td_error_B ** 2).sum()) / mask_values.sum()
            self.optimizer[group]['critic'].zero_grad()
            loss_c.backward()
            if self.use_grad_clip:
                torch.nn.utils.clip_grad_norm_(self.model.critics[group].parameters(), self.grad_clip_norm)
            self.optimizer[group]['critic'].step()
            if self.scheduler[group]['critic'] is not None:
                self.scheduler[group]['critic'].step()

            learning_rate_critic = self.optimizer[group]['critic'].state_dict()['param_groups'][0]['lr']

            info.update({
                f"{group}/learning_rate_critic": learning_rate_critic,
                f"{group}/loss_critic": loss_c.item(),
                f"{group}/predictQ_A": q_eval_A[group].mean().item(),
                f"{group}/predictQ_B": q_eval_B[group].mean().item()
            })

            info.update(self.callback.on_update_agent_wise(self.iterations, group, info=info, method="update_rnn_critic",
                                                           mask_values=mask_values,
                                                           q_eval_A_i=q_eval_A_i, q_eval_B_i=q_eval_B_i,
                                                           q_target=q_target, q_next_i=q_next_i,
                                                           td_error_A=td_error_A, td_error_B=td_error_B))

        # update actor(s)
        if self.iterations % self.actor_update_delay == 0:
            actions_eval = self.model(observations=obs_t, agent_indices=agent_indices_t,
                                      rnn_states=rnn_states_actor).actions
            if not self.agent_grouping.full_independent:
                actions_eval = self.packed_tensor(actions_eval)
            for group, agent_keys in self.groups.items():
                n_agents = len(agent_keys)
                bs = batch_size * n_agents
                group_filled = filled.unsqueeze(1).expand(batch_size, n_agents, seq_len).reshape([bs, seq_len])
                mask_values = agent_mask[group] * group_filled
                _, _, q_policy = self.model.Qpolicy(observations=obs_t, actions=actions_eval,
                                                    agent_indices=agent_indices_t, group_key=group,
                                                    rnn_states_1=rnn_states_critic_1,
                                                    rnn_states_2=rnn_states_critic_2)
                if not self.agent_grouping.full_independent:
                    q_policy = self.packed_tensor(q_policy)
                q_policy_i = q_policy[group].reshape(bs, seq_len)
                loss_a = -(q_policy_i * mask_values).sum() / mask_values.sum()
                self.optimizer[group]['actor'].zero_grad()
                loss_a.backward()
                if self.use_grad_clip:
                    torch.nn.utils.clip_grad_norm_(self.model.actors[group].parameters(), self.grad_clip_norm)
                self.optimizer[group]['actor'].step()
                if self.scheduler[group]['actor'] is not None:
                    self.scheduler[group]['actor'].step()

                learning_rate_actor = self.optimizer[group]['actor'].state_dict()['param_groups'][0]['lr']

                info.update({
                    f"{group}/learning_rate_actor": learning_rate_actor,
                    f"{group}/loss_actor": loss_a.item(),
                    f"{group}/q_policy": q_policy_i.mean().item(),
                })
                info.update(self.callback.on_update_agent_wise(self.iterations, group, info=info,
                                                               method="update_rnn_actor",
                                                               mask_values=mask_values, q_policy_i=q_policy_i))
            self.model.soft_update(self.tau)
        info.update(self.callback.on_update_end(self.iterations, method="update_rnn", model=self.model, info=info))
        return info
