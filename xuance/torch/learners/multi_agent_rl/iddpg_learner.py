"""
Independent Deep Deterministic Policy Gradient (IDDPG)
Implementation: Pytorch
"""
import torch
from torch import nn
from xuance.torch.learners import LearnerMAS
from xuance.common import AgentGrouping
from argparse import Namespace


class IDDPG_Learner(LearnerMAS):
    def __init__(self,
                 config: Namespace,
                 agent_grouping: AgentGrouping,
                 model: nn.Module,
                 callback):
        super(IDDPG_Learner, self).__init__(config, agent_grouping, model, callback)
        self.optimizer = {
            key: {'actor': torch.optim.Adam(self.model.actors[key].parameters(),
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
        agent_indices = sample_Tensor['agent_indices']

        if not self.agent_grouping.full_independent:
            obs = self.packed_tensor(obs)
            actions = self.packed_tensor(actions)
            obs_next = self.packed_tensor(obs_next)

        info = self.callback.on_update_start(self.iterations, method="update",
                                             model=self.model, sample_Tensor=sample_Tensor)

        # feedforward
        actions_eval = self.model(observations=obs, agent_indices=agent_indices).actions
        if not self.agent_grouping.full_independent:
            actions_eval = self.packed_tensor(actions_eval)
        q_model = self.model.Qpolicy(observations=obs, actions=actions_eval, agent_indices=agent_indices)
        q_eval = self.model.Qpolicy(observations=obs, actions=actions, agent_indices=agent_indices)
        next_actions = self.model.Atarget(observations=obs_next, agent_indices=agent_indices)
        if not self.agent_grouping.full_independent:
            next_actions = self.packed_tensor(next_actions)
        q_next = self.model.Qtarget(observations=obs_next, actions=next_actions, agent_indices=agent_indices)

        for group, agent_keys in self.groups.items():
            # update actor
            loss_a = []
            for key in agent_keys:
                mask_values = agent_mask[key]
                q_model_i = q_model[key].reshape(batch_size)
                loss_i = -(q_model_i * mask_values).sum() / mask_values.sum()
                loss_a.append(loss_i)
            loss_actor = sum(loss_a)
            self.optimizer[group]['actor'].zero_grad()
            loss_actor.backward()
            if self.use_grad_clip:
                torch.nn.utils.clip_grad_norm_(self.model.parameters_actor[group], self.grad_clip_norm)
            self.optimizer[group]['actor'].step()
            if self.scheduler[group]['actor'] is not None:
                self.scheduler[group]['actor'].step()

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
                torch.nn.utils.clip_grad_norm_(self.model.parameters_critic[group], self.grad_clip_norm)
            self.optimizer[group]['critic'].step()
            if self.scheduler[group]['critic'] is not None:
                self.scheduler[group]['critic'].step()

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
                                                           q_eval_a=q_eval_a, q_next_i=q_next_i,
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

        if not self.agent_grouping.full_independent:
            obs = self.packed_tensor(obs)
            actions = self.packed_tensor(actions)

        info = self.callback.on_update_start(self.iterations, method="update_rnn",
                                             model=self.model, sample_Tensor=sample_Tensor)

        # feedforward
        rnn_states_actor = self.model.init_actor_rnn_states(batch_size)
        rnn_states_critic = self.model.init_critic_rnn_states(batch_size)

        actions_eval = self.model(observations=obs, agent_indices=agent_indices, rnn_states=rnn_states_actor).actions
        if not self.agent_grouping.full_independent:
            actions_eval = self.packed_tensor(actions_eval)
        q_model = self.model.Qpolicy(observations=obs, actions=actions_eval, agent_indices=agent_indices,
                                     rnn_states=rnn_states_critic)
        obs_t = {k: v[:, :-1] for k, v in obs.items()}
        agent_indices_t = {k: v[:, :-1] for k, v in agent_indices.items()}
        q_eval = self.model.Qpolicy(observations=obs_t, actions=actions, agent_indices=agent_indices_t,
                                    rnn_states=rnn_states_critic)
        next_actions = self.model.Atarget(observations=obs, agent_indices=agent_indices,
                                          rnn_states=rnn_states_actor)
        if not self.agent_grouping.full_independent:
            next_actions = self.packed_tensor(next_actions)
        q_next = self.model.Qtarget(observations=obs, actions=next_actions, agent_indices=agent_indices,
                                    rnn_states=rnn_states_critic)

        for group, agent_keys in self.groups.items():
            # update actor
            loss_a = []
            for key in agent_keys:
                mask_values = agent_mask[key] * filled
                q_model_i = q_model[key][:, :-1].reshape(batch_size, seq_len)
                loss_i = -(q_model_i * mask_values).sum() / mask_values.sum()
                loss_a.append(loss_i)
            loss_actor = sum(loss_a)
            self.optimizer[group]['actor'].zero_grad()
            loss_actor.backward()
            if self.use_grad_clip:
                torch.nn.utils.clip_grad_norm_(self.model.parameters_actor[group], self.grad_clip_norm)
            self.optimizer[group]['actor'].step()
            if self.scheduler[group]['actor'] is not None:
                self.scheduler[group]['actor'].step()

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
                torch.nn.utils.clip_grad_norm_(self.model.parameters_critic[group], self.grad_clip_norm)
            self.optimizer[group]['critic'].step()
            if self.scheduler[group]['critic'] is not None:
                self.scheduler[group]['critic'].step()

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
