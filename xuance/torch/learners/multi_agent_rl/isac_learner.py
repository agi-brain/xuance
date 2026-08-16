"""
Independent Soft Actor-critic (ISAC)
Implementation: Pytorch
"""
import torch
from torch import nn
from argparse import Namespace
from xuance.common import AgentGrouping
from xuance.torch.learners import OffPolicyMultiAgentLearner


class ISAC_Learner(OffPolicyMultiAgentLearner):
    def __init__(self,
                 config: Namespace,
                 agent_grouping: AgentGrouping,
                 model: nn.Module,
                 callback):
        super(ISAC_Learner, self).__init__(config, agent_grouping, model, callback)
        self.tau = config.tau
        self.alpha = {key: config.alpha for key in self.group_keys}
        self.use_automatic_entropy_tuning = config.use_automatic_entropy_tuning
        if self.use_automatic_entropy_tuning:
            self.target_entropy = {key: -model.actors[key].action_space.shape[-1] for key in self.group_keys}
            self.log_alpha = {key: nn.Parameter(torch.zeros(1, requires_grad=True, device=self.device))
                              for key in self.group_keys}
            self.alpha = {key: self.log_alpha[key].exp() for key in self.group_keys}
            self.alpha_optimizer = {key: torch.optim.Adam([self.log_alpha[key]], lr=config.learning_rate_actor)
                                    for key in self.group_keys}

    def build_optimizer(self):
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

    def update(self, sample):
        self.iterations += 1

        # Prepare training data.
        batch = self.build_training_data(
            sample,
            use_parameter_sharing=self.use_parameter_sharing,
            use_actions_mask=False
        )

        info = self.callback.on_update_start(self.iterations, model=self.model, batch=batch)

        # feedforward
        if self.use_rnn:
            observations_t = {k: v[:, :-1] for k, v in batch.observations.items()}
            observations_next = {k: v[:, 1:] for k, v in batch.observations.items()}
            agent_indices = {k: v[:, :-1] for k, v in batch.agent_indices.items()}
        else:
            observations_t = batch.observations
            observations_next = batch.next_observations
            agent_indices = batch.agent_indices

        # initial hidden states for rnn
        rnn_states_actor = self.model.init_actor_rnn_states(batch.batch_size)
        rnn_states_critic_1, rnn_states_critic_2 = self.model.init_critic_rnn_states(batch.batch_size)

        model_output = self.model(observations=batch.observations,
                                  agent_indices=batch.agent_indices,
                                  rnn_states=rnn_states_actor)
        actions_eval = model_output.actions
        log_pi_eval = model_output.log_probs
        if not self.agent_grouping.full_independent:
            actions_eval = self.packed_tensor(actions_eval)
            log_pi_eval = self.packed_tensor(log_pi_eval)

        if self.use_rnn:
            actions_next = {k: v[:, 1:] for k, v in actions_eval.items()}
            actions_eval = {k: v[:, :-1] for k, v in actions_eval.items()}

            log_pi_next = {k: v[:, 1:] for k, v in log_pi_eval.items()}
            log_pi_eval = {k: v[:, :-1] for k, v in log_pi_eval.items()}
        else:
            next_model_output = self.model(observations=batch.next_observations,
                                           agent_indices=agent_indices)
            actions_next = next_model_output.actions
            log_pi_next = next_model_output.log_probs
            if not self.agent_grouping.full_independent:
                actions_next = self.packed_tensor(actions_next)
                log_pi_next = self.packed_tensor(log_pi_next)

        action_q_1, action_q_2 = self.model.Qpolicy(observations=observations_t,
                                                    actions=batch.actions,
                                                    agent_indices=agent_indices,
                                                    rnn_states_1=rnn_states_critic_1,
                                                    rnn_states_2=rnn_states_critic_2)
        with torch.no_grad():
            next_q = self.model.Qtarget(observations=observations_next,
                                        actions=actions_next,
                                        agent_indices=agent_indices,
                                        rnn_states_1=rnn_states_critic_1,
                                        rnn_states_2=rnn_states_critic_2)


        for group, n_agents in self.n_group_agents.items():
            mask_values = batch.valid_mask(group, n_agents).reshape(-1)

            # update critic
            log_pi_next_eval = log_pi_next[group].reshape(-1)
            next_q_i = next_q[group].reshape(-1)
            target_value = next_q_i - self.alpha[group] * log_pi_next_eval
            rewards = batch.rewards[group].reshape(-1)
            terminals = batch.terminals[group].reshape(-1)
            backup = rewards + (1 - terminals) * self.gamma * target_value

            action_q_1_i = action_q_1[group].reshape(-1)
            td_error_1 = action_q_1_i - backup.detach()
            td_error_1 *= mask_values

            action_q_2_i = action_q_2[group].reshape(-1)
            td_error_2 = action_q_2_i - backup.detach()
            td_error_2 *= mask_values

            loss_c = ((td_error_1 ** 2).sum() + (td_error_2 ** 2).sum()) / mask_values.sum()

            self.optimizer[group]['critic'].zero_grad()
            loss_c.backward()
            if self.use_grad_clip:
                torch.nn.utils.clip_grad_norm_(self.model.critics[group].parameters(), self.grad_clip_norm)
            self.optimizer[group]['critic'].step()
            if self.scheduler[group]['critic'] is not None:
                self.scheduler[group]['critic'].step()

            # update actor
            policy_q_1, policy_q_2 = self.model.Qpolicy(observations=observations_t,
                                                        actions=actions_eval,
                                                        agent_indices=agent_indices,
                                                        group_key=group,
                                                        rnn_states_1=rnn_states_critic_1,
                                                        rnn_states_2=rnn_states_critic_2)
            log_pi_eval_i = log_pi_eval[group].reshape(-1)
            policy_q = torch.min(policy_q_1[group], policy_q_2[group]).reshape(-1)

            loss_a = ((self.alpha[group] * log_pi_eval_i - policy_q) * mask_values).sum() / mask_values.sum()

            self.optimizer[group]['actor'].zero_grad()
            loss_a.backward()
            if self.use_grad_clip:
                torch.nn.utils.clip_grad_norm_(self.model.actors[group].parameters(), self.grad_clip_norm)
            self.optimizer[group]['actor'].step()
            if self.scheduler[group]['actor'] is not None:
                self.scheduler[group]['actor'].step()

            # automatic entropy tuning
            if self.use_automatic_entropy_tuning:
                alpha_loss = -(self.log_alpha[group] * (log_pi_eval_i + self.target_entropy[group]).detach()).mean()
                self.alpha_optimizer[group].zero_grad()
                alpha_loss.backward()
                self.alpha_optimizer[group].step()
                self.alpha[group] = self.log_alpha[group].exp()
            else:
                alpha_loss = 0

            learning_rate_actor = self.optimizer[group]['actor'].state_dict()['param_groups'][0]['lr']
            learning_rate_critic = self.optimizer[group]['critic'].state_dict()['param_groups'][0]['lr']

            info.update({
                f"{group}/learning_rate_actor": learning_rate_actor,
                f"{group}/learning_rate_critic": learning_rate_critic,
                f"{group}/loss_actor": loss_a.item(),
                f"{group}/loss_critic": loss_c.item(),
                f"{group}/predictQ": policy_q.mean().item(),
            })
            if self.use_automatic_entropy_tuning:
                info.update({f"{group}/alpha_loss": alpha_loss.item(),
                             f"{group}/alpha": self.alpha[group].item()})

            info.update(self.callback.on_update_agent_wise(self.iterations, group, info=info,
                                                           mask_values=mask_values,
                                                           action_q_1_i=action_q_1_i, action_q_2_i=action_q_2_i,
                                                           log_pi_next_eval=log_pi_next_eval, next_q_i=next_q_i,
                                                           target_value=target_value, backup=backup,
                                                           td_error_1=td_error_1, td_error_2=td_error_2,
                                                           policy_q_1=policy_q_1, policy_q_2=policy_q_2,
                                                           log_pi_eval_i=log_pi_eval_i, policy_q=policy_q))

        self.model.soft_update(self.tau)
        info.update(self.callback.on_update_end(self.iterations, model=self.model, info=info))
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
            agent_mask = self.packed_tensor(agent_mask)
            terminals = self.packed_tensor(terminals)

        info = self.callback.on_update_start(self.iterations, method="update_rnn",
                                             model=self.model, sample_Tensor=sample_Tensor)

        # initial hidden states for rnn
        rnn_states_actor = self.model.init_actor_rnn_states(batch_size)
        rnn_states_critic_1, rnn_states_critic_2 = self.model.init_critic_rnn_states(batch_size)

        # feedforward
        model_output = self.model(observations=obs, agent_indices=agent_indices, rnn_states=rnn_states_actor)
        actions_eval, log_pi_eval = model_output.actions, model_output.log_probs

        obs_t = {k: v[:, :-1] for k, v in obs.items()}
        agent_indices_t = {k: v[:, :-1] for k, v in agent_indices.items()}
        action_q_1, action_q_2 = self.model.Qpolicy(observations=obs_t, actions=actions,
                                                    agent_indices=agent_indices_t,
                                                    rnn_states_1=rnn_states_critic_1,
                                                    rnn_states_2=rnn_states_critic_2)
        if not self.agent_grouping.full_independent:
            actions_eval = self.packed_tensor(actions_eval)
        next_q = self.model.Qtarget(observations=obs, actions=actions_eval,
                                    agent_indices=agent_indices,
                                    rnn_states_1=rnn_states_critic_1,
                                    rnn_states_2=rnn_states_critic_2)
        if not self.agent_grouping.full_independent:
            next_q = self.packed_tensor(next_q)
            action_q_1 = self.packed_tensor(action_q_1)
            action_q_2 = self.packed_tensor(action_q_2)
            log_pi_eval = self.packed_tensor(log_pi_eval)

        for group, agent_keys in self.groups.items():
            n_agents = len(agent_keys)
            bs = batch_size * n_agents
            group_filled = filled.unsqueeze(1).expand(batch_size, n_agents, seq_len).reshape([bs, seq_len])
            mask_values = agent_mask[group] * group_filled
            # update critic
            action_q_1_i = action_q_1[group].reshape(bs, seq_len)
            action_q_2_i = action_q_2[group].reshape(bs, seq_len)
            log_pi_next_eval = log_pi_eval[group][:, 1:].reshape(bs, seq_len)
            next_q_i = next_q[group][:, 1:].reshape(bs, seq_len)
            target_value = next_q_i - self.alpha[group] * log_pi_next_eval
            backup = rewards[group] + (1 - terminals[group]) * self.gamma * target_value
            td_error_1, td_error_2 = action_q_1_i - backup.detach(), action_q_2_i - backup.detach()
            td_error_1 *= mask_values
            td_error_2 *= mask_values
            loss_c = ((td_error_1 ** 2).sum() + (td_error_2 ** 2).sum()) / mask_values.sum()
            self.optimizer[group]['critic'].zero_grad()
            loss_c.backward()
            if self.use_grad_clip:
                torch.nn.utils.clip_grad_norm_(self.model.critics[group].parameters(), self.grad_clip_norm)
            self.optimizer[group]['critic'].step()
            if self.scheduler[group]['critic'] is not None:
                self.scheduler[group]['critic'].step()

            # update actor
            policy_q_1, policy_q_2 = self.model.Qpolicy(observations=obs, actions=actions_eval,
                                                        agent_indices=agent_indices, group_key=group,
                                                        rnn_states_1=rnn_states_critic_1,
                                                        rnn_states_2=rnn_states_critic_2)
            if not self.agent_grouping.full_independent:
                policy_q_1 = self.packed_tensor(policy_q_1)
                policy_q_2 = self.packed_tensor(policy_q_2)
            log_pi_eval_i = log_pi_eval[group][:, :-1].reshape(bs, seq_len)
            policy_q = torch.min(policy_q_1[group][:, :-1], policy_q_2[group][:, :-1]).reshape(bs, seq_len)
            loss_a = ((self.alpha[group] * log_pi_eval_i - policy_q) * mask_values).sum() / mask_values.sum()
            self.optimizer[group]['actor'].zero_grad()
            loss_a.backward()
            if self.use_grad_clip:
                torch.nn.utils.clip_grad_norm_(self.model.actors[group].parameters(), self.grad_clip_norm)
            self.optimizer[group]['actor'].step()
            if self.scheduler[group]['actor'] is not None:
                self.scheduler[group]['actor'].step()

            # automatic entropy tuning
            if self.use_automatic_entropy_tuning:
                alpha_loss = -(self.log_alpha[group] * (log_pi_eval_i + self.target_entropy[group]).detach()).mean()
                self.alpha_optimizer[group].zero_grad()
                alpha_loss.backward()
                self.alpha_optimizer[group].step()
                self.alpha[group] = self.log_alpha[group].exp()
            else:
                alpha_loss = 0

            learning_rate_actor = self.optimizer[group]['actor'].state_dict()['param_groups'][0]['lr']
            learning_rate_critic = self.optimizer[group]['critic'].state_dict()['param_groups'][0]['lr']

            info.update({
                f"{group}/learning_rate_actor": learning_rate_actor,
                f"{group}/learning_rate_critic": learning_rate_critic,
                f"{group}/loss_actor": loss_a.item(),
                f"{group}/loss_critic": loss_c.item(),
                f"{group}/predictQ": policy_q.mean().item(),
            })
            if self.use_automatic_entropy_tuning:
                info.update({f"{group}/alpha_loss": alpha_loss.item(),
                             f"{group}/alpha": self.alpha[group].item()})

            info.update(self.callback.on_update_agent_wise(self.iterations, group, info=info, method="update_rnn",
                                                           mask_values=mask_values,
                                                           action_q_1_i=action_q_1_i, action_q_2_i=action_q_2_i,
                                                           log_pi_next_eval=log_pi_next_eval, next_q_i=next_q_i,
                                                           target_value=target_value, backup=backup,
                                                           td_error_1=td_error_1, td_error_2=td_error_2,
                                                           policy_q_1=policy_q_1, policy_q_2=policy_q_2,
                                                           log_pi_eval_i=log_pi_eval_i, policy_q=policy_q))

        self.model.soft_update(self.tau)
        info.update(self.callback.on_update_end(self.iterations, method="update_rnn", model=self.model, info=info))
        return info
