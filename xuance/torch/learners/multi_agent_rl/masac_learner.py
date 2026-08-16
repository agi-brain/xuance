"""
Multi-agent Soft Actor-critic (MASAC)
Implementation: Pytorch
"""
import torch
from torch import nn
from xuance.common import AgentGrouping
from argparse import Namespace
from xuance.torch.learners.multi_agent_rl.isac_learner import ISAC_Learner


class MASAC_Learner(ISAC_Learner):
    def __init__(self,
                 config: Namespace,
                 agent_grouping: AgentGrouping,
                 model: nn.Module,
                 callback):
        super(MASAC_Learner, self).__init__(config, agent_grouping, model, callback)

    def update(self, sample):
        self.iterations += 1

        # Prepare training data.
        batch = self.build_training_data(sample,
                                         use_parameter_sharing=self.use_parameter_sharing,
                                         use_actions_mask=False)

        obs_joint = self.get_joint_input(batch.observations, (batch.batch_size, -1))
        next_obs_joint = self.get_joint_input(batch.next_observations, (batch.batch_size, -1))
        actions_joint = self.get_joint_input(batch.actions, (batch.batch_size, -1))

        if not self.agent_grouping.full_independent:
            batch.observations = self.packed_tensor(batch.observations)
            batch.next_observations = self.packed_tensor(batch.next_observations)
            batch.agent_masks = self.packed_tensor(batch.agent_masks)
            batch.rewards = self.packed_tensor(batch.rewards)
            batch.terminals = self.packed_tensor(batch.terminals)

        info = self.callback.on_update_start(self.iterations, model=self.model,
                                             batch=batch, obs_joint=obs_joint,
                                             next_obs_joint=next_obs_joint,
                                             actions_joint=actions_joint)

        # feedforward
        model_output = self.model(observations=batch.observations,
                                  agent_indices=batch.agent_indices)
        actions_eval, log_pi_eval = model_output.actions, model_output.log_probs

        next_model_output = self.model(observations=batch.next_observations,
                                       agent_indices=batch.agent_indices)
        actions_next, log_pi_next = next_model_output.actions, next_model_output.log_probs

        actions_next_joint = self.get_joint_input(actions_next, (batch.batch_size, -1))
        actions_eval_joint = self.get_joint_input(actions_eval, (batch.batch_size, -1))

        action_q_1, action_q_2 = self.model.Qpolicy(joint_observations=obs_joint,
                                                    joint_actions=actions_joint,
                                                    agent_indices=batch.agent_indices)
        next_q = self.model.Qtarget(joint_observations=next_obs_joint,
                                    joint_actions=actions_next_joint,
                                    agent_indices=batch.agent_indices)

        if not self.agent_grouping.full_independent:
            log_pi_next = self.packed_tensor(log_pi_next)
            log_pi_eval = self.packed_tensor(log_pi_eval)

        for group, agent_keys in self.groups.items():
            bs = batch.batch_size * self.n_group_agents[group]
            mask_values = batch.agent_masks[group]
            # critic update
            action_q_1_i = action_q_1[group].reshape(-1)
            action_q_2_i = action_q_2[group].reshape(-1)
            log_pi_next_eval = log_pi_next[group].reshape(-1)
            rewards = batch.rewards[group].reshape(-1)
            terminals = batch.terminals[group].reshape(-1)

            target_value = next_q[group].reshape(bs) - self.alpha[group] * log_pi_next_eval
            backup = rewards + (1 - terminals) * self.gamma * target_value
            td_error_1 = action_q_1_i - backup.detach()
            td_error_1 *= mask_values
            td_error_2 = action_q_2_i - backup.detach()
            td_error_2 *= mask_values

            loss_c = ((td_error_1 ** 2).sum() + (td_error_2 ** 2).sum()) / mask_values.sum()

            self.optimizer[group]['critic'].zero_grad()
            loss_c.backward()
            if self.use_grad_clip:
                torch.nn.utils.clip_grad_norm_(self.model.parameters_critic[group], self.grad_clip_norm)
            self.optimizer[group]['critic'].step()
            if self.scheduler[group]['critic'] is not None:
                self.scheduler[group]['critic'].step()

            # actor update
            policy_q_1, policy_q_2 = self.model.Qpolicy(joint_observations=obs_joint,
                                                        joint_actions=actions_eval_joint,
                                                        agent_indices=batch.agent_indices,
                                                        group_key=group)
            log_pi_eval_i = log_pi_eval[group].reshape(-1)
            policy_q = torch.min(policy_q_1[group], policy_q_2[group]).reshape(-1)

            loss_a = ((self.alpha[group] * log_pi_eval_i - policy_q) * mask_values).sum() / mask_values.sum()

            self.optimizer[group]['actor'].zero_grad()
            loss_a.backward()
            if self.use_grad_clip:
                torch.nn.utils.clip_grad_norm_(self.model.parameters_actor[group], self.grad_clip_norm)
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
                                                           log_pi_next_eval=log_pi_next_eval,
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

        obs_joint = self.get_joint_input(obs, (batch_size, seq_len + 1, -1))
        actions_joint = self.get_joint_input(actions, (batch_size, seq_len, -1))

        if not self.agent_grouping.full_independent:
            obs = self.packed_tensor(obs)
            rewards = self.packed_tensor(rewards)
            agent_mask = self.packed_tensor(agent_mask)
            terminals = self.packed_tensor(terminals)

        info = self.callback.on_update_start(self.iterations, method="update_rnn", model=self.model,
                                             sample_Tensor=sample_Tensor,
                                             obs_joint=obs_joint, actions_joint=actions_joint)

        # initial hidden states for rnn
        rnn_states_actor = self.model.init_actor_rnn_states(batch_size)
        rnn_states_critic_1, rnn_states_critic_2 = self.model.init_critic_rnn_states(batch_size)

        # feedforward
        model_output = self.model(observations=obs, agent_indices=agent_indices, rnn_states=rnn_states_actor)
        actions_eval, log_pi_eval = model_output.actions, model_output.log_probs

        actions_eval_joint = self.get_joint_input(actions_eval, (batch_size, seq_len + 1, -1))

        agent_indices_t = {k: v[:, :-1] for k, v in agent_indices.items()}
        action_q_1, action_q_2 = self.model.Qpolicy(joint_observations=obs_joint[:, :-1],
                                                    joint_actions=actions_joint,
                                                    agent_indices=agent_indices_t,
                                                    rnn_states_1=rnn_states_critic_1,
                                                    rnn_states_2=rnn_states_critic_2)

        next_q = self.model.Qtarget(joint_observations=obs_joint,
                                    joint_actions=actions_eval_joint,
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
            # critic update
            action_q_1_i = action_q_1[group].reshape(bs, seq_len)
            action_q_2_i = action_q_2[group].reshape(bs, seq_len)
            log_pi_next_eval = log_pi_eval[group][:, 1:].reshape(bs, seq_len)
            target_value = next_q[group][:, 1:].reshape(bs, seq_len) - self.alpha[group] * log_pi_next_eval
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

            # actor update
            if self.use_parameter_sharing:
                actions_eval_joint = actions_eval_joint[:, :-1]
            else:
                actions_eval_detach_others = {k: actions_eval[k] if k == group else actions_eval[k].detach()
                                              for k in self.group_keys}
                actions_eval_joint = self.get_joint_input(actions_eval_detach_others,
                                                          (batch_size, seq_len + 1, -1))[:, :-1]
            policy_q_1, policy_q_2 = self.model.Qpolicy(joint_observations=obs_joint[:, :-1],
                                                        joint_actions=actions_eval_joint,
                                                        agent_indices=agent_indices_t, grou_key=group,
                                                        rnn_states_1=rnn_states_critic_1,
                                                        rnn_states_2=rnn_states_critic_2)
            if not self.agent_grouping.full_independent:
                policy_q_1 = self.packed_tensor(policy_q_1)
                policy_q_2 = self.packed_tensor(policy_q_2)
            log_pi_eval_i = log_pi_eval[group][:, :-1].reshape(bs, seq_len)
            policy_q = torch.min(policy_q_1[group], policy_q_2[group]).reshape(bs, seq_len)
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
                                                           log_pi_next_eval=log_pi_next_eval,
                                                           target_value=target_value, backup=backup,
                                                           td_error_1=td_error_1, td_error_2=td_error_2,
                                                           policy_q_1=policy_q_1, policy_q_2=policy_q_2,
                                                           log_pi_eval_i=log_pi_eval_i, policy_q=policy_q))

        self.model.soft_update(self.tau)
        info.update(self.callback.on_update_end(self.iterations, method="update_rnn", model=self.model, info=info))
        return info
