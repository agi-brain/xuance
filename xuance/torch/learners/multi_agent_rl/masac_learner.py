"""
Multi-agent Soft Actor-critic (MASAC)
Implementation: Pytorch
"""
import torch
from torch import nn
from argparse import Namespace
from xuance.common import AgentGrouping
from xuance.torch.utils import AgentGroupedTensor
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
        batch = self.build_training_data(
            sample,
            use_actions_mask=False
        )

        info = self.callback.on_update_start(self.iterations, model=self.model, batch=batch)

        if self.use_rnn:
            obs_joint = self.get_joint_input(batch.observations.agent_wise,
                                             output_shape=(batch.batch_size, batch.seq_length + 1, -1))
            obs_joint_t = obs_joint[:, :-1]
            actions_joint_t = self.get_joint_input(batch.actions.agent_wise,
                                                   output_shape=(batch.batch_size, batch.seq_length, -1))
            observations_t = AgentGroupedTensor(
                {k: v[:, :, :-1] for k, v in batch.observations.grouped_tensor.items()}, self.agent_grouping
            )
            agent_indices_t = AgentGroupedTensor(
                {k: v[:, :, :-1] for k, v in batch.agent_indices.grouped_tensor.items()}, self.agent_grouping
            )
        else:
            obs_joint_t = obs_joint = self.get_joint_input(batch.observations.agent_wise,
                                                           output_shape=(batch.batch_size, -1))
            actions_joint_t = self.get_joint_input(batch.actions.agent_wise, output_shape=(batch.batch_size, -1))
            observations_t = batch.observations
            agent_indices_t = batch.agent_indices

        # initial hidden states for rnn
        rnn_states_actor = self.model.init_actor_rnn_states(batch.batch_size)
        rnn_states_critic_1, rnn_states_critic_2 = self.model.init_critic_rnn_states(batch.batch_size)

        # feedforward
        model_output = self.model(observations=batch.observations,
                                  agent_indices=batch.agent_indices,
                                  rnn_states=rnn_states_actor)
        actions_eval = model_output.actions
        actions_eval_agent_wise = actions_eval.agent_wise
        log_pi_eval = model_output.log_probs

        action_q_1, action_q_2 = self.model.Qpolicy(joint_observations=obs_joint_t,
                                                    joint_actions=actions_joint_t,
                                                    agent_indices=agent_indices_t,
                                                    rnn_states_1=rnn_states_critic_1,
                                                    rnn_states_2=rnn_states_critic_2)

        with torch.no_grad():
            if self.use_rnn:
                actions_joint_next = self.get_joint_input(actions_eval.agent_wise,
                                                          output_shape=(batch.batch_size, batch.seq_length + 1, -1))
                q_next = self.model.Qtarget(joint_observations=obs_joint,
                                            joint_actions=actions_joint_next,
                                            agent_indices=batch.agent_indices,
                                            rnn_states_1=rnn_states_critic_1,
                                            rnn_states_2=rnn_states_critic_2)
                q_next.grouped_tensor = {k: v[:, :, 1:] for k, v in q_next.grouped_tensor.items()}
                log_pi_next = AgentGroupedTensor(
                    {k: v[:, :, 1:] for k, v in log_pi_eval.grouped_tensor.items()}, self.agent_grouping
                )
                log_pi_eval.grouped_tensor = {k: v[:, :, :-1] for k, v in log_pi_eval.grouped_tensor.items()}
                actions_eval_agent_wise = {k: v[:, :-1] for k, v in actions_eval_agent_wise.items()}
            else:
                next_model_output = self.model(observations=batch.next_observations,
                                               agent_indices=batch.agent_indices)
                actions_next = next_model_output.actions
                log_pi_next = next_model_output.log_probs
                actions_joint_next = self.get_joint_input(actions_next.agent_wise, (batch.batch_size, -1))
                obs_joint_next = self.get_joint_input(batch.next_observations.agent_wise,
                                                      output_shape=(batch.batch_size, -1))
                q_next = self.model.Qtarget(joint_observations=obs_joint_next,
                                            joint_actions=actions_joint_next,
                                            agent_indices=batch.agent_indices)

        # calculate loss and update networks
        for group, n_agents in self.n_group_agents.items():
            mask_values = batch.valid_mask(group, n_agents).reshape(-1)
            agent_keys = self.groups[group]

            # update critic
            action_q_1_i = action_q_1.packed(group).reshape(-1)
            action_q_2_i = action_q_2.packed(group).reshape(-1)
            log_pi_next_eval = log_pi_next.packed(group).reshape(-1)
            rewards = batch.rewards.packed(group).reshape(-1)
            terminals = batch.terminals.packed(group).reshape(-1)

            target_value = q_next.packed(group).reshape(-1) - self.alpha[group] * log_pi_next_eval
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

            # update actor
            # calculate the objective of actor
            actions_all = batch.actions.agent_wise.copy()
            for key in agent_keys:
                actions_all[key] = actions_eval_agent_wise[key]
            if self.use_rnn:
                actions_joint_eval = self.get_joint_input(actions_all, (batch.batch_size, batch.seq_length, -1))
            else:
                actions_joint_eval = self.get_joint_input(actions_all, (batch.batch_size, -1))

            policy_q_1, policy_q_2 = self.model.Qpolicy(joint_observations=obs_joint_t,
                                                        joint_actions=actions_joint_eval,
                                                        agent_indices=agent_indices_t,
                                                        group_key=group,
                                                        rnn_states_1=rnn_states_critic_1,
                                                        rnn_states_2=rnn_states_critic_2)
            log_pi_eval_i = log_pi_eval.packed(group).reshape(-1)
            policy_q = torch.min(policy_q_1.packed(group), policy_q_2.packed(group)).reshape(-1)

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

