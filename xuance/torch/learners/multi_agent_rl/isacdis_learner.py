"""
Independent Soft Actor-critic (ISAC) with discrete action spaces.
Implementation: Pytorch
"""
import math
from argparse import Namespace
from xuance.common import AgentGrouping

import torch
from xuance.torch import Module
from xuance.torch.utils import AgentGroupedTensor
from xuance.torch.learners.multi_agent_rl.isac_learner import ISAC_Learner


class ISACDIS_Learner(ISAC_Learner):
    def __init__(self,
                 config: Namespace,
                 agent_grouping: AgentGrouping,
                 model: Module,
                 callback):
        super(ISACDIS_Learner, self).__init__(config, agent_grouping, model, callback)

    def get_target_entropy(self):
        ratio = getattr(self.config, "target_entropy_ratio", 0.8)
        return {
            key: ratio * math.log(self.model.actors[key].action_dim)
            for key in self.group_keys
        }

    def update(self, sample):
        self.iterations += 1

        # Prepare training data.
        batch = self.build_training_data(
            sample,
            use_actions_mask=False
        )

        info = self.callback.on_update_start(self.iterations, model=self.model, batch=batch)

        if self.use_rnn:
            observations_t = AgentGroupedTensor(
                {k: v[:, :, :-1] for k, v in batch.observations.grouped_tensor.items()}, self.agent_grouping
            )
            agent_indices_t = AgentGroupedTensor(
                {k: v[:, :, :-1] for k, v in batch.agent_indices.grouped_tensor.items()}, self.agent_grouping
            )
            avail_actions_t = AgentGroupedTensor(
                {k: v[:, :, :-1] for k, v in batch.avail_actions.grouped_tensor.items()}, self.agent_grouping
            )
        else:
            observations_t = batch.observations
            agent_indices_t = batch.agent_indices
            avail_actions_t = batch.avail_actions

        # initial hidden states for rnn
        rnn_states_actor = self.model.init_actor_rnn_states(batch.batch_size)
        rnn_states_critic_1, rnn_states_critic_2 = self.model.init_critic_rnn_states(batch.batch_size)

        # feedforward
        action_q_1, action_q_2 = self.model.Qvalues(observations=observations_t,
                                                    agent_indices=agent_indices_t,
                                                    rnn_states_critic_1=rnn_states_critic_1,
                                                    rnn_states_critic_2=rnn_states_critic_2)

        with torch.no_grad():
            if self.use_rnn:
                action_prob_next, log_pi_next, q_next = self.model.Qtarget(next_observations=batch.observations,
                                                                           agent_indices=batch.agent_indices,
                                                                           rnn_states_critic_1=rnn_states_critic_1,
                                                                           rnn_states_critic_2=rnn_states_critic_2)
                action_prob_next.grouped_tensor = {k: v[:, :, 1:] for k, v in action_prob_next.grouped_tensor.items()}
                log_pi_next.grouped_tensor = {k: v[:, :, 1:] for k, v in log_pi_next.grouped_tensor.items()}
                q_next.grouped_tensor = {k: v[:, :, 1:] for k, v in q_next.grouped_tensor.items()}
            else:
                action_prob_next, log_pi_next, q_next = self.model.Qtarget(next_observations=batch.next_observations,
                                                                           agent_indices=batch.agent_indices)

        for group, n_agents in self.n_group_agents.items():
            mask_values = batch.valid_mask(group, n_agents).reshape(-1)

            # update critic
            action_q_1_a = action_q_1.packed(group).gather(
                dim=-1, index=batch.actions.packed(group).long().unsqueeze(-1))
            action_q_1_a = action_q_1_a.reshape(-1)
            action_q_2_a = action_q_2.packed(group).gather(
                dim=-1, index=batch.actions.packed(group).long().unsqueeze(-1))
            action_q_2_a = action_q_2_a.reshape(-1)
            target_value = (action_prob_next.group(group) * self.alpha[group] * log_pi_next.group(group)).sum(dim=-1)
            target_value = q_next.group(group) - target_value
            target_value = target_value.reshape(-1)
            backup = batch.rewards.packed(group) + (1 - batch.terminals.packed(group)) * self.gamma * target_value
            td_error_1 = action_q_1_a - backup.detach()
            td_error_1 *= mask_values
            td_error_2 = action_q_2_a - backup.detach()
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
            action_probs, log_action_probs, policy_q = self.model.Qpolicy(
                observations=observations_t,
                agent_indices=agent_indices_t,
                avail_actions=avail_actions_t,
                group_key=group,
                rnn_states_actor=rnn_states_actor,
                rnn_states_critic_1=rnn_states_critic_1,
                rnn_states_critic_2=rnn_states_critic_2
            )
            loss_a = self.alpha[group] * (action_probs.group(group) * log_action_probs.group(group)).sum(dim=-1)
            loss_a = loss_a - policy_q.group(group)
            loss_a = (loss_a * mask_values).sum() / mask_values.sum()
            self.optimizer[group]['actor'].zero_grad()
            loss_a.backward()
            if self.use_grad_clip:
                torch.nn.utils.clip_grad_norm_(self.model.actors[group].parameters(), self.grad_clip_norm)
            self.optimizer[group]['actor'].step()
            if self.scheduler[group]['actor'] is not None:
                self.scheduler[group]['actor'].step()

            # automatic entropy tuning
            if self.use_automatic_entropy_tuning:
                entropy = -(action_probs.group(group) * log_action_probs.group(group)).sum(dim=-1).reshape(-1)
                entropy_error = (entropy - self.target_entropy[group]).detach()
                alpha_loss = (self.log_alpha[group] * entropy_error * mask_values).sum() / mask_values.sum()
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
                f"{group}/predictQ": policy_q.group(group).mean().item(),
            })
            if self.use_automatic_entropy_tuning:
                info.update({f"{group}/alpha_loss": alpha_loss.item(),
                             f"{group}/alpha": self.alpha[group].item()})

            info.update(self.callback.on_update_agent_wise(self.iterations, group, info=info,
                                                           mask_values=mask_values,
                                                           action_q_1_a=action_q_1_a, action_q_2_a=action_q_2_a,
                                                           target_value=target_value, backup=backup,
                                                           td_error_1=td_error_1, td_error_2=td_error_2,
                                                           action_probs=action_probs, log_action_probs=log_action_probs,
                                                           action_q_1=action_q_1, action_q_2=action_q_2,
                                                           policy_q=policy_q))

        self.model.soft_update(self.tau)
        info.update(self.callback.on_update_end(self.iterations, model=self.model, info=info))
        return info
