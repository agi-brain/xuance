"""
Multi-Agent TD3
"""
import torch
from torch import nn
from argparse import Namespace
from xuance.common import AgentGrouping
from xuance.torch.utils import AgentGroupedTensor
from xuance.torch.learners.multi_agent_rl.itd3_learner import ITD3_Learner


class MATD3_Learner(ITD3_Learner):
    def __init__(self,
                 config: Namespace,
                 agent_grouping: AgentGrouping,
                 model: nn.Module,
                 callback):
        ITD3_Learner.__init__(self, config, agent_grouping, model, callback)

    def update(self, sample):
        self.iterations += 1

        # prepare training data
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

        # get values
        q_eval_A, q_eval_B, _ = self.model.Qpolicy(joint_observations=obs_joint_t,
                                                   joint_actions=actions_joint_t,
                                                   agent_indices=agent_indices_t,
                                                   rnn_states_1=rnn_states_critic_1,
                                                   rnn_states_2=rnn_states_critic_2)

        with torch.no_grad():
            if self.use_rnn:
                actions_next = self.model.Atarget(observations=batch.observations,
                                                  agent_indices=batch.agent_indices,
                                                  rnn_states=rnn_states_actor)
                actions_joint_next = self.get_joint_input(actions_next.agent_wise,
                                                          (batch.batch_size, batch.seq_length + 1, -1))
                q_next = self.model.Qtarget(joint_observations=obs_joint,
                                            joint_actions=actions_joint_next,
                                            agent_indices=batch.agent_indices,
                                            rnn_states_1=rnn_states_critic_1,
                                            rnn_states_2=rnn_states_critic_2)
                q_next.grouped_tensor = {k: v[:, :, 1:] for k, v in q_next.grouped_tensor.items()}
            else:
                actions_next = self.model.Atarget(observations=batch.next_observations,
                                                  agent_indices=batch.agent_indices)
                actions_joint_next = self.get_joint_input(actions_next.agent_wise, (batch.batch_size, -1))
                obs_joint_next = self.get_joint_input(batch.next_observations.agent_wise,
                                                      output_shape=(batch.batch_size, -1))
                q_next = self.model.Qtarget(joint_observations=obs_joint_next,
                                            joint_actions=actions_joint_next,
                                            agent_indices=batch.agent_indices)

        # update critic(s)
        for group, n_agents in self.n_group_agents.items():
            mask_values = batch.valid_mask(group, n_agents).reshape(-1)

            q_eval_A_i = q_eval_A.packed(group).reshape(-1)
            q_eval_B_i = q_eval_B.packed(group).reshape(-1)
            q_next_i = q_next.packed(group).reshape(-1)
            rewards = batch.rewards.packed(group).reshape(-1)
            terminals = batch.terminals.packed(group).reshape(-1)

            q_target = rewards + (1 - terminals) * self.gamma * q_next_i
            td_error_A = (q_eval_A_i - q_target.detach()) * mask_values
            td_error_B = (q_eval_B_i - q_target.detach()) * mask_values

            loss_critic = ((td_error_A ** 2).sum() + (td_error_B ** 2).sum()) / mask_values.sum()

            self.optimizer[group]['critic'].zero_grad()
            loss_critic.backward()
            if self.use_grad_clip:
                torch.nn.utils.clip_grad_norm_(self.model.critics[group].parameters(), self.grad_clip_norm)
            self.optimizer[group]['critic'].step()
            if self.scheduler[group]['critic'] is not None:
                self.scheduler[group]['critic'].step()

            learning_rate_critic = self.optimizer[group]['critic'].state_dict()['param_groups'][0]['lr']

            info.update({
                f"{group}/learning_rate_critic": learning_rate_critic,
                f"{group}/loss_critic": loss_critic.item(),
                f"{group}/predictQ_A": q_eval_A_i.mean().item(),
                f"{group}/predictQ_B": q_eval_B_i.mean().item()
            })

            info.update(self.callback.on_update_agent_wise(self.iterations, group, info=info,
                                                           mask_values=mask_values,
                                                           q_eval_A_i=q_eval_A_i, q_eval_B_i=q_eval_B_i,
                                                           q_target=q_target, q_next_i=q_next_i,
                                                           td_error_A=td_error_A, td_error_B=td_error_B))

        # update actor(s)
        if self.iterations % self.actor_update_delay == 0:
            actions_eval = self.model(observations=observations_t,
                                      agent_indices=agent_indices_t,
                                      rnn_states=rnn_states_actor).actions
            actions_eval_agent_wise = actions_eval.agent_wise
            for group, n_agents in self.n_group_agents.items():
                mask_values = batch.valid_mask(group, n_agents).reshape(-1)
                agent_keys = self.groups[group]

                # calculate the objective of actor
                actions_all = batch.actions.agent_wise.copy()
                for key in agent_keys:
                    actions_all[key] = actions_eval_agent_wise[key]
                if self.use_rnn:
                    actions_joint_eval = self.get_joint_input(actions_all, (batch.batch_size, batch.seq_length, -1))
                else:
                    actions_joint_eval = self.get_joint_input(actions_all, (batch.batch_size, -1))
                _, _, q_policy = self.model.Qpolicy(joint_observations=obs_joint_t,
                                                    joint_actions=actions_joint_eval,
                                                    agent_indices=agent_indices_t,
                                                    group_key=group,
                                                    rnn_states_1=rnn_states_critic_1,
                                                    rnn_states_2=rnn_states_critic_2)
                q_policy_i = q_policy.packed(group).reshape(-1)
                loss_actor = -(q_policy_i * mask_values).sum() / mask_values.sum()
                self.optimizer[group]['actor'].zero_grad()
                loss_actor.backward()
                if self.use_grad_clip:
                    torch.nn.utils.clip_grad_norm_(self.model.actors[group].parameters(), self.grad_clip_norm)
                self.optimizer[group]['actor'].step()
                if self.scheduler[group]['actor'] is not None:
                    self.scheduler[group]['actor'].step()

                learning_rate_actor = self.optimizer[group]['actor'].state_dict()['param_groups'][0]['lr']

                info.update({
                    f"{group}/learning_rate_actor": learning_rate_actor,
                    f"{group}/loss_actor": loss_actor.item(),
                    f"{group}/q_policy": q_policy_i.mean().item(),
                })
                info.update(self.callback.on_update_agent_wise(self.iterations, group, info=info,
                                                               mask_values=mask_values, q_policy_i=q_policy_i))
            self.model.soft_update(self.tau)

        info.update(self.callback.on_update_end(self.iterations, model=self.model, info=info))
        return info
