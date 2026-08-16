"""
Multi-Agent TD3
"""
import torch
from torch import nn
from typing import Optional
from argparse import Namespace
from xuance.common import AgentGrouping
from xuance.torch.learners.multi_agent_rl.itd3_learner import ITD3_Learner
from xuance.torch.learners.multi_agent_rl.maddpg_learner import MADDPG_Learner
from xuance.torch.rl_models.modules import OffPolicyMARLBatch


class MATD3_Learner(ITD3_Learner, MADDPG_Learner):
    def __init__(self,
                 config: Namespace,
                 agent_grouping: AgentGrouping,
                 model: nn.Module,
                 callback):
        ITD3_Learner.__init__(self, config, agent_grouping, model, callback)

    def build_training_data(
            self,
            sample: Optional[dict],
            use_parameter_sharing: Optional[bool] = False,
            use_actions_mask: Optional[bool] = False,
            use_global_state: Optional[bool] = False,
            use_shared_rewards: Optional[bool] = False,
    ) -> OffPolicyMARLBatch:
        return MADDPG_Learner.build_training_data(
            self,
            sample=sample,
            use_parameter_sharing=use_parameter_sharing,
            use_actions_mask=use_actions_mask,
            use_global_state=use_global_state,
            use_shared_rewards=use_shared_rewards
        )

    def update(self, sample):
        self.iterations += 1

        # prepare training data
        batch = self.build_training_data(sample,
                                         use_parameter_sharing=self.use_parameter_sharing,
                                         use_actions_mask=False)
        if self.use_rnn:
            obs_joint = self.get_joint_input(batch.observations,
                                             output_shape=(batch.batch_size, batch.seq_length + 1, -1))
            actions_joint_t = self.get_joint_input(batch.actions,
                                                   output_shape=(batch.batch_size, batch.seq_length, -1))
            if not self.agent_grouping.full_independent:
                batch.observations = self.packed_tensor(batch.observations)

            observations_t = {k: v[:, :-1] for k, v in batch.observations.items()}
            observations_next = {k: v[:, 1:] for k, v in batch.observations.items()}

            obs_joint_t = obs_joint[:, :-1]
            obs_joint_next = obs_joint[:, 1:]

            agent_indices = {k: v[:, :-1] for k, v in batch.agent_indices.items()}
        else:
            obs_joint = self.get_joint_input(batch.observations,
                                             output_shape=(batch.batch_size, -1))
            obs_joint_t = obs_joint
            obs_joint_next = self.get_joint_input(batch.next_observations,
                                                  output_shape=(batch.batch_size, -1))
            actions_joint_t = self.get_joint_input(batch.actions,
                                                   output_shape=(batch.batch_size, -1))
            if not self.agent_grouping.full_independent:
                batch.observations = self.packed_tensor(batch.observations)
                batch.next_observations = self.packed_tensor(batch.next_observations)

            observations_t = batch.observations
            observations_next = batch.next_observations

            agent_indices = batch.agent_indices

        info = self.callback.on_update_start(self.iterations, model=self.model, batch=batch)

        # initial hidden states for rnn
        rnn_states_actor = self.model.init_actor_rnn_states(batch.batch_size)
        rnn_states_critic_1, rnn_states_critic_2 = self.model.init_critic_rnn_states(batch.batch_size)

        # get values
        q_eval_A, q_eval_B, _ = self.model.Qpolicy(joint_observations=obs_joint_t,
                                                   joint_actions=actions_joint_t,
                                                   agent_indices=agent_indices,
                                                   rnn_states_1=rnn_states_critic_1,
                                                   rnn_states_2=rnn_states_critic_2)
        with torch.no_grad():
            actions_next = self.model.Atarget(observations=observations_next,
                                              agent_indices=agent_indices,
                                              rnn_states=rnn_states_actor)
            if self.use_rnn:
                actions_joint_next = self.get_joint_input(actions_next, (batch.batch_size, batch.seq_length, -1))
            else:
                actions_joint_next = self.get_joint_input(actions_next, (batch.batch_size, -1))

            q_next = self.model.Qtarget(joint_observations=obs_joint_next,
                                        joint_actions=actions_joint_next,
                                        agent_indices=agent_indices,
                                        rnn_states_1=rnn_states_critic_1,
                                        rnn_states_2=rnn_states_critic_2)

        # update critic(s)
        for group, n_agents in self.n_group_agents.items():
            mask_values = batch.valid_mask(group, n_agents).reshape(-1)

            q_eval_A_i = q_eval_A[group].reshape(-1)
            q_eval_B_i = q_eval_B[group].reshape(-1)
            q_next_i = q_next[group].reshape(-1)
            rewards = batch.rewards[group].reshape(-1)
            terminals = batch.terminals[group].reshape(-1)

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
                f"{group}/predictQ_A": q_eval_A[group].mean().item(),
                f"{group}/predictQ_B": q_eval_B[group].mean().item()
            })

            info.update(self.callback.on_update_agent_wise(self.iterations, group, info=info,
                                                           mask_values=mask_values,
                                                           q_eval_A_i=q_eval_A_i, q_eval_B_i=q_eval_B_i,
                                                           q_target=q_target, q_next_i=q_next_i,
                                                           td_error_A=td_error_A, td_error_B=td_error_B))

        # update actor(s)
        if self.iterations % self.actor_update_delay == 0:
            actions_eval = self.model(observations=observations_t,
                                      agent_indices=agent_indices,
                                      rnn_states=rnn_states_actor).actions
            for group, n_agents in self.n_group_agents.items():
                mask_values = batch.valid_mask(group, n_agents).reshape(-1)
                agent_keys = self.groups[group]

                # calculate the objective of actor
                actions_all = batch.actions.copy()
                for key in agent_keys:
                    actions_all[key] = actions_eval[key]
                if self.use_rnn:
                    actions_joint_eval = self.get_joint_input(actions_all, (batch.batch_size, batch.seq_length, -1))
                else:
                    actions_joint_eval = self.get_joint_input(actions_all, (batch.batch_size, -1))
                _, _, q_policy = self.model.Qpolicy(joint_observations=obs_joint_t,
                                                    joint_actions=actions_joint_eval,
                                                    agent_indices=agent_indices,
                                                    group_key=group,
                                                    rnn_states_1=rnn_states_critic_1,
                                                    rnn_states_2=rnn_states_critic_2)
                q_policy_i = q_policy[group].reshape(-1)
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
            actions = self.packed_tensor(actions)
            rewards = self.packed_tensor(rewards)
            terminals = self.packed_tensor(terminals)
            agent_mask = self.packed_tensor(agent_mask)

        info = self.callback.on_update_start(self.iterations, method="update_rnn", model=self.model,
                                             sample_Tensor=sample_Tensor,
                                             obs_joint=obs_joint, actions_joint=actions_joint)

        # initial hidden states for rnn
        rnn_states_actor = self.model.init_actor_rnn_states(batch_size)
        rnn_states_critic_1, rnn_states_critic_2 = self.model.init_critic_rnn_states(batch_size)

        # get q values
        next_actions = self.model.Atarget(observations=obs, agent_indices=agent_indices,
                                          rnn_states=rnn_states_actor)
        next_actions_joint = self.get_joint_input(next_actions, (batch_size, seq_len + 1, -1))

        obs_t = {k: v[:, :-1] for k, v in obs.items()}
        agent_indices_t = {k: v[:, :-1] for k, v in agent_indices.items()}
        q_eval_A, q_eval_B, _ = self.model.Qpolicy(joint_observations=obs_joint[:, :-1],
                                                   joint_actions=actions_joint,
                                                   agent_indices=agent_indices_t,
                                                   rnn_states_1=rnn_states_critic_1,
                                                   rnn_states_2=rnn_states_critic_2)
        q_next = self.model.Qtarget(joint_observations=obs_joint,
                                    joint_actions=next_actions_joint,
                                    agent_indices=agent_indices,
                                    rnn_states_1=rnn_states_critic_1,
                                    rnn_states_2=rnn_states_critic_2)

        if not self.agent_grouping.full_independent:
            q_next = self.packed_tensor(q_next)
            q_eval_A = self.packed_tensor(q_eval_A)
            q_eval_B = self.packed_tensor(q_eval_B)

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

            info.update(
                self.callback.on_update_agent_wise(self.iterations, group, info=info, method="update_rnn_critic",
                                                   mask_values=mask_values,
                                                   q_eval_A_i=q_eval_A_i, q_eval_B_i=q_eval_B_i,
                                                   q_target=q_target, q_next_i=q_next_i,
                                                   td_error_A=td_error_A, td_error_B=td_error_B))

        # update actor(s)
        if self.iterations % self.actor_update_delay == 0:
            actions_eval = self.model(observations=obs_t, agent_indices=agent_indices_t,
                                      rnn_states=rnn_states_actor).actions
            for group, agent_keys in self.groups.items():
                n_agents = len(agent_keys)
                bs = batch_size * n_agents
                group_filled = filled.unsqueeze(1).expand(batch_size, n_agents, seq_len).reshape([bs, seq_len])
                mask_values = agent_mask[group] * group_filled
                act_eval = self.get_joint_input(actions_eval, (batch_size, seq_len, -1))
                _, _, q_policy = self.model.Qpolicy(joint_observations=obs_joint[:, :-1], joint_actions=act_eval,
                                                    group_key=group, agent_indices=agent_indices_t,
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
                info.update(
                    self.callback.on_update_agent_wise(self.iterations, group, info=info, method="update_rnn_actor",
                                                       mask_values=mask_values, q_policy_i=q_policy_i))
            self.model.soft_update(self.tau)
        info.update(self.callback.on_update_end(self.iterations, method="update_rnn", model=self.model, info=info))
        return info
