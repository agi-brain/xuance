"""
Multi-Agent Deep Deterministic Policy Gradient
Paper link:
https://proceedings.neurips.cc/paper/2017/file/68a9750337a418a86fe06c1991a1d64c-Paper.pdf
Implementation: Pytorch
Trick: Parameter sharing for all agents, with agents' one-hot IDs as actor-critic's inputs.
"""
import torch
from typing import Optional
from xuance.torch.learners.multi_agent_rl.iddpg_learner import IDDPG_Learner
from xuance.torch.rl_models.modules import OffPolicyMARLBatch


class MADDPG_Learner(IDDPG_Learner):

    def build_training_data(
            self,
            sample: Optional[dict],
            use_parameter_sharing: Optional[bool] = False,
            use_actions_mask: Optional[bool] = False,
            use_global_state: Optional[bool] = False,
            use_shared_rewards: Optional[bool] = False,
    ) -> OffPolicyMARLBatch:
        """
        Prepare the training data.

        Parameters:
            sample (dict): The raw sampled data.
            use_parameter_sharing (bool): Whether to use parameter sharing for individual agent models.
            use_actions_mask (bool): Whether to use actions mask for unavailable actions.
            use_global_state (bool): Whether to use global state.
            use_shared_rewards (bool)： Whether to use shared rewards for each agent.

        Returns:
            OffPolicyMARLBatch: The formatted sampled data.
        """
        batch_size = sample['batch_size']
        seq_length = sample['sequence_length'] if self.use_rnn else 1
        state, state_next, filled = None, None, None
        obs, actions, rewards, terminals, agent_mask = {}, {}, {}, {}, {}
        agent_indices = {}
        if self.use_rnn:
            obs_next, avail_actions_next = None, None
        else:
            obs_next = {}

        for agent in self.agent_keys:
            obs[agent] = torch.as_tensor(sample['obs'][agent], device=self.device)
            if not self.use_rnn:
                obs_next[agent] = torch.as_tensor(sample['obs_next'][agent], device=self.device)
            actions[agent] = torch.as_tensor(sample['actions'][agent], device=self.device)
            rewards[agent] = torch.as_tensor(sample['rewards'][agent], device=self.device)
            terminals[agent] = torch.as_tensor(sample['terminals'][agent], device=self.device, dtype=torch.float32)
            agent_mask[agent] = torch.as_tensor(sample['agent_mask'][agent], device=self.device, dtype=torch.float32)

        if self.use_rnn:
            filled = torch.as_tensor(sample['filled'], device=self.device, dtype=torch.float32)

        for key, n_agents in self.n_group_agents.items():
            bs = batch_size * n_agents

            if self.use_rnn:
                agents_id = torch.as_tensor(self.agent_grouping.agent_indices(key), dtype=torch.int64).repeat(
                    batch_size, 1).reshape(bs, 1, 1).expand(-1, seq_length + 1, -1).to(self.device)
            else:
                agents_id = torch.as_tensor(self.agent_indices[key], dtype=torch.int64).repeat(
                    batch_size, 1).reshape([bs, 1]).to(self.device)

            agent_indices[key] = agents_id

        if not self.agent_grouping.full_independent:
            if not use_shared_rewards:
                rewards = self.packed_tensor(rewards)
                terminals = self.packed_tensor(terminals)
            agent_mask = self.packed_tensor(agent_mask)

        return OffPolicyMARLBatch(
            batch_size=batch_size,
            observations=obs,
            actions=actions,
            next_observations=obs_next,
            rewards=rewards,
            terminals=terminals,
            agent_masks=agent_mask,
            agent_indices=agent_indices,
            filled_masks=filled,
            seq_length=seq_length,
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
        rnn_states_critic = self.model.init_critic_rnn_states(batch.batch_size)

        # get actions
        actions_eval = self.model(observations=observations_t,
                                  agent_indices=agent_indices,
                                  rnn_states=rnn_states_actor).actions

        q_eval = self.model.Qpolicy(joint_observations=obs_joint_t,
                                    joint_actions=actions_joint_t,
                                    agent_indices=agent_indices,
                                    rnn_states=rnn_states_critic)

        with torch.no_grad():
            actions_next = self.model.Atarget(observations=observations_next,
                                              agent_indices=agent_indices,
                                              rnn_states=rnn_states_actor)
            # get values
            if self.use_rnn:
                actions_joint_next = self.get_joint_input(actions_next, (batch.batch_size, batch.seq_length, -1))
            else:
                actions_joint_next = self.get_joint_input(actions_next, (batch.batch_size, -1))

            q_next = self.model.Qtarget(joint_observations=obs_joint_next,
                                        joint_actions=actions_joint_next,
                                        agent_indices=agent_indices,
                                        rnn_states=rnn_states_critic)

        for group, n_agents in self.n_group_agents.items():
            mask_values = batch.valid_mask(group, n_agents).reshape(-1)
            agent_keys = self.groups[group]

            # update critic
            q_eval_a = q_eval[group].reshape(-1)
            q_next_i = q_next[group].reshape(-1)
            rewards = batch.rewards[group].reshape(-1)
            terminals = batch.terminals[group].reshape(-1)

            q_target = rewards + (1 - terminals) * self.gamma * q_next_i
            td_error = (q_eval_a - q_target.detach()) * mask_values

            loss_critic = (td_error ** 2).sum() / mask_values.sum()

            self.optimizer[group]['critic'].zero_grad()
            loss_critic.backward()
            if self.use_grad_clip:
                torch.nn.utils.clip_grad_norm_(self.model.critics[group].parameters(), self.grad_clip_norm)
            self.optimizer[group]['critic'].step()
            if self.scheduler[group]['critic'] is not None:
                self.scheduler[group]['critic'].step()

            # update actor
            # calculate the objective of actor
            actions_all = batch.actions.copy()
            for key in agent_keys:
                actions_all[key] = actions_eval[key]
            if self.use_rnn:
                actions_joint_eval = self.get_joint_input(actions_all, (batch.batch_size, batch.seq_length, -1))
            else:
                actions_joint_eval = self.get_joint_input(actions_all, (batch.batch_size, -1))
            q_policy = self.model.Qpolicy(joint_observations=obs_joint_t,
                                          joint_actions=actions_joint_eval,
                                          agent_indices=agent_indices,
                                          group_key=group,
                                          rnn_states=rnn_states_critic)
            q_policy_i = q_policy[group].reshape(-1)
            loss_actor = -(q_policy_i * mask_values).sum() / mask_values.sum()

            # update actor network
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

            info.update(self.callback.on_update_agent_wise(self.iterations, group, info=info,
                                                           mask_values=mask_values, q_model_i=q_model_i,
                                                           actions_joint_eval=actions_joint_eval, q_eval_a=q_eval_a,
                                                           q_next_i=q_next_i,
                                                           q_target=q_target, td_error=td_error))

        self.model.soft_update(self.tau)
        info.update(self.callback.on_update_end(self.iterations, model=self.model, info=info))
        return info
