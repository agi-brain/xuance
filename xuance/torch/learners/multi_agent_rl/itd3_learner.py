"""
Independent TD3 for multi-agent cooperative task
"""
import torch
from torch import nn
from argparse import Namespace
from xuance.common import AgentGrouping
from xuance.torch.utils import AgentGroupedTensor
from xuance.torch.learners import OffPolicyMultiAgentLearner


class ITD3_Learner(OffPolicyMultiAgentLearner):
    def __init__(self,
                 config: Namespace,
                 agent_grouping: AgentGrouping,
                 model: nn.Module,
                 callback):
        super(ITD3_Learner, self).__init__(config, agent_grouping, model, callback)
        self.tau = config.tau
        self.actor_update_delay = config.actor_update_delay

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

        # prepare training data
        batch = self.build_training_data(
            sample,
            use_actions_mask=False
        )

        info = self.callback.on_update_start(self.iterations, model=self.model, batch=batch)

        rnn_states_actor = self.model.init_actor_rnn_states(batch.batch_size)
        rnn_states_critic_1, rnn_states_critic_2 = self.model.init_critic_rnn_states(batch.batch_size)

        if self.use_rnn:
            observations_t = AgentGroupedTensor(
                {k: v[:, :, :-1] for k, v in batch.observations.grouped_tensor.items()}, self.agent_grouping
            )
            agent_indices_t = AgentGroupedTensor(
                {k: v[:, :, :-1] for k, v in batch.agent_indices.grouped_tensor.items()}, self.agent_grouping
            )
        else:
            observations_t = batch.observations
            agent_indices_t = batch.agent_indices

        # feedforward
        q_eval_A, q_eval_B, _ = self.model.Qpolicy(observations=observations_t,
                                                   actions=batch.actions,
                                                   agent_indices=agent_indices_t,
                                                   rnn_states_1=rnn_states_critic_1,
                                                   rnn_states_2=rnn_states_critic_2)
        with torch.no_grad():
            if self.use_rnn:
                next_actions = self.model.Atarget(observations=batch.observations,
                                                  agent_indices=batch.agent_indices,
                                                  rnn_states=rnn_states_actor)
                q_next = self.model.Qtarget(observations=batch.observations,
                                            actions=next_actions,
                                            agent_indices=batch.agent_indices,
                                            rnn_states_1=rnn_states_critic_1,
                                            rnn_states_2=rnn_states_critic_2)
                q_next.grouped_tensor = {k: v[:, :, 1:] for k, v in q_next.grouped_tensor.items()}
            else:
                next_actions = self.model.Atarget(observations=batch.next_observations,
                                                  agent_indices=batch.agent_indices)
                q_next = self.model.Qtarget(observations=batch.next_observations,
                                            actions=next_actions,
                                            agent_indices=batch.agent_indices)

        # update critic(s)
        for group, n_agents in self.n_group_agents.items():
            mask_values = batch.valid_mask(group, n_agents).reshape(-1)

            q_eval_A_i, q_eval_B_i = q_eval_A.packed(group).reshape(-1), q_eval_B.packed(group).reshape(-1)
            q_next_i = q_next.packed(group).reshape(-1)
            rewards_i = batch.rewards.packed(group).reshape(-1)
            terminals_i = batch.terminals.packed(group).reshape(-1)
            q_target = rewards_i + (1 - terminals_i) * self.gamma * q_next_i

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
            for group, n_agents in self.n_group_agents.items():
                mask_values = batch.valid_mask(group, n_agents).reshape(-1)

                _, _, q_policy = self.model.Qpolicy(observations=observations_t,
                                                    actions=actions_eval,
                                                    agent_indices=agent_indices_t,
                                                    grou_key=group,
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
