"""
Independent TD3 for multi-agent cooperative task
"""
import torch
from torch import nn
from argparse import Namespace
from xuance.common import AgentGrouping
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

        rnn_states_actor = self.model.init_actor_rnn_states(batch.batch_size)
        rnn_states_critic_1, rnn_states_critic_2 = self.model.init_critic_rnn_states(batch.batch_size)

        q_eval_A, q_eval_B, _ = self.model.Qpolicy(observations=observations_t,
                                                   actions=batch.actions,
                                                   agent_indices=agent_indices,
                                                   rnn_states_1=rnn_states_critic_1,
                                                   rnn_states_2=rnn_states_critic_2)
        with torch.no_grad():
            next_actions = self.model.Atarget(observations=observations_next,
                                              agent_indices=agent_indices,
                                              rnn_states=rnn_states_actor)
            if not self.agent_grouping.full_independent:
                next_actions = self.packed_tensor(next_actions)

            q_next = self.model.Qtarget(observations=observations_next,
                                        actions=next_actions,
                                        agent_indices=agent_indices,
                                        rnn_states_1=rnn_states_critic_1,
                                        rnn_states_2=rnn_states_critic_2)

        # update critic(s)
        for group, n_agents in self.n_group_agents.items():
            mask_values = batch.valid_mask(group, n_agents).reshape(-1)

            q_eval_A_i, q_eval_B_i = q_eval_A[group].reshape(-1), q_eval_B[group].reshape(-1)
            q_next_i = q_next[group].reshape(-1)
            rewards_i = batch.rewards[group].reshape(-1)
            terminals_i = batch.terminals[group].reshape(-1)
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
            if not self.agent_grouping.full_independent:
                actions_eval = self.packed_tensor(actions_eval)

            for group, n_agents in self.n_group_agents.items():
                mask_values = batch.valid_mask(group, n_agents).reshape(-1)

                _, _, q_policy = self.model.Qpolicy(observations=observations_t,
                                                    actions=actions_eval,
                                                    agent_indices=agent_indices,
                                                    grou_key=group,
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
