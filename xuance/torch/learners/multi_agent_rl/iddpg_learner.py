"""
Independent Deep Deterministic Policy Gradient (IDDPG)
Implementation: Pytorch
"""
import torch
from torch import nn
from argparse import Namespace
from xuance.common import AgentGrouping
from xuance.torch.learners import OffPolicyMultiAgentLearner
from xuance.torch.rl_models.modules import OffPolicyMARLBatch


class IDDPG_Learner(OffPolicyMultiAgentLearner):
    def __init__(self,
                 config: Namespace,
                 agent_grouping: AgentGrouping,
                 model: nn.Module,
                 callback):
        super(IDDPG_Learner, self).__init__(config, agent_grouping, model, callback)
        self.tau = config.tau

    def build_optimizer(self):
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

    def _forward_transition(self, batch: OffPolicyMARLBatch):
        rnn_states_actor = self.model.init_actor_rnn_states(batch.batch_size)
        rnn_states_critic = self.model.init_critic_rnn_states(batch.batch_size)
        if self.use_rnn:
            obs_t = {k: v[:, :-1] for k, v in batch.observations.items()}
            agent_indices_t = {k: v[:, :-1] for k, v in batch.agent_indices.items()}
        else:
            obs_t = batch.observations
            agent_indices_t = batch.agent_indices

        actions_eval = self.model(
            observations=batch.observations,
            agent_indices=batch.agent_indices,
            rnn_states=rnn_states_actor
        ).actions
        if not self.agent_grouping.full_independent:
            actions_eval = self.packed_tensor(actions_eval)

        q_model = self.model.Qpolicy(
            observations=batch.observations,
            actions=actions_eval,
            agent_indices=batch.agent_indices,
            rnn_states=rnn_states_critic
        )
        q_eval = self.model.Qpolicy(
            observations=obs_t,
            actions=batch.actions,
            agent_indices=agent_indices_t,
            rnn_states=rnn_states_critic
        )
        with torch.no_grad():
            if self.use_rnn:
                next_actions = self.model.Atarget(
                    observations=batch.observations,
                    agent_indices=batch.agent_indices,
                    rnn_states=rnn_states_actor
                )
                if not self.agent_grouping.full_independent:
                    next_actions = self.packed_tensor(next_actions)
                q_next = self.model.Qtarget(
                    observations=batch.observations,
                    actions=next_actions,
                    agent_indices=batch.agent_indices,
                    rnn_states=rnn_states_critic
                )
                q_model = {k: v[:, :-1] for k, v in q_model.items()}
                q_next = {k: v[:, 1:] for k, v in q_next.items()}
            else:
                next_actions = self.model.Atarget(
                    observations=batch.next_observations,
                    agent_indices=batch.agent_indices
                )
                if not self.agent_grouping.full_independent:
                    next_actions = self.packed_tensor(next_actions)
                q_next = self.model.Qtarget(
                    observations=batch.next_observations,
                    actions=next_actions,
                    agent_indices=batch.agent_indices
                )

        return q_model, q_eval, q_next

    def update(self, sample):
        self.iterations += 1

        # prepare training data.
        batch = self.build_training_data(
            sample,
            use_parameter_sharing=self.use_parameter_sharing,
            use_actions_mask=False
        )

        info = self.callback.on_update_start(self.iterations, model=self.model, batch=batch)

        # feedforward
        q_model, q_eval, q_next = self._forward_transition(batch)

        for group, n_agents in self.n_group_agents.items():
            mask_values = batch.valid_mask(group, n_agents).reshape(-1)

            # update actor
            loss_actor = (q_model[group].reshape(-1) * mask_values).sum() / mask_values.sum()
            self.optimizer[group]['actor'].zero_grad()
            loss_actor.backward()
            if self.use_grad_clip:
                torch.nn.utils.clip_grad_norm_(self.model.parameters_actor[group], self.grad_clip_norm)
            self.optimizer[group]['actor'].step()
            if self.scheduler[group]['actor'] is not None:
                self.scheduler[group]['actor'].step()

            # update critic
            rewards = batch.rewards[group].reshape(-1)
            terminals = batch.terminals[group].reshape(-1)
            q_target = rewards + (1 - terminals) * self.gamma * q_next[group].reshape(-1)
            td_error = (q_eval[group].reshape(-1) - q_target.detach()) * mask_values
            loss_critic = (td_error ** 2).sum() / mask_values.sum()
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
                f"{group}/predictQ": q_eval[group].mean().item()
            })

            info.update(self.callback.on_update_agent_wise(self.iterations, group, info=info,
                                                           mask_values=mask_values, q_model_i=q_model[group],
                                                           q_eval_i=q_eval[group], q_next_i=q_next[group],
                                                           q_target=q_target, td_error=td_error))

        self.model.soft_update(self.tau)
        info.update(self.callback.on_update_end(self.iterations, model=self.model, info=info))
        return info
