"""
Independent Deep Deterministic Policy Gradient (IDDPG)
Implementation: MindSpore
"""
from argparse import Namespace
from xuance.common import AgentGrouping

from xuance.mindspore import ms, ops, Module, Tensor, optim
from xuance.mindspore.utils import clip_grads, AgentGroupedTensor
from xuance.mindspore.learners import OffPolicyMultiAgentLearner
from xuance.mindspore.rl_models.modules import OffPolicyMARLBatch


class IDDPG_Learner(OffPolicyMultiAgentLearner):
    def __init__(self,
                 config: Namespace,
                 agent_grouping: AgentGrouping,
                 model: Module,
                 callback):
        super(IDDPG_Learner, self).__init__(config, agent_grouping, model, callback)
        self.tau = config.tau

        # Get gradient function
        self.grad_fn_actor = {
            key: ms.value_and_grad(self.forward_fn_actor, None, self.optimizer[key]['actor'].parameters, has_aux=True)
            for key in self.group_keys
        }
        self.grad_fn_critic = {
            key: ms.value_and_grad(self.forward_fn_critic, None, self.optimizer[key]['critic'].parameters, has_aux=True)
            for key in self.group_keys
        }
        self.model.set_train()

    def build_optimizer(self):
        self.optimizer = {
            key: {'actor': optim.Adam(
                params=self.model.actors[key].trainable_params(), lr=self.config.learning_rate_actor, eps=1e-5),
                'critic': optim.Adam(
                    params=self.model.critics[key].trainable_params(), lr=self.config.learning_rate_critic, eps=1e-5)}
            for key in self.group_keys
        }
        self.scheduler = {
            key: {'actor': optim.lr_scheduler.LinearLR(self.optimizer[key]['actor'],
                                                       start_factor=1.0,
                                                       end_factor=self.end_factor_lr_decay,
                                                       total_iters=self.config.running_steps),
                  'critic': optim.lr_scheduler.LinearLR(self.optimizer[key]['critic'],
                                                        start_factor=1.0,
                                                        end_factor=self.end_factor_lr_decay,
                                                        total_iters=self.config.running_steps)}
            for key in self.group_keys
        }

    def forward_fn_actor(self, batch: OffPolicyMARLBatch, *args, **kwargs):
        observations_t, agent_indices_t, rnn_states_actor, rnn_states_critic, group_key = args
        actions_eval = self.model(observations=observations_t,
                                  agent_indices=agent_indices_t,
                                  group_key=group_key,
                                  rnn_states=rnn_states_actor).actions
        q_policy = self.model.Qpolicy(observations=observations_t,
                                      actions=actions_eval,
                                      agent_indices=agent_indices_t,
                                      group_key=group_key,
                                      rnn_states=rnn_states_critic)

        #########################################
        # Calculate loss
        #########################################
        q_policy_i = q_policy.packed(group_key).reshape(-1)
        mask_values = batch.valid_mask(group_key, self.n_group_agents[group_key]).reshape(-1)
        loss_actor = -(q_policy_i * mask_values).sum() / mask_values.sum()

        return loss_actor, q_policy_i

    def forward_fn_critic(self, batch: OffPolicyMARLBatch, *args, **kwargs):
        observations_t, agent_indices_t, rnn_states_actor, rnn_states_critic, group_key = args
        q_eval = self.model.Qpolicy(observations=observations_t,
                                    actions=batch.actions,
                                    agent_indices=agent_indices_t,
                                    group_key=group_key,
                                    rnn_states=rnn_states_critic)

        #########################################
        # Calculate values with target networks
        #########################################
        if self.use_rnn:
            next_actions = self.model.Atarget(observations=batch.observations,
                                              agent_indices=batch.agent_indices,
                                              group_key=group_key,
                                              rnn_states=rnn_states_actor)
            q_next = self.model.Qtarget(observations=batch.observations,
                                        actions=next_actions,
                                        agent_indices=batch.agent_indices,
                                        group_key=group_key,
                                        rnn_states=rnn_states_critic)
            q_next.grouped_tensor = {k: v[:, :, 1:] for k, v in q_next.grouped_tensor.items()}
        else:
            next_actions = self.model.Atarget(observations=batch.next_observations,
                                              agent_indices=batch.agent_indices,
                                              group_key=group_key)
            q_next = self.model.Qtarget(observations=batch.next_observations,
                                        actions=next_actions,
                                        agent_indices=batch.agent_indices,
                                        group_key=group_key)

        #########################################
        # Calculate loss
        #########################################
        mask_values = batch.valid_mask(group_key, self.n_group_agents[group_key]).reshape(-1)
        q_eval_i = q_eval.packed(group_key).reshape(-1)

        # update critic
        rewards = batch.rewards.packed(group_key).reshape(-1)
        terminals = batch.terminals.packed(group_key).reshape(-1)
        q_target = rewards + (1 - terminals) * self.gamma * q_next.packed(group_key).reshape(-1)
        td_error = (q_eval_i - ops.stop_gradient(q_target)) * mask_values
        loss_critics = (td_error ** 2).sum() / mask_values.sum()

        return loss_critics, q_eval_i, td_error

    def update(self, sample):
        self.iterations += 1

        # prepare training data.
        batch = self.build_training_data(
            sample,
            use_actions_mask=False
        )

        info = self.callback.on_update_start(self.iterations, model=self.model, batch=batch)

        ######################
        # Feedforward
        ######################
        rnn_states_actor = self.model.init_actor_rnn_states(batch.batch_size)
        rnn_states_critic = self.model.init_critic_rnn_states(batch.batch_size)
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

        for group_key in self.group_keys:
            # update critics
            (loss_c, q_eval, td_error), grads_critic = self.grad_fn_critic[group_key](
                batch, observations_t, agent_indices_t, rnn_states_actor, rnn_states_critic, group_key)
            if self.use_grad_clip:
                grads_critic = clip_grads(grads_critic, Tensor(-self.grad_clip_norm), Tensor(self.grad_clip_norm))
            self.optimizer[group_key]['critic'](grads_critic)

            # update actors
            (loss_a, q_eval_i), grads_actor = self.grad_fn_actor[group_key](
                batch, observations_t, agent_indices_t, rnn_states_actor, rnn_states_critic, group_key)
            if self.use_grad_clip:
                grads_actor = clip_grads(grads_actor, Tensor(-self.grad_clip_norm), Tensor(self.grad_clip_norm))
            self.optimizer[group_key]['actor'](grads_actor)

            self.scheduler[group_key]['actor'].step()
            self.scheduler[group_key]['critic'].step()
            learning_rate_actor = self.scheduler[group_key]['actor'].get_last_lr()[0]
            learning_rate_critic = self.scheduler[group_key]['critic'].get_last_lr()[0]

            info.update({
                f"{group_key}/learning_rate_actor": learning_rate_actor.asnumpy(),
                f"{group_key}/learning_rate_critic": learning_rate_critic.asnumpy(),
                f"{group_key}/loss_actor": loss_a.asnumpy(),
                f"{group_key}/loss_critic": loss_c.asnumpy(),
                f"{group_key}/predictQ": q_eval_i.mean().asnumpy()
            })

        self.model.soft_update(self.tau)
        info.update(self.callback.on_update_end(self.iterations, model=self.model, info=info))
        return info
