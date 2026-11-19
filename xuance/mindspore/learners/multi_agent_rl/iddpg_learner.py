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
        self.grad_fn_actor = ms.value_and_grad(self.forward_fn_actor, None, self.optimizer['actor'].parameters,
                                               has_aux=False)
        self.grad_fn_critic = ms.value_and_grad(self.forward_fn_critic, None, self.optimizer['critic'].parameters,
                                                has_aux=True)
        self.model.set_train()

    def build_optimizer(self):
        self.optimizer = {
            'actor': optim.Adam(
                params=self.model.actors.trainable_params(), lr=self.config.learning_rate_actor, eps=1e-5),
            'critic': optim.Adam(
                params=self.model.critics.trainable_params(), lr=self.config.learning_rate_critic, eps=1e-5)
        }
        self.scheduler = {
            'actor': optim.lr_scheduler.LinearLR(self.optimizer['actor'],
                                                 start_factor=1.0,
                                                 end_factor=self.end_factor_lr_decay,
                                                 total_iters=self.config.running_steps),
            'critic': optim.lr_scheduler.LinearLR(self.optimizer['critic'],
                                                  start_factor=1.0,
                                                  end_factor=self.end_factor_lr_decay,
                                                  total_iters=self.config.running_steps)
        }

    def forward_fn_actor(self, batch: OffPolicyMARLBatch,
                         observations_t, agent_indices_t, rnn_states_actor, rnn_states_critic):
        actions_eval = self.model(observations=observations_t,
                                  agent_indices=agent_indices_t,
                                  rnn_states=rnn_states_actor).actions
        q_policy = self.model.Qpolicy(observations=observations_t,
                                      actions=actions_eval,
                                      agent_indices=agent_indices_t,
                                      rnn_states=rnn_states_critic)

        #########################################
        # Calculate loss and backpropogation
        #########################################
        loss_actors = []
        for group, n_agents in self.n_group_agents.items():
            mask_values = batch.valid_mask(group, n_agents).reshape(-1)

            # update actor
            loss_actors.append(-(q_policy.packed(group).reshape(-1) * mask_values).sum() / mask_values.sum())

        loss_a = sum(loss_actors)
        return loss_a

    def forward_fn_critic(self, batch: OffPolicyMARLBatch,
                          observations_t, agent_indices_t, rnn_states_actor, rnn_states_critic):
        q_eval = self.model.Qpolicy(observations=observations_t,
                                    actions=batch.actions,
                                    agent_indices=agent_indices_t,
                                    rnn_states=rnn_states_critic)

        #########################################
        # Calculate values with target networks
        #########################################
        if self.use_rnn:
            next_actions = self.model.Atarget(observations=batch.observations,
                                              agent_indices=batch.agent_indices,
                                              rnn_states=rnn_states_actor)
            q_next = self.model.Qtarget(observations=batch.observations,
                                        actions=next_actions,
                                        agent_indices=batch.agent_indices,
                                        rnn_states=rnn_states_critic)
            q_next.grouped_tensor = {k: v[:, :, 1:] for k, v in q_next.grouped_tensor.items()}
        else:
            next_actions = self.model.Atarget(observations=batch.next_observations,
                                              agent_indices=batch.agent_indices)
            q_next = self.model.Qtarget(observations=batch.next_observations,
                                        actions=next_actions,
                                        agent_indices=batch.agent_indices,
                                        rnn_states=rnn_states_critic)

        #########################################
        # Calculate loss and backpropogation
        #########################################
        loss_critics, td_errors = [], []
        for group, n_agents in self.n_group_agents.items():
            mask_values = batch.valid_mask(group, n_agents).reshape(-1)

            # update critic
            rewards = batch.rewards.packed(group).reshape(-1)
            terminals = batch.terminals.packed(group).reshape(-1)
            q_target = rewards + (1 - terminals) * self.gamma * q_next.packed(group).reshape(-1)
            td_error = (q_eval.packed(group).reshape(-1) - ops.stop_gradient(q_target)) * mask_values
            td_errors.append(td_error)
            loss_critics.append((td_error ** 2).sum() / mask_values.sum())

        loss_c = sum(loss_critics)
        return loss_c, q_eval, td_errors

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

        # update critics
        (loss_c, q_eval, td_error), grads_critic = self.grad_fn_critic(
            batch, observations_t, agent_indices_t, rnn_states_actor, rnn_states_critic)
        if self.use_grad_clip:
            grads_critic = clip_grads(grads_critic, Tensor(-self.grad_clip_norm), Tensor(self.grad_clip_norm))
        self.optimizer['critic'](grads_critic)

        # update actors
        loss_a, grads_actor = self.grad_fn_actor(
            batch, observations_t, agent_indices_t, rnn_states_actor, rnn_states_critic)
        if self.use_grad_clip:
            grads_actor = clip_grads(grads_actor, Tensor(-self.grad_clip_norm), Tensor(self.grad_clip_norm))
        self.optimizer['actor'](grads_actor)

        self.scheduler['actor'].step()
        self.scheduler['critic'].step()
        learning_rate_actor = self.scheduler['actor'].get_last_lr()[0]
        learning_rate_critic = self.scheduler['critic'].get_last_lr()[0]

        info.update({
            "learning_rate_actor": learning_rate_actor.asnumpy(),
            "learning_rate_critic": learning_rate_critic.asnumpy(),
            "loss_actor": loss_a.asnumpy(),
            "loss_critic": loss_c.asnumpy()
        })
        info.update({f"{group}/predictQ": q_eval.group(group).mean().asnumpy() for group in self.group_keys})

        self.model.soft_update(self.tau)
        info.update(self.callback.on_update_end(self.iterations, model=self.model, info=info))
        return info
