"""
Independent Soft Actor-critic (ISAC)
Implementation: MindSpore
"""
from argparse import Namespace
from xuance.common import AgentGrouping

from xuance.mindspore import ms, ops, optim, Module, Tensor
from xuance.mindspore.utils import clip_grads, AgentGroupedTensor
from xuance.mindspore.learners import OffPolicyMultiAgentLearner
from xuance.mindspore.rl_models.modules import OffPolicyMARLBatch


class ISAC_Learner(OffPolicyMultiAgentLearner):
    def __init__(self,
                 config: Namespace,
                 agent_grouping: AgentGrouping,
                 model: Module,
                 callback):
        super(ISAC_Learner, self).__init__(config, agent_grouping, model, callback)
        self.tau = config.tau
        self.alpha = {key: config.alpha for key in self.group_keys}
        self._ones = ops.Ones()
        self.use_automatic_entropy_tuning = config.use_automatic_entropy_tuning
        if self.use_automatic_entropy_tuning:
            self.target_entropy = self.get_target_entropy()
            self.log_alpha = {key: ms.Parameter(self._ones(1, ms.float32)) for key in self.group_keys}
            self.alpha = {key: ops.exp(self.log_alpha[key]) for key in self.group_keys}
            self.alpha_optimizer = {key: optim.Adam(params=[self.log_alpha[key]], lr=config.learning_rate_actor)
                                    for key in self.group_keys}
            # Get gradient function
            self.grad_fn_alpha = {key: ms.value_and_grad(self.forward_fn_alpha, None,
                                                         self.alpha_optimizer[key].parameters, has_aux=True)
                                  for key in self.group_keys}
        # Get gradient function
        self.grad_fn_actor = {key: ms.value_and_grad(self.forward_fn_actor, None,
                                                     self.optimizer[key]['actor'].parameters, has_aux=True)
                              for key in self.group_keys}
        self.grad_fn_critic = {key: ms.value_and_grad(self.forward_fn_critic, None,
                                                      self.optimizer[key]['critic'].parameters, has_aux=True)
                               for key in self.group_keys}

    def get_target_entropy(self):
        return {key: -self.model.actors[key].action_space.shape[-1] for key in self.group_keys}

    def build_optimizer(self):
        self.optimizer = {
            key: {
                'actor': optim.Adam(params=self.model.actors[key].trainable_params(),
                                    lr=self.config.learning_rate_actor,
                                    eps=1e-5),
                'critic': optim.Adam(params=self.model.critics[key].trainable_params(),
                                     lr=self.config.learning_rate_critic,
                                     eps=1e-5)}
            for key in self.group_keys}
        self.scheduler = {
            key: {'actor': optim.lr_scheduler.LinearLR(self.optimizer[key]['actor'], start_factor=1.0,
                                                       end_factor=self.end_factor_lr_decay,
                                                       total_iters=self.config.running_steps),
                  'critic': optim.lr_scheduler.LinearLR(self.optimizer[key]['critic'], start_factor=1.0,
                                                        end_factor=self.end_factor_lr_decay,
                                                        total_iters=self.config.running_steps)}
            for key in self.group_keys}

    def forward_fn_alpha(self, log_pi_eval_i, key):
        alpha_loss = -(self.log_alpha[key] * ops.stop_gradient((log_pi_eval_i + self.target_entropy[key]))).mean()
        return alpha_loss, self.log_alpha[key]

    def forward_fn_actor(self, batch: OffPolicyMARLBatch, *args, **kwargs):
        observations_t, agent_indices_t, rnn_states_actor, rnn_states_critic_1, rnn_states_critic_2, group_key = args

        model_output = self.model(observations=batch.observations,
                                  agent_indices=batch.agent_indices,
                                  group_key=group_key,
                                  rnn_states=rnn_states_actor)
        actions_eval = model_output.actions
        log_pi_eval = model_output.log_probs

        policy_q_1, policy_q_2 = self.model.Qpolicy(observations=observations_t,
                                                    actions=actions_eval,
                                                    agent_indices=agent_indices_t,
                                                    group_key=group_key,
                                                    rnn_states_1=rnn_states_critic_1,
                                                    rnn_states_2=rnn_states_critic_2)

        n_agents = self.n_group_agents[group_key]
        mask_values = batch.valid_mask(group_key, n_agents).reshape(-1)

        log_pi_eval_i = log_pi_eval.packed(group_key).reshape(-1)
        policy_q = ops.minimum(policy_q_1.packed(group_key), policy_q_2.packed(group_key)).reshape(-1)

        loss_a = ((self.alpha[group_key] * log_pi_eval_i - policy_q) * mask_values).sum() / mask_values.sum()
        return loss_a, log_pi_eval_i, policy_q, policy_q_1, policy_q_2

    def forward_fn_critic(self, batch: OffPolicyMARLBatch, *args, **kwargs):
        observations_t, agent_indices_t, rnn_states_actor, rnn_states_critic_1, rnn_states_critic_2, group_key = args

        model_output = self.model(observations=batch.observations,
                                  agent_indices=batch.agent_indices,
                                  group_key=group_key,
                                  rnn_states=rnn_states_actor)
        actions_eval = model_output.actions
        log_pi_eval = model_output.log_probs

        action_q_1, action_q_2 = self.model.Qpolicy(observations=observations_t,
                                                    actions=batch.actions,
                                                    agent_indices=agent_indices_t,
                                                    group_key=group_key,
                                                    rnn_states_1=rnn_states_critic_1,
                                                    rnn_states_2=rnn_states_critic_2)

        if self.use_rnn:
            q_next = self.model.Qtarget(observations=batch.observations,
                                        actions=actions_eval,
                                        agent_indices=batch.agent_indices,
                                        rnn_states_1=rnn_states_critic_1,
                                        rnn_states_2=rnn_states_critic_2)
            q_next.grouped_tensor = {k: v[:, :, 1:] for k, v in q_next.grouped_tensor.items()}
            log_pi_next = AgentGroupedTensor(
                {k: v[:, :, 1:] for k, v in log_pi_eval.grouped_tensor.items()}, self.agent_grouping
            )
            log_pi_eval.grouped_tensor = {k: v[:, :, :-1] for k, v in log_pi_eval.grouped_tensor.items()}
            actions_eval.grouped_tensor = {k: v[:, :, :-1] for k, v in actions_eval.grouped_tensor.items()}
        else:
            next_model_output = self.model(observations=batch.next_observations,
                                           agent_indices=batch.agent_indices)
            actions_next = next_model_output.actions
            log_pi_next = next_model_output.log_probs
            q_next = self.model.Qtarget(observations=batch.next_observations,
                                        actions=actions_next,
                                        agent_indices=batch.agent_indices)

        n_agents = self.n_group_agents[group_key]
        mask_values = batch.valid_mask(group_key, n_agents).reshape(-1)

        # update critic
        log_pi_next_eval = log_pi_next.packed(group_key).reshape(-1)
        next_q_i = q_next.packed(group_key).reshape(-1)
        target_value = next_q_i - self.alpha[group_key] * log_pi_next_eval
        rewards = batch.rewards.packed(group_key).reshape(-1)
        terminals = batch.terminals.packed(group_key).reshape(-1)
        backup = rewards + (1 - terminals) * self.gamma * target_value
        backup = ops.stop_gradient(backup)

        action_q_1_i = action_q_1.packed(group_key).reshape(-1)
        td_error_1 = action_q_1_i - backup
        td_error_1 *= mask_values

        action_q_2_i = action_q_2.packed(group_key).reshape(-1)
        td_error_2 = action_q_2_i - backup
        td_error_2 *= mask_values

        loss_c = ((td_error_1 ** 2).sum() + (td_error_2 ** 2).sum()) / mask_values.sum()

        return loss_c, action_q_1_i, action_q_2_i, td_error_1, td_error_2

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
        else:
            observations_t = batch.observations
            agent_indices_t = batch.agent_indices

        # initial hidden states for rnn
        rnn_states_actor = self.model.init_actor_rnn_states(batch.batch_size)
        rnn_states_critic_1, rnn_states_critic_2 = self.model.init_critic_rnn_states(batch.batch_size)

        for group_key in self.group_keys:
            # update critic
            (loss_c, action_q_1_i, action_q_2_i, td_error_1, td_error_2), grads_critic = self.grad_fn_critic[group_key](
                batch, observations_t, agent_indices_t, rnn_states_actor, rnn_states_critic_1, rnn_states_critic_2,
                group_key)
            if self.use_grad_clip:
                grads_critic = clip_grads(grads_critic, Tensor(-self.grad_clip_norm), Tensor(self.grad_clip_norm))
            self.optimizer[group_key]['critic'](grads_critic)

            # update actor
            (loss_a, log_pi_eval_i, policy_q, policy_q_1, policy_q_2), grads_actor = self.grad_fn_actor[group_key](
                batch, observations_t, agent_indices_t, rnn_states_actor, rnn_states_critic_1, rnn_states_critic_2,
                group_key)
            if self.use_grad_clip:
                grads_actor = clip_grads(grads_actor, Tensor(-self.grad_clip_norm), Tensor(self.grad_clip_norm))
            self.optimizer[group_key]['actor'](grads_actor)

            # automatic entropy tuning
            if self.use_automatic_entropy_tuning:
                (alpha_loss, _), grads_alpha = self.grad_fn_alpha[group_key](log_pi_eval_i, group_key)
                self.alpha_optimizer[group_key](grads_alpha)
                self.alpha[group_key] = ops.exp(self.log_alpha[group_key])
            else:
                alpha_loss = 0

            learning_rate_actor = self.scheduler[group_key]['actor'].get_last_lr()[0]
            learning_rate_critic = self.scheduler[group_key]['critic'].get_last_lr()[0]

            info.update({
                f"{group_key}/learning_rate_actor": learning_rate_actor.asnumpy(),
                f"{group_key}/learning_rate_critic": learning_rate_critic.asnumpy(),
                f"{group_key}/loss_actor": loss_a.asnumpy(),
                f"{group_key}/loss_critic": loss_c.asnumpy(),
                f"{group_key}/predictQ": policy_q.mean().asnumpy(),
                f"{group_key}/alpha_loss": alpha_loss.asnumpy(),
                f"{group_key}/alpha": self.alpha[group_key].asnumpy(),
            })

            info.update(self.callback.on_update_agent_wise(self.iterations, group_key, info=info,
                                                           action_q_1_i=action_q_1_i, action_q_2_i=action_q_2_i,
                                                           td_error_1=td_error_1, td_error_2=td_error_2,
                                                           policy_q_1=policy_q_1, policy_q_2=policy_q_2,
                                                           log_pi_eval_i=log_pi_eval_i, policy_q=policy_q))

        self.model.soft_update(self.tau)
        info.update(self.callback.on_update_end(self.iterations, model=self.model, info=info))
        return info
