"""
Independent Advantage Actor Critic (IAC)
Paper link: https://ojs.aaai.org/index.php/AAAI/article/view/11794
Implementation: MindSpore
"""
from xuance.mindspore import ops, Tensor
from xuance.mindspore.rl_models.modules import OnPolicyMARLBatch
from xuance.mindspore.learners import OnPolicyMultiAgentLearner
from xuance.mindspore.utils import clip_grads


class IAC_Learner(OnPolicyMultiAgentLearner):

    def forward_fn(self, batch: OnPolicyMARLBatch, rnn_states_actor, rnn_states_critic):
        info_train = {}

        # feedforward
        policy_outputs = self.model(
            observations=batch.observations,
            agent_indices=batch.agent_indices,
            avail_actions=batch.avail_actions,
            rnn_states=rnn_states_actor
        )
        value_outputs = self.model.get_values(
            observations=batch.observations,
            agent_indices=batch.agent_indices,
            rnn_states=rnn_states_critic
        )
        values_pred = value_outputs.values

        # calculate losses and update networks for each group of agents
        loss_groups = []
        for group, n_agents in self.n_group_agents.items():
            mask_values = batch.valid_mask(group, n_agents).reshape(-1)

            # actor loss
            dist = policy_outputs.distributions[group]
            log_pi = dist.log_prob(batch.actions.packed(group)).reshape(-1)
            advantages = batch.advantages.group(group).reshape(-1)
            advantages = ops.stop_gradient(advantages)

            actor_loss = -((advantages * log_pi) * mask_values).sum() / mask_values.sum()

            # entropy loss
            entropy = dist.entropy().reshape(-1)
            entropy_loss = (entropy * mask_values).sum() / mask_values.sum()

            # value loss
            value_pred_i = values_pred.packed(group).reshape(-1)
            value_target = batch.returns.packed(group).reshape(-1)
            values_i = batch.values.packed(group).reshape(-1)
            if self.use_value_clip:
                value_clipped = values_i + (value_pred_i - values_i).clamp(
                    -self.value_clip_range,
                    self.value_clip_range
                )
                if self.use_value_norm:
                    self.value_normalizer[group].update(value_target.reshape(-1, 1))
                    value_target = self.value_normalizer[group].normalize(value_target.reshape(-1, 1))
                    value_target = value_target.reshape(-1)
                if self.use_huber_loss:
                    loss_v = self.huber_loss(value_pred_i, value_target)
                    loss_v_clipped = self.huber_loss(value_clipped, value_target)
                else:
                    loss_v = (value_pred_i - value_target) ** 2
                    loss_v_clipped = (value_clipped - value_target) ** 2
                loss_c_ = ops.max(loss_v, loss_v_clipped)
                critic_loss = (loss_c_ * mask_values).sum() / mask_values.sum()
            else:
                if self.use_value_norm:
                    self.value_normalizer[group].update(value_target)
                    value_target = self.value_normalizer[group].normalize(value_target)
                if self.use_huber_loss:
                    loss_v = self.huber_loss(value_pred_i, value_target)
                else:
                    loss_v = (value_pred_i - value_target) ** 2
                critic_loss = (loss_v * mask_values).sum() / mask_values.sum()

            loss_i = actor_loss + self.vf_coef * critic_loss - self.ent_coef * entropy_loss
            loss_groups.append(loss_i)

            info_train.update({
                f"{group}/actor_loss": actor_loss.asnumpy(),
                f"{group}/critic_loss": critic_loss.asnumpy(),
                f"{group}/entropy": entropy_loss.asnumpy(),
                f"{group}/total_loss": loss_i.asnumpy(),
                f"{group}/predict_value": value_pred_i.mean().asnumpy(),
            })

        loss = sum(loss_groups)

        return loss, info_train

    def update(self, sample):
        self.iterations += 1

        # prepare training data
        batch = self.build_training_data(
            sample=sample,
            use_actions_mask=self.use_actions_mask
        )

        info = self.callback.on_update_start(self.iterations, model=self.model, batch=batch)

        # initial hidden states for rnn
        rnn_states_actor = self.model.init_actor_rnn_states(batch.batch_size)
        rnn_states_critic = self.model.init_critic_rnn_states(batch.batch_size)

        (loss, info_train), grads = self.grad_fn(batch, rnn_states_actor, rnn_states_critic)

        if self.use_grad_clip:
            grads = clip_grads(grads, Tensor(-self.grad_clip_norm), Tensor(self.grad_clip_norm))
        self.optimizer(grads)

        self.scheduler.step()
        lr = self.scheduler.get_last_lr()[0]

        info.update(info_train)
        info["learning_rate"] = lr.asnumpy()

        info.update(self.callback.on_update_end(self.iterations, model=self.model, info=info))

        return info
