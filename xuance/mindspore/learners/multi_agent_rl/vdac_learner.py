"""
Value Decomposition Networks (VDN)
Paper link: https://arxiv.org/pdf/1706.05296.pdf
Implementation: MindSpore
"""
from argparse import Namespace
from xuance.common import AgentGrouping

from xuance.mindspore import ops, Tensor, Module
from xuance.mindspore.rl_models.modules import OnPolicyMARLBatch, AgentGroupedTensor
from xuance.mindspore.learners import OnPolicyMultiAgentLearner
from xuance.mindspore.utils import clip_grads


class VDAC_Learner(OnPolicyMultiAgentLearner):
    def __init__(self,
                 config: Namespace,
                 agent_grouping: AgentGrouping,
                 model: Module,
                 callback):
        super(VDAC_Learner, self).__init__(config, agent_grouping, model, callback)
        self.use_global_state = True if config.mixer == "QMIX" else getattr(config, "use_global_state", False)

    def forward_fn(self, batch: OnPolicyMARLBatch, *args, **kwargs):
        rnn_states_actor, rnn_states_critic = args
        info = {}

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

        values_pred_individual = value_outputs.values
        values_tot = self.model.values_tot(individual_values=values_pred_individual.agent_wise,
                                           global_states=batch.global_states)
        values_pred = AgentGroupedTensor.from_agent_wise({k: values_tot for k in self.agent_keys},
                                                         grouping=self.agent_grouping)

        # calculate losses for each agent
        actor_loss, entropy_loss, critic_loss = [], [], []
        for group, n_agents in self.n_group_agents.items():
            mask_values = batch.valid_mask(group, n_agents).reshape(-1)

            # actor loss
            dist = policy_outputs.distributions[group]
            log_pi = dist.log_prob(batch.actions.packed(group)).reshape(-1)
            advantages = batch.advantages.packed(group).reshape(-1)
            advantages = ops.stop_gradient(advantages)

            actor_loss.append(-((advantages * log_pi) * mask_values).sum() / mask_values.sum())

            # entropy loss
            entropy = dist.entropy().reshape(-1)
            entropy_loss.append((entropy * mask_values).sum() / mask_values.sum())

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
                loss_c_ = ops.maximum(loss_v, loss_v_clipped)
                loss_c = (loss_c_ * mask_values).sum() / mask_values.sum()
            else:
                if self.use_value_norm:
                    self.value_normalizer[group].update(value_target)
                    value_target = self.value_normalizer[group].normalize(value_target)
                if self.use_huber_loss:
                    loss_v = self.huber_loss(value_pred_i, value_target)
                else:
                    loss_v = (value_pred_i - value_target) ** 2
                loss_c = (loss_v * mask_values).sum() / mask_values.sum()

            critic_loss.append(loss_c)

            info.update({
                f"{group}/actor_loss": actor_loss[-1].asnumpy(),
                f"{group}/critic_loss": critic_loss[-1].asnumpy(),
                f"{group}/entropy": entropy_loss[-1].asnumpy(),
                f"{group}/predict_value": value_pred_i.mean().asnumpy()
            })
            info.update(self.callback.on_update_agent_wise(self.iterations, group, info=info,
                                                           mask_values=mask_values, log_pi=log_pi,
                                                           actor_loss=actor_loss,
                                                           entropy=entropy, entropy_loss=entropy_loss,
                                                           value_pred_i=value_pred_i, value_target=value_target,
                                                           values_i=values_i, loss_v=loss_v))

        # calculate the loss
        loss = sum(actor_loss) + self.vf_coef * sum(critic_loss) - self.ent_coef * sum(entropy_loss)

        return loss, info

    def update(self, sample):
        self.iterations += 1

        # prepare training data
        batch = self.build_training_data(
            sample=sample,
            use_actions_mask=self.use_actions_mask,
            use_global_state=self.use_global_state
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
        info.update({
            "learning_rate": lr.asnumpy(),
            "loss": loss.asnumpy(),
        })

        info.update(self.callback.on_update_end(self.iterations, model=self.model, info=info))

        return info
