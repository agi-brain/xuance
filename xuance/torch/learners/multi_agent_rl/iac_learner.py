"""
Independent Advantage Actor Critic (IAC)
Paper link: https://ojs.aaai.org/index.php/AAAI/article/view/11794
Implementation: Pytorch
"""
import torch
from xuance.torch.learners import OnPolicyMultiAgentLearner


class IAC_Learner(OnPolicyMultiAgentLearner):

    def build_optimizer(self):
        self.optimizer = {
            group: torch.optim.Adam(
                self.model.parameters_model[group],
                lr=self.learning_rate,
                eps=1e-5,
                weight_decay=self.config.weight_decay
            )
            for group in self.group_keys
        }
        self.scheduler = {
            group: torch.optim.lr_scheduler.LinearLR(
                self.optimizer[group],
                start_factor=1.0,
                end_factor=self.end_factor_lr_decay,
                total_iters=self.total_iters
            )
            for group in self.group_keys
        }

    def update(self, sample):
        self.iterations += 1

        # prepare training data
        batch = self.build_training_data(sample=sample,
                                         use_parameter_sharing=self.use_parameter_sharing,
                                         use_actions_mask=self.use_actions_mask)

        info = self.callback.on_update_start(self.iterations, model=self.model, batch=batch)

        # initial hidden states for rnn
        rnn_states_actor = self.model.init_actor_rnn_states(batch.batch_size)
        rnn_states_critic = self.model.init_critic_rnn_states(batch.batch_size)

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
        if not self.agent_grouping.full_independent:
            values_pred = self.packed_tensor(values_pred)

        # calculate losses and update networks for each group of agents
        for group, n_agents in self.n_group_agents.items():
            mask_values = batch.valid_mask(group, n_agents).reshape(-1)

            # actor loss
            dist = policy_outputs.distributions[group]
            log_pi = dist.log_prob(batch.actions[group]).reshape(-1)
            advantages = batch.advantages[group].reshape(-1)

            actor_loss = -((advantages.detach() * log_pi) * mask_values).sum() / mask_values.sum()

            # entropy loss
            entropy = dist.entropy().reshape(-1)
            entropy_loss = (entropy * mask_values).sum() / mask_values.sum()

            # value loss
            value_pred_i = values_pred[group].reshape(-1)
            value_target = batch.returns[group].reshape(-1)
            values_i = batch.values[group].reshape(-1)
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
                loss_c_ = torch.max(loss_v, loss_v_clipped)
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

            loss = actor_loss + self.vf_coef * critic_loss - self.ent_coef * entropy_loss

            self.optimizer[group].zero_grad()
            loss.backward()
            if self.use_grad_clip:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters_model[group],
                    self.grad_clip_norm
                )
                info["gradient_norm"] = grad_norm.item()
            self.optimizer[group].step()
            if self.scheduler is not None:
                self.scheduler[group].step()

            # Logger
            lr = self.optimizer[group].state_dict()['param_groups'][0]['lr']

            info.update({
                f"{group}/actor_loss": actor_loss.item(),
                f"{group}/critic_loss": critic_loss.item(),
                f"{group}/entropy": entropy_loss.item(),
                f"{group}/total_loss": loss.item(),
                f"{group}/predict_value": value_pred_i.mean().item(),
                f"{group}/learning_rate": lr,
            })

            info.update(self.callback.on_update_agent_wise(self.iterations, group, info=info,
                                                           mask_values=mask_values, log_pi=log_pi,
                                                           actor_loss=actor_loss,
                                                           entropy=entropy, entropy_loss=entropy_loss,
                                                           value_pred_i=value_pred_i, value_target=value_target,
                                                           values_i=values_i, loss_v=loss_v))

        info.update(self.callback.on_update_end(self.iterations, method="update", model=self.model, info=info))

        return info
