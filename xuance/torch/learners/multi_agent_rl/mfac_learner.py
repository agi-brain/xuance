"""
MFAC: Mean Field Actor-Critic
Paper link:
http://proceedings.mlr.press/v80/yang18d/yang18d.pdf
Implementation: Pytorch
"""
import torch
from torch import nn
from argparse import Namespace
from xuance.common import Optional, AgentGrouping
from xuance.torch.utils import AgentGroupedTensor
from xuance.torch.learners.multi_agent_rl.mappo_learner import MAPPO_Learner


class MFAC_Learner(MAPPO_Learner):
    def __init__(self,
                 config: Namespace,
                 agent_grouping: AgentGrouping,
                 model: nn.Module,
                 callback):
        super(MFAC_Learner, self).__init__(config, agent_grouping, model, callback)

    def build_actions_mean_input(self, sample: Optional[dict]):
        actions_mean_agent_wise = {
            agent: torch.as_tensor(sample['actions_mean'][agent], device=self.device)
            for agent in self.agent_keys
        }
        return AgentGroupedTensor.from_agent_wise(actions_mean_agent_wise, self.agent_grouping)

    def update(self, sample):
        self.iterations += 1

        # prepare training data
        act_mean = self.build_actions_mean_input(sample=sample)
        batch = self.build_training_data(
            sample=sample,
            use_actions_mask=self.use_actions_mask
        )

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
            mean_actions=act_mean,
            agent_indices=batch.agent_indices,
            rnn_states=rnn_states_critic
        )
        values_pred = value_outputs.values

        # calculate losses for each agent
        actor_loss, entropy_loss, critic_loss = [], [], []
        for group, n_agents in self.n_group_agents.items():
            mask_values = batch.valid_mask(group, n_agents).reshape(-1)

            # actor loss
            dist = policy_outputs.distributions[group]
            log_pi = dist.log_prob(batch.actions.packed(group)).reshape(-1)
            advantages = batch.advantages.packed(group).reshape(-1)
            old_log_prob = batch.old_log_probs.packed(group).reshape(-1)

            ratio = torch.exp(log_pi - old_log_prob)
            surrogate1 = ratio * advantages
            surrogate2 = torch.clip(
                ratio,
                1.0 - self.clip_range,
                1.0 + self.clip_range
            ) * advantages

            actor_loss.append(-(torch.min(surrogate1, surrogate2) * mask_values).sum() / mask_values.sum())

            # entropy loss
            entropy = dist.entropy().reshape(-1)
            entropy_loss.append((entropy * mask_values).sum() / mask_values.sum())

            # critic loss
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
                loss_c_ = torch.max(loss_v, loss_v_clipped)
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
                f"{group}/actor_loss": actor_loss[-1].item(),
                f"{group}/critic_loss": critic_loss[-1].item(),
                f"{group}/entropy": entropy_loss[-1].item(),
                f"{group}/predict_value": value_pred_i.mean().item()
            })
            info.update(self.callback.on_update_agent_wise(self.iterations, group, info=info,
                                                           mask_values=mask_values, log_pi=log_pi, ratio=ratio,
                                                           surrogate1=surrogate1, surrogate2=surrogate2,
                                                           entropy=entropy,
                                                           value_pred_i=value_pred_i, value_target=value_target,
                                                           values_i=values_i, loss_v=loss_v))

        loss = sum(actor_loss) + self.vf_coef * sum(critic_loss) - self.ent_coef * sum(entropy_loss)
        self.optimizer.zero_grad()
        loss.backward()
        if self.use_grad_clip:
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.grad_clip_norm
            )
            info["gradient_norm"] = grad_norm.item()
        self.optimizer.step()
        if self.scheduler is not None and self.use_linear_lr_decay:
            self.scheduler.step()

        # Logger
        lr = self.optimizer.state_dict()['param_groups'][0]['lr']

        info.update({
            "learning_rate": lr,
            "loss": loss.item(),
        })

        info.update(self.callback.on_update_end(self.iterations, model=self.model, info=info))

        return info
