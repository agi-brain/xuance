from typing import Tuple, Union

from . import DreamerV3Policy, dotdict

from xuance.torch.learners import Learner
import torch
from torch import nn
from argparse import Namespace
from torch.distributions.kl import kl_divergence
from torch.distributions import Distribution, Independent, OneHotCategoricalStraightThrough

class DreamerV3Learner(Learner):
    def __init__(self,
                 config: Namespace,
                 policy: DreamerV3Policy,
                 action_shape: Union[int, Tuple[int, ...]]):
        super(DreamerV3Learner, self).__init__(config, policy)
        self.policy = policy  # for code completion
        self.action_shape = action_shape

        # config
        self.config = dotdict(vars(config))
        self.is_continuous = self.config.is_continuous
        self.tau = self.config.critic.tau
        self.gamma = self.config.gamma
        self.soft_update_freq = self.config.critic.soft_update_freq

        self.kl_dynamic = self.config.world_model.kl_dynamic  # 0.5
        self.kl_representation = self.config.world_model.kl_representation  # 0.1
        self.kl_free_nats = self.config.world_model.kl_free_nats  # 1.0
        self.kl_regularizer = self.config.world_model.kl_regularizer  # 1.0
        self.continue_scale_factor = self.config.world_model.continue_scale_factor  # 1.0

        # optimizers
        self.optimizer = {
            'model': torch.optim.Adam(self.policy.world_model.parameters(), self.config.learning_rate_model),
            'actor': torch.optim.Adam(self.policy.actor.parameters(), self.config.learning_rate_actor),
            'critic': torch.optim.Adam(self.policy.critic.parameters(), self.config.learning_rate_critic)
        }

        self.gradient_step = 0

    def update(self, **samples):
        if self.gradient_step % self.soft_update_freq == 0:
            self.policy.soft_update(self.tau)
        # [seq, batch, ~]  # checked
        obs = torch.as_tensor(samples['obs'], device=self.device, dtype=torch.float32)
        acts = torch.as_tensor(samples['acts'], device=self.device)
        if not self.is_continuous:
            # acts to one_hot [seq, batch, action_size]
            acts = nn.functional.one_hot(acts.long(), num_classes=self.action_shape).float()
        rews = torch.as_tensor(samples['rews'], device=self.device)
        terms = torch.as_tensor(samples['terms'], device=self.device)
        truncs = torch.as_tensor(samples['truncs'], device=self.device)  # no use
        is_first = torch.as_tensor(samples['is_first'], device=self.device)
        """
        seq_shift
        (o1, a1 -> a0, r1, terms1, truncs1, is_first1)
        """
        is_first[0, :] = torch.ones_like(is_first[0, :])
        acts = torch.cat((torch.zeros_like(acts[:1]), acts[:-1]), 0)  # bug fixed ones_like -> zeros_like
        cont = 1 - terms

        po, pr, pc, priors_logits, posteriors_logits, recurrent_states, posteriors =\
            self.policy.model_forward(obs, acts, is_first)

        """model"""
        observation_loss = -po.log_prob(obs)
        reward_loss = -pr.log_prob(rews)
        # KL balancing
        dyn_loss = kl = kl_divergence(  # prior -> post
            Independent(OneHotCategoricalStraightThrough(logits=posteriors_logits.detach()), 1),
            Independent(OneHotCategoricalStraightThrough(logits=priors_logits), 1),
        )
        free_nats = torch.full_like(dyn_loss, self.config.world_model.kl_free_nats)
        dyn_loss = self.kl_dynamic * torch.maximum(dyn_loss, free_nats)
        repr_loss = kl_divergence(  # post -> prior
            Independent(OneHotCategoricalStraightThrough(logits=posteriors_logits), 1),
            Independent(OneHotCategoricalStraightThrough(logits=priors_logits.detach()), 1),
        )
        repr_loss = self.kl_representation * torch.maximum(repr_loss, free_nats)
        kl_loss = dyn_loss + repr_loss
        if pc is not None and cont is not None:
            continue_loss = self.continue_scale_factor * -pc.log_prob(cont)
        else:
            continue_loss = torch.zeros_like(reward_loss)
        model_loss = (self.kl_regularizer * kl_loss + observation_loss + reward_loss + continue_loss).mean()

        self.optimizer['model'].zero_grad()
        model_loss.backward()
        if self.config.world_model.clip_gradients is not None:
            torch.nn.utils.clip_grad_norm_(self.policy.world_model.parameters(), self.config.world_model.clip_gradients)
        self.optimizer['model'].step()

        """actor"""
        out = self.policy.actor_critic_forward(posteriors, recurrent_states, terms)
        objective, discount, entropy = out['for_actor']
        qv, predicted_target_values, lambda_values = out['for_critic']
        actor_loss = -torch.mean(discount[:-1].detach() * (objective + entropy.unsqueeze(dim=-1)[:-1]))

        self.optimizer['actor'].zero_grad()
        actor_loss.backward()
        if self.config.actor.clip_gradients is not None:
            torch.nn.utils.clip_grad_norm_(self.policy.actor.parameters(), self.config.actor.clip_gradients)
        self.optimizer['actor'].step()

        """critic"""
        critic_loss = -qv.log_prob(lambda_values.detach())
        critic_loss = critic_loss - qv.log_prob(predicted_target_values.detach())
        critic_loss = torch.mean(critic_loss * discount[:-1].squeeze(-1))
        self.optimizer['critic'].zero_grad()
        critic_loss.backward()
        if self.config.critic.clip_gradients is not None:
            torch.nn.utils.clip_grad_norm_(self.policy.critic.parameters(), self.config.critic.clip_gradients)
        self.optimizer['critic'].step()

        self.gradient_step += 1
        # if self.gradient_step % 100 == 0:
        #     print(f'gradient_step: {self.gradient_step}')

        info = {
            "model_loss/model_loss": model_loss.item(),
            "model_loss/obs_loss": observation_loss.mean().item(),
            "model_loss/rew_loss": reward_loss.mean().item(),
            "model_loss/continue_loss": continue_loss.mean().item(),
            "model_loss/kl_loss": kl_loss.mean().item(),

            "actor_loss/actor_loss": actor_loss.item(),
            "actor_loss/reinforce_loss": objective.mean().item(),
            "actor_loss/entropy_loss": entropy.unsqueeze(dim=-1)[:-1].mean().item(),

            "critic_loss/critic_loss": critic_loss.item(),
            "critic_loss/lambda_values": lambda_values.mean().item(),

            "step/gradient_step": self.gradient_step
        }
        return info
