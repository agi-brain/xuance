import numpy as np
from copy import deepcopy
from typing import Union, Tuple

from xuance.mindspore import ms, ops, msd, Module, Tensor
from xuance.mindspore.rl_models.actors import CategoricalActor, GaussianActor
from xuance.mindspore.rl_models.heads import CategoricalActorHead, GaussianActorHead


class ActorCritic(Module):
    def __init__(self,
                 actor: Module,
                 critic: Module,
                 **kwargs):
        super().__init__()
        self.actor = actor
        self.critic = critic

        if isinstance(self.actor, CategoricalActor):
            self.is_continuous = False
        elif isinstance(self.actor, GaussianActor):
            self.is_continuous = True
        else:
            raise NotImplementedError

    def construct(self,
                  observation: Union[Tensor, dict],
                  **kwargs) -> Tuple[Tensor, ...]:
        values, _ = self.critic(observation, **kwargs)
        if self.is_continuous:
            a_mean, log_std, _ = self.actor(observation, **kwargs)
            return a_mean, log_std, values
        else:
            logits, _ = self.actor(observation, **kwargs)
            return logits, values

    def values(self,
               observation: Union[Tensor, dict],
               **kwargs) -> Tensor:
        values, _ = self.critic(observation, **kwargs)
        return values


class SharedActorCritic(Module):
    def __init__(self,
                 representation: Module,
                 actor: Module,
                 critic: Module,
                 **kwargs):
        super().__init__()
        self.representation = representation
        self.actor = actor
        self.critic = critic

        if isinstance(self.actor, CategoricalActorHead):
            self.is_continuous = False
        elif isinstance(self.actor, GaussianActorHead):
            self.is_continuous = True
        else:
            raise NotImplementedError

    def construct(self,
                  observation: Union[Tensor, dict],
                  **kwargs) -> Tuple[Tensor, ...]:
        rep_embeddings, _ = self.representation(observation, **kwargs)
        if self.is_continuous:
            a_mean, log_std = self.actor(rep_embeddings, **kwargs)
            values = self.critic(rep_embeddings, **kwargs)
            return a_mean, log_std, values
        else:
            logits = self.actor(rep_embeddings, **kwargs)
            values = self.critic(rep_embeddings, **kwargs)
            return logits, values

    def values(self,
               observation: Union[Tensor, dict],
               **kwargs) -> Tensor:
        rep_embeddings, _ = self.representation(observation, **kwargs)
        values = self.critic(rep_embeddings, **kwargs)
        return values


class PhasicActorCritic(ActorCritic):
    def __init__(self,
                 actor: Module,
                 critic: Module,
                 aux_critic: Module,
                 **kwargs):
        super().__init__(actor, critic)
        self.aux_critic = aux_critic

    def construct(self,
                  observation: Union[Tensor, dict],
                  **kwargs) -> Tuple[Tensor, ...]:
        distributions, rnn_states_actor = self.actor(observation, **kwargs)
        values, rnn_states_critic = self.critic(observation, **kwargs)
        return distributions, values, rnn_states_actor, rnn_states_critic

    def act(self,
            observation: Union[Tensor, dict],
            deterministic: bool = False,
            **kwargs) -> Tensor:
        actor_output = self.actor(observation, **kwargs)

        if deterministic:
            actions = actor_output.distributions.deterministic_sample()
        else:
            actions = actor_output.distributions.stochastic_sample()
        return actions


class SoftActorCritic(ActorCritic):
    def __init__(self,
                 actor: Module,
                 critic: Module,
                 **kwargs):
        super().__init__(actor, critic, **kwargs)
        self.target_critic = deepcopy(critic)
        self.activation_action_fn = self.get_activation_action_fn()
        self.pi_dist = self.get_distribution()

    def get_activation_action_fn(self):
        return self.actor.actor_head.policy_distribution.activation_fn

    def get_distribution(self):
        return msd.Normal(dtype=ms.float32)

    def construct(self,
                  observation: Union[Tensor, dict],
                  **kwargs) -> Tuple[Tensor, ...]:
        a_mean, log_std, _ = self.actor(observation, **kwargs)
        a_std = ops.exp(log_std)
        eps = ops.normal(shape=a_mean.shape, mean=Tensor(0.0), stddev=Tensor(1.0))  # 𝜖 ~ N(0, 1)
        action_sampled = a_mean + a_std * eps  # Reparameterization trick
        actions_activated = self.activation_action_fn(action_sampled)
        # calculate log prob
        log_prob = -0.5 * (
                ((action_sampled - a_mean) / (a_std + 1e-8)) ** 2 + 2.0 * log_std + ops.log(Tensor(2.0 * np.pi)))
        correction = - 2. * (ops.log(Tensor(2.0)) - action_sampled - ops.softplus(-2. * action_sampled))
        log_prob += correction
        log_action_prob = ops.reduce_sum(log_prob, axis=-1)
        return actions_activated, log_action_prob

    def Qpolicy(self, observation: Union[Tensor, dict]):
        a_mean, log_std, _ = self.actor(observation)
        a_std = ops.exp(log_std)
        eps = ops.normal(shape=a_mean.shape, mean=Tensor(0.0), stddev=Tensor(1.0))  # 𝜖 ~ N(0, 1)
        action_sampled = a_mean + a_std * eps  # Reparameterization trick
        act_sample = self.activation_action_fn(action_sampled)
        # calculate log prob
        log_prob = -0.5 * (
                ((action_sampled - a_mean) / (a_std + 1e-8)) ** 2 + 2.0 * log_std + ops.log(Tensor(2.0 * np.pi)))
        correction = - 2. * (ops.log(Tensor(2.0)) - action_sampled - ops.softplus(-2. * action_sampled))
        log_prob += correction
        log_action_prob = ops.reduce_sum(log_prob, axis=-1)

        values_1, values_2 = self.Qaction(observation, act_sample)
        return log_action_prob, values_1, values_2

    def Qtarget(self, observation: Union[Tensor, dict]):
        new_act_mean, log_std, _ = self.actor(observation)
        new_act_std = ops.exp(log_std)
        eps = ops.normal(shape=new_act_mean.shape, mean=Tensor(0.0), stddev=Tensor(1.0))  # 𝜖 ~ N(0, 1)
        action_sampled = new_act_mean + new_act_std * eps  # Reparameterization trick
        new_act_sample = self.activation_action_fn(action_sampled)
        # calculate log prob
        log_prob = -0.5 * (((action_sampled - new_act_mean) / (new_act_std + 1e-8)) ** 2 + 2.0 * log_std + ops.log(
            Tensor(2.0 * np.pi)))
        correction = - 2. * (ops.log(Tensor(2.0)) - action_sampled - ops.softplus(-2. * action_sampled))
        log_prob += correction
        log_action_prob = ops.reduce_sum(log_prob, axis=-1)

        values_1, values_2, _, _ = self.target_critic(observation, new_act_sample)
        target_q = ops.minimum(values_1, values_2)
        return log_action_prob, target_q

    def Qaction(self, observation: Union[Tensor, dict], action: Tensor):
        values_1, values_2, _, _ = self.critic(observation, action)
        return values_1, values_2

    def soft_update(self, tau=0.005):
        for ep, tp in zip(self.critic.trainable_params(), self.target_critic.trainable_params()):
            tp.assign_value((tau * ep.data + (1 - tau) * tp.data))


class SoftActorCriticDiscrete(SoftActorCritic):
    def get_activation_action_fn(self):
        return None

    def get_distribution(self):
        return ops.RandomCategorical(ms.int32)

    def construct(self,
                  observation: Union[Tensor, dict],
                  **kwargs) -> Tuple[Tensor, ...]:
        logits, rnn_states_actor = self.actor(observation, **kwargs)
        actions = self.pi_dist(logits, 1, 0).squeeze(-1)
        return actions, rnn_states_actor

    def Qpolicy(self, observation: Union[Tensor, dict]):
        logits, _ = self.actor(observation)
        act_prob = ops.softmax(logits, axis=-1)
        z = act_prob == 0.0
        z = z.float() * 1e-8
        log_action_prob = ops.log(act_prob + z)

        q_1, q_2 = self.Qaction(observation)
        return act_prob, log_action_prob, q_1, q_2

    def Qtarget(self, observation: Union[Tensor, dict]):
        logits, _ = self.actor(observation)
        act_prob = ops.softmax(logits, axis=-1)
        z = act_prob == 0.0
        z = z.float() * 1e-8  # avoid log(0)
        log_action_prob = ops.log(act_prob + z)

        values_1, values_2, _, _ = self.target_critic(observation)
        target_q = ops.minimum(values_1, values_2)
        return act_prob, log_action_prob, target_q

    def Qaction(self, observation: Union[Tensor, dict], **kwargs):
        values_1, values_2, _, _ = self.critic(observation)
        return values_1, values_2

    def soft_update(self, tau=0.005):
        for ep, tp in zip(self.critic.trainable_params(), self.target_critic.trainable_params()):
            tp.assign_value((tau * ep.data + (1 - tau) * tp.data))


class DeterministicActorCritic(Module):
    def __init__(self,
                 actor: Module,
                 critic: Module,
                 **kwargs):
        super().__init__(**kwargs)
        self.actor = actor
        self.critic = critic
        self.target_actor = deepcopy(actor)
        self.target_critic = deepcopy(critic)

    def construct(self,
                  observations: Union[Tensor, dict],
                  **kwargs) -> Tuple[Tensor, ...]:
        actions, rnn_states_actor = self.actor(observations, **kwargs)
        values, rnn_states_critic = self.critic(observations, actions, **kwargs)
        return actions, values, rnn_states_actor, rnn_states_critic

    def act(self,
            observation: Union[Tensor, dict],
            **kwargs) -> Tensor:
        actions, _ = self.actor(observation, **kwargs)
        return actions

    def Qtarget(self, observation: Union[Tensor, dict]):
        actions, _ = self.target_actor(observation)
        values, _ = self.target_critic(observation, actions)
        return values

    def Qaction(self, observations: Union[Tensor, dict], actions: Tensor):
        values, _ = self.critic(observations, actions)
        return values

    def Qpolicy(self, observation: Union[Tensor, dict]):
        actions, _ = self.actor(observation)
        values, _ = self.critic(observation, actions)
        return values

    def soft_update(self, tau=0.005):
        for ep, tp in zip(self.actor.trainable_params(), self.target_actor.trainable_params()):
            tp.assign_value((tau * ep.data + (1 - tau) * tp.data))
        for ep, tp in zip(self.critic.trainable_params(), self.target_critic.trainable_params()):
            tp.assign_value((tau * ep.data + (1 - tau) * tp.data))


class TwinDelayedActorCritic(Module):
    def __init__(self,
                 actor: Module,
                 critic: Module,
                 target_policy_noise=0.2,
                 target_noise_clip=0.5,
                 **kwargs):
        super().__init__(**kwargs)
        self.actor = actor
        self.critic = critic
        self.target_actor = deepcopy(self.actor)
        self.target_critic = deepcopy(self.critic)
        self.target_policy_noise = target_policy_noise
        self.target_noise_clip = target_noise_clip

    def construct(self, observation: Union[Tensor, dict]) -> Tensor:
        actions, _ = self.actor(observation)
        return actions

    def act(self, observation: Union[Tensor, dict]) -> Tensor:
        return self.construct(observation)

    def Qtarget(self, observation: Union[Tensor, dict]):
        target_actions, _ = self.target_actor(observation)
        target_noise = (ops.randn_like(target_actions) * self.target_policy_noise).clamp(-self.target_noise_clip,
                                                                                         self.target_noise_clip)
        target_actions = target_actions + target_noise
        target_actions = ops.maximum(ops.minimum(target_actions, self.actor.action_high), self.actor.action_low)

        target_q1, target_q2, _, _ = self.target_critic(observation, target_actions)
        target_q = ops.minimum(target_q1, target_q2)
        return target_q

    def Qaction(self, observations: Union[Tensor, dict], actions: Tensor):
        q_eval_a, q_eval_b, _, _ = self.critic(observations, actions)
        return q_eval_a, q_eval_b

    def Qpolicy(self, observation: Union[Tensor, dict]):
        actions, _ = self.actor(observation)
        q_eval_a, q_eval_b, _, _ = self.critic(observation, actions)
        return (q_eval_a + q_eval_b) / 2.0

    def soft_update(self, tau=0.005):
        for ep, tp in zip(self.actor.trainable_params(), self.target_actor.trainable_params()):
            tp.assign_value((tau * ep.data + (1 - tau) * tp.data))
        for ep, tp in zip(self.critic.trainable_params(), self.target_critic.trainable_params()):
            tp.assign_value((tau * ep.data + (1 - tau) * tp.data))
