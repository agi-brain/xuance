import torch
from typing import Union
from copy import deepcopy
from xuance.torch import Module, Tensor
from xuance.torch.rl_models.modules import ModelOutput


class ActorCritic(Module):
    def __init__(self,
                 actor: Module,
                 critic: Module,
                 **kwargs):
        super().__init__()
        self.actor = actor
        self.critic = critic

    def forward(self,
                observation: Union[Tensor, dict],
                **kwargs) -> ModelOutput:
        actor_output = self.actor(observation, **kwargs)
        critic_output = self.critic(observation, **kwargs)
        return ModelOutput(distributions=actor_output.distributions,
                           values=critic_output.values,
                           actor_rep_out=actor_output.representations,
                           critic_rep_out=critic_output.representations)

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

    def forward(self,
                observation: Union[Tensor, dict],
                **kwargs) -> ModelOutput:
        rep_out = self.representation(observation, **kwargs)
        pi_distributions = self.actor(rep_out.embeddings, **kwargs)
        values = self.critic(rep_out.embeddings, **kwargs)
        return ModelOutput(distributions=pi_distributions,
                           values=values,
                           rep_out=rep_out)

    def act(self,
            observation: Union[Tensor, dict],
            deterministic: bool = False,
            **kwargs) -> Tensor:
        rep_out = self.representation(observation, **kwargs)
        pi_distributions = self.actor(rep_out.embeddings, **kwargs)

        if deterministic:
            actions = pi_distributions.deterministic_sample()
        else:
            actions = pi_distributions.stochastic_sample()
        return actions


class PhasicActorCritic(ActorCritic):
    def __init__(self,
                 actor: Module,
                 critic: Module,
                 aux_critic: Module,
                 **kwargs):
        super().__init__(actor, critic)
        self.aux_critic = aux_critic

    def forward(self,
                observation: Union[Tensor, dict],
                **kwargs) -> ModelOutput:
        actor_output = self.actor(observation, **kwargs)
        critic_output = self.critic(observation, **kwargs)
        return ModelOutput(distributions=actor_output.distributions,
                           values=critic_output.values,
                           actor_rep_out=actor_output.representations,
                           critic_rep_out=critic_output.representations)

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

    def forward(self,
                observation: Union[Tensor, dict],
                **kwargs) -> ModelOutput:
        actor_output = self.actor(observation, **kwargs)
        critic_output = self.critic(observation, actor_output.actions, **kwargs)
        return ModelOutput(distributions=actor_output.distributions,
                           values=critic_output,
                           actor_rep_out=actor_output.representations)

    def act(self,
            observation: Union[Tensor, dict],
            deterministic: bool = False,
            **kwargs) -> Tensor:
        actor_output = self.actor(observation, **kwargs)

        if deterministic:
            actions = actor_output.distributions.activated_deterministic_sample()
        else:
            actions = actor_output.distributions.activated_rsample()
        return actions

    def Qpolicy(self, observation: Union[Tensor, dict]):
        outputs_actor = self.actor(observation)
        policy_dist = outputs_actor.distributions
        act_sample, log_action_prob = policy_dist.activated_rsample_and_logprob()

        values_1, values_2 = self.Qaction(observation, act_sample)
        return log_action_prob, values_1, values_2

    def Qtarget(self, observation: Union[Tensor, dict]):
        outputs_actor = self.actor(observation)
        policy_dist = outputs_actor.distributions
        act_sample, log_action_prob = policy_dist.activated_rsample_and_logprob()
        outputs_critic = self.target_critic(observation, act_sample)
        target_q = torch.min(outputs_critic.values_1, outputs_critic.values_2)
        return log_action_prob, target_q

    def Qaction(self, observation: Union[Tensor, dict], action: Tensor):
        outputs_critic = self.critic(observation, action)
        return outputs_critic.values_1, outputs_critic.values_2

    def soft_update(self, tau=0.005):
        for ep, tp in zip(self.critic.parameters(), self.target_critic.parameters()):
            tp.data.mul_(1 - tau)
            tp.data.add_(tau * ep.data)


class SoftActorCriticDiscrete(SoftActorCritic):
    def forward(self,
                observation: Union[Tensor, dict],
                **kwargs) -> ModelOutput:
        actor_output = self.actor(observation, **kwargs)
        critic_output = self.critic(observation, **kwargs)
        return ModelOutput(distributions=actor_output.distributions,
                           values=critic_output,
                           actor_rep_out=actor_output.representations)

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

    def Qpolicy(self, observation: Union[Tensor, dict]):
        outputs_actor = self.actor(observation)
        policy_dist = outputs_actor.distributions
        act_prob = policy_dist.probs
        z = act_prob == 0.0
        z = z.float() * 1e-8
        log_action_prob = torch.log(act_prob + z)

        q_1, q_2 = self.Qaction(observation)
        return act_prob, log_action_prob, q_1, q_2

    def Qtarget(self, observation: Union[Tensor, dict]):
        outputs_actor = self.actor(observation)
        policy_dist = outputs_actor.distributions
        act_prob = policy_dist.probs
        z = act_prob == 0.0
        z = z.float() * 1e-8  # avoid log(0)
        log_action_prob = torch.log(act_prob + z)

        outputs_critic = self.target_critic(observation)
        target_q = torch.min(outputs_critic.values_1, outputs_critic.values_2)
        return act_prob, log_action_prob, target_q

    def Qaction(self, observation: Union[Tensor, dict], **kwargs):
        outputs_critic = self.critic(observation)
        return outputs_critic.values_1, outputs_critic.values_2

    def soft_update(self, tau=0.005):
        for ep, tp in zip(self.critic.parameters(), self.target_critic.parameters()):
            tp.data.mul_(1 - tau)
            tp.data.add_(tau * ep.data)


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

    def forward(self,
                observations: Union[Tensor, dict],
                **kwargs) -> ModelOutput:
        actor_output = self.actor(observations, **kwargs)
        critic_output = self.critic(observations, actor_output.actions, **kwargs)
        return ModelOutput(actions=actor_output.actions,
                           values=critic_output.values,
                           actor_rep_out=actor_output.representations,
                           critic_rep_out=critic_output.representations)

    def act(self,
            observation: Union[Tensor, dict],
            **kwargs) -> Tensor:
        return self.actor(observation, **kwargs).actions

    def Qtarget(self, observation: Union[Tensor, dict]):
        outputs_actor = self.target_actor(observation)
        outputs_critic = self.target_critic(observation, outputs_actor.actions)
        return outputs_critic.values

    def Qaction(self, observations: Union[Tensor, dict], actions: Tensor):
        return self.critic(observations, actions).values

    def Qpolicy(self, observation: Union[Tensor, dict]):
        outputs_actor = self.actor(observation)
        return self.critic(observation, outputs_actor.actions).values

    def soft_update(self, tau=0.005):
        for ep, tp in zip(self.actor.parameters(), self.target_actor.parameters()):
            tp.data.mul_(1 - tau)
            tp.data.add_(tau * ep.data)
        for ep, tp in zip(self.critic.parameters(), self.target_critic.parameters()):
            tp.data.mul_(1 - tau)
            tp.data.add_(tau * ep.data)


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

    def forward(self, observation: Union[Tensor, dict]) -> ModelOutput:
        outputs_actor = self.actor(observation)
        return ModelOutput(actions=outputs_actor.actions,
                           actor_rep_out=outputs_actor.representations)

    def act(self,
            observation: Union[Tensor, dict],
            **kwargs) -> Tensor:
        return self.actor(observation, **kwargs).actions

    def Qtarget(self, observation: Union[Tensor, dict]):
        outputs_actor = self.target_actor(observation)
        target_actions = outputs_actor.actions
        target_noise = (torch.randn_like(target_actions) * self.target_policy_noise).clamp(-self.target_noise_clip,
                                                                                           self.target_noise_clip)
        target_actions = target_actions + target_noise
        target_actions = torch.maximum(torch.minimum(target_actions, self.actor.action_high), self.actor.action_low)

        outputs_critic = self.target_critic(observation, target_actions)
        target_q1, target_q2 = outputs_critic.values_1, outputs_critic.values_2
        target_q = torch.minimum(target_q1, target_q2)
        return target_q

    def Qaction(self, observations: Union[Tensor, dict], actions: Tensor):
        outputs_critic = self.critic(observations, actions)
        q_eval_a, q_eval_b = outputs_critic.values_1, outputs_critic.values_2
        return q_eval_a, q_eval_b

    def Qpolicy(self, observation: Union[Tensor, dict]):
        actions = self.actor(observation).actions
        outputs_critic = self.critic(observation, actions)
        q_eval_a, q_eval_b = outputs_critic.values_1, outputs_critic.values_2
        return (q_eval_a + q_eval_b) / 2.0

    def soft_update(self, tau=0.005):
        for ep, tp in zip(self.actor.parameters(), self.target_actor.parameters()):
            tp.data.mul_(1 - tau)
            tp.data.add_(tau * ep.data)
        for ep, tp in zip(self.critic.parameters(), self.target_critic.parameters()):
            tp.data.mul_(1 - tau)
            tp.data.add_(tau * ep.data)
