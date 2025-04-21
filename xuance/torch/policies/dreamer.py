import torch
from torch.distributions import Independent, Normal, Bernoulli
from torch.distributions.utils import logits_to_probs
import numpy as np
from argparse import Namespace
from xuance.common import Any, Dict, Tuple, Sequence, List
from xuance.torch import Tensor, Module
from xuance.torch.utils.distributions import MSEDistribution, SymLogDistribution, TwoHotEncodingDistribution, BernoulliSafeMode
from xuance.torch.utils import dotdict, Moments, compute_lambda_values
from xuance.torch.utils.harmonizer import Harmonizer


class DreamerV3Policy(Module):  # checked
    def __init__(self,
                 model: Module,
                 config: Namespace):
        super(DreamerV3Policy, self).__init__()
        # convert to dotdict
        self.config = dotdict(vars(config))
        self.stoch_size = self.config.world_model.stochastic_size
        self.disc_size = self.config.world_model.discrete_size
        self.stoch_state_size = self.stoch_size * self.disc_size  # 1024 = 32 * 32
        self.batch_size = self.config.batch_size
        self.seq_len = self.config.seq_len
        self.recurrent_state_size = self.config.world_model.recurrent_model.recurrent_state_size
        self.device = self.config.device
        self.is_continuous = self.config.is_continuous
        self.actions_dim = np.sum(self.config.act_shape)  # continuous: num of action props; discrete: num of actions

        # nets
        self.model: Module = model
        self.world_model: Module = self.model.world_model
        self.actor: Module = self.model.actor
        self.critic: Module = self.model.critic
        self.target_critic: Module = self.model.target_critic

        # running mean
        self.moments = Moments(
            self.config.actor.moments.decay,
            self.config.actor.moments.max,
            self.config.actor.moments.percentile.low,
            self.config.actor.moments.percentile.high,
        )

        self.harmonizer_s1 = Harmonizer(self.device)
        self.harmonizer_s2 = Harmonizer(self.device)
        self.harmonizer_s3 = Harmonizer(self.device)

    def model_forward(self,
                      obs: Tensor,
                      acts: Tensor,
                      is_first: Tensor) \
            -> Tuple[SymLogDistribution, TwoHotEncodingDistribution, Independent, Tensor, Tensor,
                     Tensor, Tensor]:
        recurrent_state = torch.zeros(1, self.batch_size, self.recurrent_state_size, device=self.device)  # [1, 16, 512]
        recurrent_states = torch.empty(self.seq_len, self.batch_size, self.recurrent_state_size,
                                       device=self.device)  # [64, 16, 512]
        priors_logits = torch.empty(self.seq_len, self.batch_size, self.stoch_state_size, device=self.device)  # [64, 16, 1024]
        embedded_obs = self.world_model.encoder(obs)  # [64, 16, 512]

        # [1, 16, 32, 32], [64, 16, 32, 32], [64, 16, 1024]
        posterior = torch.zeros(1, self.batch_size, self.stoch_size, self.disc_size, device=self.device)
        posteriors = torch.empty(self.seq_len, self.batch_size, self.stoch_size, self.disc_size, device=self.device)
        posteriors_logits = torch.empty(self.seq_len, self.batch_size, self.stoch_state_size, device=self.device)
        for i in range(0, self.seq_len):
            recurrent_state, posterior, _, posterior_logits, prior_logits = self.world_model.rssm.dynamic(
                posterior,  # z0  [1, 16, 32, 32]
                recurrent_state,  # h0  [1, 16, 512]
                acts[i: i + 1],  # a0  [1, 16, 2]
                embedded_obs[i: i + 1],  # x1  [1, 16, 512]
                is_first[i: i + 1],  # is_first1  [1, 16, 1]
            )  # h0, cat(z0, a0) -> h1; h1 + x1 -> z1; h1 -> z1_hat
            recurrent_states[i] = recurrent_state
            priors_logits[i] = prior_logits  # z1_hat
            posteriors[i] = posterior
            posteriors_logits[i] = posterior_logits  # z1
        latent_states = torch.cat((posteriors.view(*posteriors.shape[:-2], -1), recurrent_states), -1)
        """model_states: [64, 16, 32 * 32 + 512 = 1536]"""
        reconstructed_obs: Tensor = self.world_model.observation_model(latent_states)
        """po(obs, symlog_dist)"""
        po = SymLogDistribution(reconstructed_obs, dims=len(reconstructed_obs.shape[2:]))
        """pr(rews, two_hot_dist)"""
        pr = TwoHotEncodingDistribution(self.world_model.reward_model(latent_states), dims=1)
        """pc(cont, bernoulli_dist)"""
        pc = Independent(BernoulliSafeMode(logits=self.world_model.continue_model(latent_states)), 1)

        # -> [seq, batch, 32, 32]
        priors_logits = priors_logits.view(*priors_logits.shape[:-1], self.stoch_size, self.disc_size)
        posteriors_logits = posteriors_logits.view(*posteriors_logits.shape[:-1], self.stoch_size, self.disc_size)

        return (po, pr, pc, priors_logits, posteriors_logits,
                recurrent_states, posteriors)

    def actor_critic_forward(self,
                             posteriors: Tensor,
                             recurrent_states: Tensor,
                             terms: Tensor) \
            -> Dict[str, List[Any]]:
        imagined_prior = posteriors.detach().reshape(1, -1, self.stoch_state_size)
        recurrent_state = recurrent_states.detach().reshape(1, -1, self.recurrent_state_size)  # [1, 1024, 512]
        imagined_latent_state = torch.cat((imagined_prior, recurrent_state), -1)  # [1, 1024, 1536]
        imagined_trajectories = torch.empty(
            self.config.horizon + 1,
            self.batch_size * self.seq_len,
            self.stoch_state_size + self.recurrent_state_size,
            device=self.device,
        )  # [16, 1024, 1536]
        imagined_trajectories[0] = imagined_latent_state
        imagined_actions = torch.empty(
            self.config.horizon + 1,
            self.batch_size * self.seq_len,
            self.actions_dim,
            device=self.device,
        )  # [16, 1024, 2]
        actions = torch.cat(self.actor(imagined_latent_state.detach())[0], dim=-1)  # z0 -> a0
        imagined_actions[0] = actions

        for i in range(1, self.config.horizon + 1):
            imagined_prior, recurrent_state = self.world_model.rssm.imagination(imagined_prior, recurrent_state, actions)
            imagined_prior = imagined_prior.view(1, -1, self.stoch_state_size)  # [1, 1024, 1024]
            imagined_latent_state = torch.cat((imagined_prior, recurrent_state), -1)
            imagined_trajectories[i] = imagined_latent_state
            actions = torch.cat(self.actor(imagined_latent_state.detach())[0], dim=-1)
            imagined_actions[i] = actions
        predicted_values = TwoHotEncodingDistribution(self.critic(imagined_trajectories), dims=1).mean
        predicted_rewards = TwoHotEncodingDistribution(self.world_model.reward_model(imagined_trajectories), dims=1).mean
        continues = Independent(BernoulliSafeMode(logits=self.world_model.continue_model(imagined_trajectories)), 1).mode
        true_continue = (1 - terms).flatten().reshape(1, -1, 1)  # continues: [16, 1024, 1]; true: [1, 1024, 1]
        continues = torch.cat((true_continue, continues[1:]))
        """seq_shift[1:]"""
        lambda_values = compute_lambda_values(
            predicted_rewards[1:],
            predicted_values[1:],
            continues[1:] * self.config.gamma,
            lmbda=self.config.lmbda,
        )

        with torch.no_grad():
            discount = torch.cumprod(continues * self.config.gamma, dim=0) / self.config.gamma

        policies: Sequence[torch.distributions.Distribution] = self.actor(imagined_trajectories.detach())[1]

        baseline = predicted_values[:-1]
        offset, invscale = self.moments(lambda_values)
        normed_lambda_values = (lambda_values - offset) / invscale
        normed_baseline = (baseline - offset) / invscale
        advantage = normed_lambda_values - normed_baseline
        if self.is_continuous:
            objective = advantage
        else:
            objective = (
                torch.stack(
                    [
                        p.log_prob(imgnd_act.detach()).unsqueeze(-1)[:-1]
                        for p, imgnd_act in zip(policies, torch.split(imagined_actions, [self.actions_dim], dim=-1))
                    ],
                    dim=-1,
                ).sum(dim=-1)
                * advantage.detach()
            )
        try:
            entropy = self.config.actor.ent_coef * torch.stack([p.entropy() for p in policies], -1).sum(dim=-1)
        except NotImplementedError:
            entropy = torch.zeros_like(objective)

        """seq_shift"""
        qv = TwoHotEncodingDistribution(self.critic(imagined_trajectories.detach()[:-1]), dims=1)
        predicted_target_values = TwoHotEncodingDistribution(
            self.target_critic(imagined_trajectories.detach()[:-1]), dims=1
        ).mean
        return {
            'for_actor': [objective, discount, entropy],
            'for_critic': [qv, predicted_target_values, lambda_values]
        }

    def soft_update(self, tau=0.02):  # checked
        for ep, tp in zip(self.critic.parameters(), self.target_critic.parameters()):
            tp.data.mul_(1 - tau)
            tp.data.add_(tau * ep.data)



class DreamerV2Policy(Module):  # checked
    def __init__(self,
                 model: Module,
                 config: Namespace):
        super(DreamerV2Policy, self).__init__()
        # convert to dotdict
        self.config = dotdict(vars(config))
        self.stoch_size = self.config.world_model.stochastic_size
        self.disc_size = self.config.world_model.discrete_size
        self.stoch_state_size = self.stoch_size * self.disc_size  # 1024 = 32 * 32
        self.batch_size = self.config.batch_size
        self.seq_len = self.config.seq_len
        self.recurrent_state_size = self.config.world_model.recurrent_model.recurrent_state_size
        self.device = self.config.device
        self.is_continuous = self.config.is_continuous
        self.actions_dim = np.sum(self.config.act_shape)  # continuous: num of action props; discrete: num of actions

        # nets
        self.model: Module = model
        self.world_model: Module = self.model.world_model
        self.actor: Module = self.model.actor
        self.critic: Module = self.model.critic
        self.target_critic: Module = self.model.target_critic

        self.harmonizer_s1 = Harmonizer(self.device)
        self.harmonizer_s2 = Harmonizer(self.device)
        self.harmonizer_s3 = Harmonizer(self.device)

    def model_forward(self,
                      obs: Tensor,
                      acts: Tensor,
                      is_first: Tensor) \
            -> Tuple[Independent, Independent, Independent, Tensor, Tensor,
                     Tensor, Tensor]:
        recurrent_state = torch.zeros(1, self.batch_size, self.recurrent_state_size, device=self.device)
        recurrent_states = torch.zeros(self.seq_len, self.batch_size, self.recurrent_state_size,
                                       device=self.device)
        priors_logits = torch.empty(self.seq_len, self.batch_size, self.stoch_state_size, device=self.device)
        embedded_obs = self.world_model.encoder(obs)

        posterior = torch.zeros(1, self.batch_size, self.stoch_size, self.disc_size, device=self.device)
        posteriors = torch.empty(self.seq_len, self.batch_size, self.stoch_size, self.disc_size, device=self.device)
        posteriors_logits = torch.empty(self.seq_len, self.batch_size, self.stoch_state_size, device=self.device)
        for i in range(0, self.seq_len):
            recurrent_state, posterior, _, posterior_logits, prior_logits = self.world_model.rssm.dynamic(
                posterior,
                recurrent_state,
                acts[i: i + 1],
                embedded_obs[i: i + 1],
                is_first[i: i + 1],
            )
            recurrent_states[i] = recurrent_state
            priors_logits[i] = prior_logits
            posteriors[i] = posterior
            posteriors_logits[i] = posterior_logits
        latent_states = torch.cat((posteriors.view(*posteriors.shape[:-2], -1), recurrent_states), -1)
        reconstructed_obs: Tensor = self.world_model.observation_model(latent_states)

        po = Independent(Normal(reconstructed_obs, 1), len(reconstructed_obs.shape[2:]))
        pr = Independent(Normal(self.world_model.reward_model(latent_states), 1), 1)
        # error due to not support of Boolean()
        if self.config.world_model.use_continues:
            pc = Independent(Bernoulli(logits=self.world_model.continue_model(latent_states)), 1)
        else:
            pc = None

        # -> [seq, batch, 32, 32]
        priors_logits = priors_logits.view(*priors_logits.shape[:-1], self.stoch_size, self.disc_size)
        posteriors_logits = posteriors_logits.view(*posteriors_logits.shape[:-1], self.stoch_size, self.disc_size)

        return (po, pr, pc, priors_logits, posteriors_logits,
                recurrent_states, posteriors)


    def actor_critic_forward(self,
                             posteriors: Tensor,
                             recurrent_states: Tensor,
                             terms: Tensor) \
            -> Dict[str, List[Any]]:
        imagined_prior = posteriors.detach().reshape(1, -1, self.stoch_state_size)
        recurrent_state = recurrent_states.detach().reshape(1, -1, self.recurrent_state_size)
        imagined_latent_state = torch.cat((imagined_prior, recurrent_state), -1)
        imagined_trajectories = torch.empty(
            self.config.horizon + 1,
            self.batch_size * self.seq_len,
            self.stoch_state_size + self.recurrent_state_size,
            device=self.device,
        )
        imagined_trajectories[0] = imagined_latent_state
        imagined_actions = torch.empty(
            self.config.horizon + 1,
            self.batch_size * self.seq_len,
            self.actions_dim,
            device=self.device,
        )
        # diff to v3; here is, at imagined_trajectories[0] takes action imagined_actions[1]
        imagined_actions[0] = torch.zeros(1, self.batch_size * self.seq_len, self.actions_dim)
        for i in range(1, self.config.horizon + 1):
            # (1, batch_size * seq_len, sum(actions_dim))
            actions = torch.cat(self.actor(imagined_latent_state.detach())[0], dim=-1)
            imagined_actions[i] = actions
            imagined_prior, recurrent_state = self.world_model.rssm.imagination(imagined_prior, recurrent_state, actions)
            imagined_prior = imagined_prior.view(1, -1, self.stoch_state_size)
            imagined_latent_state = torch.cat((imagined_prior, recurrent_state), -1)
            imagined_trajectories[i] = imagined_latent_state
        predicted_target_values = self.target_critic(imagined_trajectories)
        predicted_rewards = self.world_model.reward_model(imagined_trajectories)
        if self.config.world_model.use_continues:
            continues = logits_to_probs(self.world_model.continue_model(imagined_trajectories), is_binary=True)  # diff to v3
            # diff to v3(v3: no self.config.gamma here, but mult gamma before passing to 'compute_lambda_values')
            true_continue = (1 - terms).reshape(1, -1, 1) * self.config.gamma
            continues = torch.cat((true_continue, continues[1:]))
        else:
            continues = torch.ones_like(predicted_rewards.detach()) * self.config.gamma

        # Compute the lambda_values, by passing as last value the value of the last imagined state
        # (horizon, batch_size * seq_len, 1)
        lambda_values = DreamerV2Policy.compute_lambda_values(
            predicted_rewards[:-1],
            predicted_target_values[:-1],
            continues[:-1],
            bootstrap=predicted_target_values[-1:],
            horizon=self.config.horizon,
            lmbda=self.config.lmbda,
        )

        with torch.no_grad():
            discount = torch.cumprod(torch.cat((torch.ones_like(continues[:1]), continues[:-1]), 0), 0)

        policies: Sequence[torch.distributions.Distribution] = self.actor(imagined_trajectories[:-2].detach())[1]
        # Dynamics backpropagation
        dynamics = lambda_values[1:]
        # Reinforce
        advantage = (lambda_values[1:] - predicted_target_values[:-2]).detach()
        reinforce = (
                torch.stack(
                    [
                        p.log_prob(imgnd_act[1:-1].detach()).unsqueeze(-1)
                        for p, imgnd_act in zip(policies, torch.split(imagined_actions, [self.actions_dim], -1))
                    ],
                    -1,
                ).sum(-1)
                * advantage
        )
        objective = self.config.actor.objective_mix * reinforce + (1 - self.config.actor.objective_mix) * dynamics
        try:
            entropy = self.config.actor.ent_coef * torch.stack([p.entropy() for p in policies], -1).sum(dim=-1)
        except NotImplementedError:
            entropy = torch.zeros_like(objective)
        # policy_loss = -torch.mean(discount[:-2].detach() * (objective + entropy.unsqueeze(-1)))

        # last imagined state (with position=horizon+1) in the trajectory only used for bootstrapping;
        qv = Independent(Normal(self.critic(imagined_trajectories.detach()[:-1]), 1), 1)

        return {
            'for_actor': [objective, discount, entropy],
            'for_critic': [qv, predicted_target_values, lambda_values]
        }

    def hard_update(self):  # checked
        for ep, tp in zip(self.critic.parameters(), self.target_critic.parameters()):
            tp.data.mul_(0)
            tp.data.add_(1.0 * ep.data)

    @staticmethod
    def compute_lambda_values(
            rewards: Tensor,
            values: Tensor,
            continues: Tensor,
            bootstrap: Tensor = None,
            horizon: int = 15,
            lmbda: float = 0.95,
    ) -> Tensor:
        if bootstrap is None:
            bootstrap = torch.zeros_like(values[-1:])
        agg = bootstrap
        next_val = torch.cat((values[1:], bootstrap), dim=0)
        inputs = rewards + continues * next_val * (1 - lmbda)
        lv = []
        for i in reversed(range(horizon)):
            agg = inputs[i] + continues[i] * lmbda * agg
            lv.append(agg)
        return torch.cat(list(reversed(lv)), dim=0)