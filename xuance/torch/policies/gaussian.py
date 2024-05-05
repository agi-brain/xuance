import torch.nn.functional
from xuance.torch.policies import *
from xuance.torch.utils import *
import numpy as np


class ActorNet(nn.Module):
    def __init__(self,
                 state_dim: int,
                 action_dim: int,
                 hidden_sizes: Sequence[int],
                 normalize: Optional[ModuleType] = None,
                 initialize: Optional[Callable[..., torch.Tensor]] = None,
                 activation: Optional[ModuleType] = None,
                 activation_action: Optional[ModuleType] = None,
                 device: Optional[Union[str, int, torch.device]] = None):
        super(ActorNet, self).__init__()
        layers = []
        input_shape = (state_dim,)
        for h in hidden_sizes:
            mlp, input_shape = mlp_block(input_shape[0], h, normalize, activation, initialize, device)
            layers.extend(mlp)
        layers.extend(mlp_block(input_shape[0], action_dim, None, activation_action, initialize, device)[0])
        self.mu = nn.Sequential(*layers)
        self.logstd = nn.Parameter(-torch.ones((action_dim,), device=device))
        self.dist = DiagGaussianDistribution(action_dim)

    def forward(self, x: torch.Tensor):
        self.dist.set_param(self.mu(x), self.logstd.exp())
        return self.dist


class CriticNet(nn.Module):
    def __init__(self,
                 state_dim: int,
                 hidden_sizes: Sequence[int],
                 normalize: Optional[ModuleType] = None,
                 initialize: Optional[Callable[..., torch.Tensor]] = None,
                 activation: Optional[ModuleType] = None,
                 device: Optional[Union[str, int, torch.device]] = None):
        super(CriticNet, self).__init__()
        layers = []
        input_shape = (state_dim,)
        for h in hidden_sizes:
            mlp, input_shape = mlp_block(input_shape[0], h, normalize, activation, initialize, device)
            layers.extend(mlp)
        layers.extend(mlp_block(input_shape[0], 1, None, None, None, device)[0])
        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor):
        return self.model(x)[:, 0]


class ActorCriticPolicy(nn.Module):
    def __init__(self,
                 action_space: Space,
                 representation: nn.Module,
                 actor_hidden_size: Sequence[int] = None,
                 critic_hidden_size: Sequence[int] = None,
                 normalize: Optional[ModuleType] = None,
                 initialize: Optional[Callable[..., torch.Tensor]] = None,
                 activation: Optional[ModuleType] = None,
                 activation_action: Optional[ModuleType] = None,
                 device: Optional[Union[str, int, torch.device]] = None):
        super(ActorCriticPolicy, self).__init__()
        self.action_dim = action_space.shape[0]
        self.representation = representation
        self.representation_info_shape = representation.output_shapes
        self.actor = ActorNet(representation.output_shapes['state'][0], self.action_dim, actor_hidden_size,
                              normalize, initialize, activation, activation_action, device)
        self.critic = CriticNet(representation.output_shapes['state'][0], critic_hidden_size,
                                normalize, initialize, activation, device)

    def forward(self, observation: Union[np.ndarray, dict]):
        outputs = self.representation(observation)
        a = self.actor(outputs['state'])
        v = self.critic(outputs['state'])
        return outputs, a, v


class ActorPolicy(nn.Module):
    def __init__(self,
                 action_space: Space,
                 representation: nn.Module,
                 actor_hidden_size: Sequence[int] = None,
                 normalize: Optional[ModuleType] = None,
                 initialize: Optional[Callable[..., torch.Tensor]] = None,
                 activation: Optional[ModuleType] = None,
                 activation_action: Optional[ModuleType] = None,
                 device: Optional[Union[str, int, torch.device]] = None,
                 fixed_std: bool = True):
        super(ActorPolicy, self).__init__()
        self.action_dim = action_space.shape[0]
        self.representation = representation
        self.representation_info_shape = self.representation.output_shapes
        self.actor = ActorNet(representation.output_shapes['state'][0], self.action_dim, actor_hidden_size,
                              normalize, initialize, activation, activation_action, device)

    def forward(self, observation: Union[np.ndarray, dict]):
        outputs = self.representation(observation)
        a = self.actor(outputs['state'])
        return outputs, a


class PPGActorCritic(nn.Module):
    def __init__(self,
                 action_space: Space,
                 representation: nn.Module,
                 actor_hidden_size: Sequence[int] = None,
                 critic_hidden_size: Sequence[int] = None,
                 normalize: Optional[ModuleType] = None,
                 initialize: Optional[Callable[..., torch.Tensor]] = None,
                 activation: Optional[ModuleType] = None,
                 activation_action: Optional[ModuleType] = None,
                 device: Optional[Union[str, int, torch.device]] = None):
        super(PPGActorCritic, self).__init__()
        self.action_dim = action_space.shape[0]
        self.actor_representation = representation
        self.critic_representation = copy.deepcopy(representation)
        self.representation_info_shape = self.actor_representation.output_shapes
        self.actor = ActorNet(representation.output_shapes['state'][0], self.action_dim, actor_hidden_size,
                              normalize, initialize, activation, activation_action, device)
        self.critic = CriticNet(representation.output_shapes['state'][0], critic_hidden_size,
                                normalize, initialize, activation, device)
        self.aux_critic = CriticNet(representation.output_shapes['state'][0], critic_hidden_size,
                                    normalize, initialize, activation, device)

    def forward(self, observation: Union[np.ndarray, dict]):
        policy_outputs = self.actor_representation(observation)
        critic_outputs = self.critic_representation(observation)
        a = self.actor(policy_outputs['state'])
        v = self.critic(critic_outputs['state'])
        aux_v = self.aux_critic(policy_outputs['state'])
        return policy_outputs, a, v, aux_v


class ActorNet_SAC(nn.Module):
    def __init__(self,
                 state_dim: int,
                 action_dim: int,
                 hidden_sizes: Sequence[int],
                 normalize: Optional[ModuleType] = None,
                 initialize: Optional[Callable[..., torch.Tensor]] = None,
                 activation: Optional[ModuleType] = None,
                 activation_action: Optional[ModuleType] = None,
                 device: Optional[Union[str, int, torch.device]] = None):
        super(ActorNet_SAC, self).__init__()
        layers = []
        input_shape = (state_dim,)
        for h in hidden_sizes:
            mlp, input_shape = mlp_block(input_shape[0], h, normalize, activation, initialize, device)
            layers.extend(mlp)
        self.device = device
        self.output = nn.Sequential(*layers)
        self.out_mu = nn.Linear(hidden_sizes[-1], action_dim, device=device)
        self.out_log_std = nn.Linear(hidden_sizes[-1], action_dim, device=device)
        self.dist = ActivatedDiagGaussianDistribution(action_dim, activation_action, device)

    def forward(self, x: torch.tensor):
        output = self.output(x)
        mu = self.out_mu(output)
        log_std = torch.clamp(self.out_log_std(output), -20, 2)
        std = log_std.exp()
        self.dist.set_param(mu, std)
        return self.dist


class CriticNet_SAC(nn.Module):
    def __init__(self,
                 state_dim: int,
                 action_dim: int,
                 hidden_sizes: Sequence[int],
                 normalize: Optional[ModuleType] = None,
                 initialize: Optional[Callable[..., torch.Tensor]] = None,
                 activation: Optional[ModuleType] = None,
                 device: Optional[Union[str, int, torch.device]] = None):
        super(CriticNet_SAC, self).__init__()
        layers = []
        input_shape = (state_dim + action_dim,)
        for h in hidden_sizes:
            mlp, input_shape = mlp_block(input_shape[0], h, normalize, activation, initialize, device)
            layers.extend(mlp)
        layers.extend(mlp_block(input_shape[0], 1, None, None, initialize, device)[0])
        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.tensor, a: torch.tensor):
        return self.model(torch.concat((x, a), dim=-1))[:, 0]


class SACPolicy(nn.Module):
    def __init__(self,
                 action_space: Space,
                 representation: nn.Module,
                 actor_hidden_size: Sequence[int],
                 critic_hidden_size: Sequence[int],
                 normalize: Optional[ModuleType] = None,
                 initialize: Optional[Callable[..., torch.Tensor]] = None,
                 activation: Optional[ModuleType] = None,
                 activation_action: Optional[ModuleType] = None,
                 device: Optional[Union[str, int, torch.device]] = None):
        super(SACPolicy, self).__init__()
        self.action_dim = action_space.shape[0]
        self.activation_action = activation_action()
        self.representation_info_shape = representation.output_shapes

        # representations
        self.actor_representation = representation
        self.actor = ActorNet_SAC(representation.output_shapes['state'][0], self.action_dim, actor_hidden_size,
                                  normalize, initialize, activation, activation_action, device)

        self.critic_1_representation = copy.deepcopy(representation)
        self.critic_1 = CriticNet_SAC(representation.output_shapes['state'][0], self.action_dim, critic_hidden_size,
                                      normalize, initialize, activation, device)
        self.critic_2_representation = copy.deepcopy(representation)
        self.critic_2 = CriticNet_SAC(representation.output_shapes['state'][0], self.action_dim, critic_hidden_size,
                                      normalize, initialize, activation, device)
        self.target_critic_1_representation = copy.deepcopy(self.critic_1_representation)
        self.target_critic_1 = copy.deepcopy(self.critic_1)
        self.target_critic_2_representation = copy.deepcopy(self.critic_2_representation)
        self.target_critic_2 = copy.deepcopy(self.critic_2)

        self.actor_parameters = list(self.actor_representation.parameters()) + list(self.actor.parameters())
        self.critic_parameters = list(self.critic_1_representation.parameters()) + list(
            self.critic_1.parameters()) + list(self.critic_2_representation.parameters()) + list(
            self.critic_2.parameters())

    def forward(self, observation: Union[np.ndarray, dict]):
        outputs_actor = self.actor_representation(observation)
        act_dist = self.actor(outputs_actor['state'])
        act_output = act_dist.activated_rsample()
        return outputs_actor, act_output

    def Qpolicy(self, observation: Union[np.ndarray, dict]):
        outputs_actor = self.actor_representation(observation)
        outputs_critic_1 = self.critic_1_representation(observation)
        outputs_critic_2 = self.critic_2_representation(observation)

        act_dist = self.actor(outputs_actor['state'])
        # act_sample = act_dist.rsample()
        # act_output = self.activation_action(act_sample)
        # act_log = act_dist.log_prob(act_sample).sum(-1)
        act_output, act_log = act_dist.activated_rsample_and_logprob()

        q_1 = self.critic_1(outputs_critic_1['state'], act_output)
        q_2 = self.critic_2(outputs_critic_2['state'], act_output)
        return act_log, q_1, q_2

    def Qtarget(self, observation: Union[np.ndarray, dict]):
        outputs_actor = self.actor_representation(observation)
        outputs_critic_1 = self.target_critic_1_representation(observation)
        outputs_critic_2 = self.target_critic_2_representation(observation)

        new_act_dist = self.actor(outputs_actor['state'])
        # new_act_sample = new_act_dist.rsample()
        # new_act_output = self.activation_action(new_act_sample)
        # new_act_log = new_act_dist.log_prob(new_act_sample).sum(-1)
        new_act_output, new_act_log = new_act_dist.activated_rsample_and_logprob()

        target_q_1 = self.target_critic_1(outputs_critic_1['state'], new_act_output)
        target_q_2 = self.target_critic_2(outputs_critic_2['state'], new_act_output)
        target_q = torch.min(target_q_1, target_q_2)
        return new_act_log, target_q

    def Qaction(self, observation: Union[np.ndarray, dict], action: torch.Tensor):
        outputs_critic_1 = self.critic_1_representation(observation)
        outputs_critic_2 = self.critic_2_representation(observation)
        q_1 = self.critic_1(outputs_critic_1['state'], action)
        q_2 = self.critic_2(outputs_critic_2['state'], action)
        return q_1, q_2

    def soft_update(self, tau=0.005):
        for ep, tp in zip(self.critic_1_representation.parameters(), self.target_critic_1_representation.parameters()):
            tp.data.mul_(1 - tau)
            tp.data.add_(tau * ep.data)
        for ep, tp in zip(self.critic_2_representation.parameters(), self.target_critic_2_representation.parameters()):
            tp.data.mul_(1 - tau)
            tp.data.add_(tau * ep.data)
        for ep, tp in zip(self.critic_1.parameters(), self.target_critic_1.parameters()):
            tp.data.mul_(1 - tau)
            tp.data.add_(tau * ep.data)
        for ep, tp in zip(self.critic_2.parameters(), self.target_critic_2.parameters()):
            tp.data.mul_(1 - tau)
            tp.data.add_(tau * ep.data)
