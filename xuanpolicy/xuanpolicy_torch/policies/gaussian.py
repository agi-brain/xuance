import torch

from xuanpolicy.xuanpolicy_torch.policies import *
from xuanpolicy.xuanpolicy_torch.utils import *
from xuanpolicy.xuanpolicy_torch.representations import Basic_Identical


class ActorNet(nn.Module):
    def __init__(self,
                 state_dim: int,
                 action_dim: int,
                 hidden_sizes: Sequence[int],
                 normalize: Optional[ModuleType] = None,
                 initialize: Optional[Callable[..., torch.Tensor]] = None,
                 activation: Optional[ModuleType] = None,
                 device: Optional[Union[str, int, torch.device]] = None):
        super(ActorNet, self).__init__()
        layers = []
        input_shape = (state_dim,)
        for h in hidden_sizes:
            mlp, input_shape = mlp_block(input_shape[0], h, normalize, activation, initialize, device)
            layers.extend(mlp)
        layers.extend(mlp_block(input_shape[0], action_dim, None, None, initialize, device)[0])
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
                 representation: ModuleType,
                 actor_hidden_size: Sequence[int] = None,
                 critic_hidden_size: Sequence[int] = None,
                 normalize: Optional[ModuleType] = None,
                 initialize: Optional[Callable[..., torch.Tensor]] = None,
                 activation: Optional[ModuleType] = None,
                 device: Optional[Union[str, int, torch.device]] = None):
        assert isinstance(action_space, Box)
        super(ActorCriticPolicy, self).__init__()
        self.action_dim = action_space.shape[0]
        self.representation = representation
        self.representation_info_shape = self.representation.output_shapes
        self.actor = ActorNet(representation.output_shapes['state'][0], self.action_dim, actor_hidden_size,
                              normalize, initialize, activation, device)
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
                 representation: ModuleType,
                 actor_hidden_size: Sequence[int] = None,
                 normalize: Optional[ModuleType] = None,
                 initialize: Optional[Callable[..., torch.Tensor]] = None,
                 activation: Optional[ModuleType] = None,
                 device: Optional[Union[str, int, torch.device]] = None,
                 fixed_std: bool = True):
        assert isinstance(action_space, Box)
        super(ActorPolicy, self).__init__()
        self.action_dim = action_space.shape[0]
        self.representation = representation
        self.representation_info_shape = self.representation.output_shapes
        self.actor = ActorNet(representation.output_shapes['state'][0], self.action_dim, actor_hidden_size,
                              normalize, initialize, activation, device)

    def forward(self, observation: Union[np.ndarray, dict]):
        outputs = self.representation(observation)
        a = self.actor(outputs['state'])
        return outputs, a


class PPGActorCritic(nn.Module):
    def __init__(self,
                 action_space: Space,
                 representation: ModuleType,
                 actor_hidden_size: Sequence[int] = None,
                 critic_hidden_size: Sequence[int] = None,
                 normalize: Optional[ModuleType] = None,
                 initialize: Optional[Callable[..., torch.Tensor]] = None,
                 activation: Optional[ModuleType] = None,
                 device: Optional[Union[str, int, torch.device]] = None):
        assert isinstance(action_space, Box)
        super(PPGActorCritic, self).__init__()
        self.action_dim = action_space.shape[0]
        self.policy_representation = representation
        self.critic_representation = copy.deepcopy(representation)
        self.representation_info_shape = self.policy_representation.output_shapes
        self.actor = ActorNet(representation.output_shapes['state'][0], self.action_dim, actor_hidden_size,
                              normalize, initialize, activation, device)
        self.critic = CriticNet(representation.output_shapes['state'][0], critic_hidden_size,
                                normalize, initialize, activation, device)
        self.aux_critic = CriticNet(representation.output_shapes['state'][0], critic_hidden_size,
                                    normalize, initialize, activation, device)

    def forward(self, observation: Union[np.ndarray, dict]):
        policy_outputs = self.policy_representation(observation)
        critic_outputs = self.critic_representation(observation)
        a = self.actor(policy_outputs['state'])
        v = self.critic(critic_outputs['state'])
        aux_v = self.aux_critic(policy_outputs)
        return policy_outputs, a, v, aux_v


class ActorNet_SAC(nn.Module):
    def __init__(self,
                 state_dim: int,
                 action_dim: int,
                 hidden_sizes: Sequence[int],
                 initialize: Optional[Callable[..., torch.Tensor]] = None,
                 activation: Optional[ModuleType] = None,
                 device: Optional[Union[str, int, torch.device]] = None):
        super(ActorNet_SAC, self).__init__()
        layers = []
        input_shape = (state_dim,)
        for h in hidden_sizes:
            mlp, input_shape = mlp_block(input_shape[0], h, None, activation, initialize, device)
            layers.extend(mlp)
        self.device = device
        self.output = nn.Sequential(*layers)
        self.out_mu = nn.Linear(hidden_sizes[0], action_dim, device=device)
        self.out_std = nn.Linear(hidden_sizes[0], action_dim, device=device)

    def forward(self, x: torch.tensor):
        output = self.output(x)
        mu = torch.tanh(self.out_mu(output))
        # std = torch.tanh(self.out_std(output))
        std = torch.clamp(self.out_std(output), -20, 2)
        std = std.exp()
        # dia_std = torch.diag_embed(std)
        self.dist = torch.distributions.Normal(mu, std)
        return self.dist


class CriticNet_SAC(nn.Module):
    def __init__(self,
                 state_dim: int,
                 action_dim: int,
                 hidden_sizes: Sequence[int],
                 initialize: Optional[Callable[..., torch.Tensor]] = None,
                 activation: Optional[ModuleType] = None,
                 device: Optional[Union[str, int, torch.device]] = None):
        super(CriticNet_SAC, self).__init__()
        layers = []
        input_shape = (state_dim + action_dim,)
        for h in hidden_sizes:
            mlp, input_shape = mlp_block(input_shape[0], h, None, activation, initialize, device)
            layers.extend(mlp)
        layers.extend(mlp_block(input_shape[0], 1, None, None, initialize, device)[0])
        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.tensor, a: torch.tensor):
        return self.model(torch.concat((x, a), dim=-1))[:, 0]


class SACPolicy(nn.Module):
    def __init__(self,
                 action_space: Space,
                 representation: Basic_Identical,
                 actor_hidden_size: Sequence[int],
                 critic_hidden_size: Sequence[int],
                 initialize: Optional[Callable[..., torch.Tensor]] = None,
                 activation: Optional[ModuleType] = None,
                 device: Optional[Union[str, int, torch.device]] = None):
        assert isinstance(action_space, Box)
        super(SACPolicy, self).__init__()
        self.action_dim = action_space.shape[0]
        self.representation = representation
        self.representation_info_shape = self.representation.output_shapes

        self.actor = ActorNet_SAC(representation.output_shapes['state'][0], self.action_dim, actor_hidden_size,
                                  initialize, activation, device)
        self.critic = CriticNet_SAC(representation.output_shapes['state'][0], self.action_dim, critic_hidden_size,
                                    initialize, activation, device)
        self.target_actor = copy.deepcopy(self.actor)
        self.target_critic = copy.deepcopy(self.critic)

    def action(self, observation: Union[np.ndarray, dict]):
        outputs = self.representation(observation)
        act_dist = self.actor(outputs['state'])

        return outputs, act_dist

    def Qtarget(self, observation: Union[np.ndarray, dict]):
        outputs = self.representation(observation)
        act_dist = self.target_actor(outputs['state'])
        act = act_dist.rsample()
        act_log = act_dist.log_prob(act)
        return outputs, act_log, self.target_critic(outputs['state'], act)

    def Qaction(self, observation: Union[np.ndarray, dict], action: torch.Tensor):
        outputs = self.representation(observation)
        return outputs, self.critic(outputs['state'], action)

    def Qpolicy(self, observation: Union[np.ndarray, dict]):
        outputs = self.representation(observation)
        act_dist = self.actor(outputs['state'])
        act = act_dist.rsample()
        act_log = act_dist.log_prob(act)
        return outputs, act_log, self.critic(outputs['state'], act)

    def forward(self):
        return super().forward()

    def soft_update(self, tau=0.005):
        for ep, tp in zip(self.actor.parameters(), self.target_actor.parameters()):
            tp.data.mul_(1 - tau)
            tp.data.add_(tau * ep.data)
        for ep, tp in zip(self.critic.parameters(), self.target_critic.parameters()):
            tp.data.mul_(1 - tau)
            tp.data.add_(tau * ep.data)
