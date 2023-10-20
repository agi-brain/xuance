import copy

from xuance.torch.policies import *
from xuance.torch.utils import *
from xuance.torch.representations import Basic_Identical


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
                 representation: nn.Module,
                 actor_hidden_size: Sequence[int] = None,
                 critic_hidden_size: Sequence[int] = None,
                 normalize: Optional[ModuleType] = None,
                 initialize: Optional[Callable[..., torch.Tensor]] = None,
                 activation: Optional[ModuleType] = None,
                 device: Optional[Union[str, int, torch.device]] = None):
        super(ActorCriticPolicy, self).__init__()
        self.action_dim = action_space.shape[0]
        self.representation = representation
        self.representation_info_shape = representation.output_shapes
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
                 representation: nn.Module,
                 actor_hidden_size: Sequence[int] = None,
                 normalize: Optional[ModuleType] = None,
                 initialize: Optional[Callable[..., torch.Tensor]] = None,
                 activation: Optional[ModuleType] = None,
                 device: Optional[Union[str, int, torch.device]] = None,
                 fixed_std: bool = True):
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
                 representation: nn.Module,
                 actor_hidden_size: Sequence[int] = None,
                 critic_hidden_size: Sequence[int] = None,
                 normalize: Optional[ModuleType] = None,
                 initialize: Optional[Callable[..., torch.Tensor]] = None,
                 activation: Optional[ModuleType] = None,
                 device: Optional[Union[str, int, torch.device]] = None):
        super(PPGActorCritic, self).__init__()
        self.action_dim = action_space.shape[0]
        self.actor_representation = representation
        self.critic_representation = copy.deepcopy(representation)
        self.representation_info_shape = self.actor_representation.output_shapes
        self.actor = ActorNet(representation.output_shapes['state'][0], self.action_dim, actor_hidden_size,
                              normalize, initialize, activation, device)
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
                 device: Optional[Union[str, int, torch.device]] = None):
        super(ActorNet_SAC, self).__init__()
        layers = []
        input_shape = (state_dim,)
        for h in hidden_sizes:
            mlp, input_shape = mlp_block(input_shape[0], h, normalize, activation, initialize, device)
            layers.extend(mlp)
        self.device = device
        self.output = nn.Sequential(*layers)
        self.out_mu = nn.Sequential(nn.Linear(hidden_sizes[-1], action_dim, device=device), nn.Tanh())
        self.out_std = nn.Linear(hidden_sizes[-1], action_dim, device=device)

    def forward(self, x: torch.tensor):
        output = self.output(x)
        mu = self.out_mu(output)
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
                 device: Optional[Union[str, int, torch.device]] = None):
        super(SACPolicy, self).__init__()
        self.action_dim = action_space.shape[0]
        self.representation_info_shape = representation.output_shapes
        self.representation_actor = representation
        self.representation_critic = copy.deepcopy(representation)
        self.actor = ActorNet_SAC(representation.output_shapes['state'][0], self.action_dim, actor_hidden_size,
                                  normalize, initialize, activation, device)
        self.critic = CriticNet_SAC(representation.output_shapes['state'][0], self.action_dim, critic_hidden_size,
                                    normalize, initialize, activation, device)

        self.target_representation_actor = copy.deepcopy(self.representation_actor)
        self.target_actor = copy.deepcopy(self.actor)
        self.target_representation_critic = copy.deepcopy(self.representation_critic)
        self.target_critic = copy.deepcopy(self.critic)

    def forward(self, observation: Union[np.ndarray, dict]):
        outputs_actor = self.representation_actor(observation)
        act_dist = self.actor(outputs_actor['state'])
        return outputs_actor, act_dist

    def Qtarget(self, observation: Union[np.ndarray, dict]):
        outputs_actor = self.target_representation_actor(observation)
        outputs_critic = self.target_representation_critic(observation)
        act_dist = self.target_actor(outputs_actor['state'])
        act = act_dist.rsample()
        act_log = act_dist.log_prob(act).sum(-1)
        return act_log, self.target_critic(outputs_critic['state'], act)

    def Qaction(self, observation: Union[np.ndarray, dict], action: torch.Tensor):
        outputs_critic = self.representation_critic(observation)
        return self.critic(outputs_critic['state'], action)

    def Qpolicy(self, observation: Union[np.ndarray, dict]):
        outputs_actor = self.representation_actor(observation)
        outputs_critic = self.representation_critic(observation)
        act_dist = self.actor(outputs_actor['state'])
        act = act_dist.rsample()
        act_log = act_dist.log_prob(act).sum(-1)
        return act_log, self.critic(outputs_critic['state'], act)

    def soft_update(self, tau=0.005):
        for ep, tp in zip(self.representation_actor.parameters(), self.target_representation_actor.parameters()):
            tp.data.mul_(1 - tau)
            tp.data.add_(tau * ep.data)
        for ep, tp in zip(self.representation_critic.parameters(), self.target_representation_critic.parameters()):
            tp.data.mul_(1 - tau)
            tp.data.add_(tau * ep.data)
        for ep, tp in zip(self.actor.parameters(), self.target_actor.parameters()):
            tp.data.mul_(1 - tau)
            tp.data.add_(tau * ep.data)
        for ep, tp in zip(self.critic.parameters(), self.target_critic.parameters()):
            tp.data.mul_(1 - tau)
            tp.data.add_(tau * ep.data)
