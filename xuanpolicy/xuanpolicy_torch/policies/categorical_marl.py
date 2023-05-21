from xuanpolicy.xuanpolicy_torch.policies import *
from xuanpolicy.xuanpolicy_torch.utils import *
from xuanpolicy.xuanpolicy_torch.representations import Basic_Identical
from .deterministic_marl import BasicQhead


class ActorNet(nn.Module):
    def __init__(self,
                 state_dim: int,
                 action_dim: int,
                 n_agents: int,
                 hidden_sizes: Sequence[int],
                 normalize: Optional[ModuleType] = None,
                 initialize: Optional[Callable[..., torch.Tensor]] = None,
                 activation: Optional[ModuleType] = None,
                 device: Optional[Union[str, int, torch.device]] = None):
        super(ActorNet, self).__init__()
        layers = []
        input_shape = (state_dim + n_agents,)
        for h in hidden_sizes:
            mlp, input_shape = mlp_block(input_shape[0], h, normalize, activation, initialize, device)
            layers.extend(mlp)
        layers.extend(mlp_block(input_shape[0], action_dim, None, None, None, device)[0])
        self.model = nn.Sequential(*layers)
        self.dist = CategoricalDistribution(action_dim)

    def forward(self, x: torch.Tensor):
        self.dist.set_param(self.model(x))
        return self.dist


class CriticNet(nn.Module):
    def __init__(self,
                 state_dim: int,
                 n_agents: int,
                 hidden_sizes: Sequence[int],
                 normalize: Optional[ModuleType] = None,
                 initialize: Optional[Callable[..., torch.Tensor]] = None,
                 activation: Optional[ModuleType] = None,
                 device: Optional[Union[str, int, torch.device]] = None):
        super(CriticNet, self).__init__()
        layers = []
        input_shape = (state_dim + n_agents,)
        for h in hidden_sizes:
            mlp, input_shape = mlp_block(input_shape[0], h, normalize, activation, initialize, device)
            layers.extend(mlp)
        layers.extend(mlp_block(input_shape[0], 1, None, None, None, device)[0])
        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor):
        return self.model(x)[:, :, 0]


class COMA_CriticNet(nn.Module):
    def __init__(self,
                 state_dim: int,
                 obs_dim: int,
                 act_dim: int,
                 n_agents: int,
                 hidden_sizes: Sequence[int],
                 normalize: Optional[ModuleType] = None,
                 initialize: Optional[Callable[..., torch.Tensor]] = None,
                 activation: Optional[ModuleType] = None,
                 device: Optional[Union[str, int, torch.device]] = None):
        super(COMA_CriticNet, self).__init__()
        layers = []
        input_shape = (state_dim + obs_dim + act_dim * n_agents + n_agents,)
        for h in hidden_sizes:
            mlp, input_shape = mlp_block(input_shape[0], h, normalize, activation, initialize, device)
            layers.extend(mlp)
        layers.extend(mlp_block(input_shape[0], act_dim, None, None, None, device)[0])
        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor):
        return self.model(x)


class MultiAgentActorCriticPolicy(nn.Module):
    def __init__(self,
                 action_space: Discrete,
                 n_agents: int,
                 representation: Optional[Basic_Identical],
                 mixer: Optional[VDN_mixer] = None,
                 actor_hidden_size: Sequence[int] = None,
                 critic_hidden_size: Sequence[int] = None,
                 normalize: Optional[ModuleType] = None,
                 initialize: Optional[Callable[..., torch.Tensor]] = None,
                 activation: Optional[ModuleType] = None,
                 device: Optional[Union[str, int, torch.device]] = None):
        assert isinstance(action_space, Discrete)
        super(MultiAgentActorCriticPolicy, self).__init__()
        self.device = device
        self.action_dim = action_space.n
        self.representation = representation
        self.representation_info_shape = self.representation.output_shapes
        self.actor = ActorNet(representation.output_shapes['state'][0], self.action_dim, n_agents,
                              actor_hidden_size, normalize, initialize, activation, device)
        self.critic = CriticNet(representation.output_shapes['state'][0], n_agents, critic_hidden_size,
                                normalize, initialize, activation, device)
        self.mixer = mixer

    def forward(self, observation: torch.Tensor, agent_ids: torch.Tensor):
        outputs = self.representation(observation)
        input_with_id = torch.concat([outputs['state'], agent_ids], dim=-1)
        act_dist = self.actor(input_with_id)
        v = self.critic(input_with_id).unsqueeze(-1)
        return outputs, act_dist, v

    def value_tot(self, values_n: torch.Tensor, global_state=None):
        if global_state is not None:
            global_state = torch.as_tensor(global_state).to(self.device)
        return values_n if self.mixer is None else self.mixer(values_n, global_state)


class MAPPO_ActorCriticPolicy(MultiAgentActorCriticPolicy):
    def __init__(self,
                 dim_state: int,
                 action_space: Discrete,
                 n_agents: int,
                 representation: Optional[Basic_Identical],
                 actor_hidden_size: Sequence[int] = None,
                 critic_hidden_size: Sequence[int] = None,
                 normalize: Optional[ModuleType] = None,
                 initialize: Optional[Callable[..., torch.Tensor]] = None,
                 activation: Optional[ModuleType] = None,
                 device: Optional[Union[str, int, torch.device]] = None):
        assert isinstance(action_space, Discrete)
        super(MAPPO_ActorCriticPolicy, self).__init__(action_space, n_agents, representation, None,
                                                      actor_hidden_size, critic_hidden_size,
                                                      normalize, initialize, activation, device)
        self.critic = CriticNet(dim_state, n_agents, critic_hidden_size, normalize, initialize, activation, device)

    def forward(self, observation: torch.Tensor, agent_ids: torch.Tensor):
        outputs = self.representation(observation)
        input_with_id = torch.concat([outputs['state'], agent_ids], dim=-1)
        act_dist = self.actor(input_with_id)
        return outputs, act_dist

    def values(self, state: torch.Tensor, agent_ids: torch.Tensor):
        input_with_id = torch.concat([state, agent_ids], dim=-1)
        return self.critic(input_with_id).unsqueeze(-1)


class MeanFieldActorCriticPolicy(nn.Module):
    def __init__(self,
                 action_space: Discrete,
                 n_agents: int,
                 representation: Optional[Basic_Identical],
                 actor_hidden_size: Sequence[int] = None,
                 critic_hidden_size: Sequence[int] = None,
                 normalize: Optional[ModuleType] = None,
                 initialize: Optional[Callable[..., torch.Tensor]] = None,
                 activation: Optional[ModuleType] = None,
                 device: Optional[Union[str, int, torch.device]] = None
                 ):
        assert isinstance(action_space, Discrete)
        super(MeanFieldActorCriticPolicy, self).__init__()
        self.action_dim = action_space.n
        self.representation = representation
        self.representation_info_shape = self.representation.output_shapes
        self.actor_net = ActorNet(representation.output_shapes['state'][0], self.action_dim, n_agents,
                                  actor_hidden_size, normalize, initialize, activation, device)
        self.critic_net = BasicQhead(representation.output_shapes['state'][0] + self.action_dim, self.action_dim,
                                     n_agents, critic_hidden_size, normalize, initialize, activation, device)
        self.target_actor_net = copy.deepcopy(self.actor_net)
        self.target_critic_net = copy.deepcopy(self.critic_net)
        self.parameters_actor = list(self.actor_net.parameters()) + list(self.representation.parameters())
        self.parameters_critic = self.critic_net.parameters()

    def forward(self, observation: torch.Tensor, agent_ids: torch.Tensor):
        outputs = self.representation(observation)
        input_actor = torch.concat([outputs['state'], agent_ids], dim=-1)
        act_dist = self.actor_net(input_actor)
        return outputs, act_dist

    def target_actor(self, observation: torch.Tensor, agent_ids: torch.Tensor):
        outputs = self.representation(observation)
        input_actor = torch.concat([outputs['state'], agent_ids], dim=-1)
        act_dist = self.target_actor_net(input_actor)
        return act_dist

    def critic(self, observation: torch.Tensor, actions_mean: torch.Tensor, agent_ids: torch.Tensor):
        outputs = self.representation(observation)
        critic_in = torch.concat([outputs['state'], actions_mean, agent_ids], dim=-1)
        return self.critic_net(critic_in)

    def target_critic(self, observation: torch.Tensor, actions_mean: torch.Tensor, agent_ids: torch.Tensor):
        outputs = self.representation(observation)
        critic_in = torch.concat([outputs['state'], actions_mean, agent_ids], dim=-1)
        return self.target_critic_net(critic_in)

    def soft_update(self, tau=0.005):
        for ep, tp in zip(self.actor_net.parameters(), self.target_actor_net.parameters()):
            tp.data.mul_(1 - tau)
            tp.data.add_(tau * ep.data)
        for ep, tp in zip(self.critic_net.parameters(), self.target_critic_net.parameters()):
            tp.data.mul_(1 - tau)
            tp.data.add_(tau * ep.data)


class COMAPolicy(nn.Module):
    def __init__(self,
                 state_dim: int,
                 action_space: Discrete,
                 n_agents: int,
                 representation: Optional[Basic_Identical],
                 actor_hidden_size: Sequence[int] = None,
                 critic_hidden_size: Sequence[int] = None,
                 normalize: Optional[ModuleType] = None,
                 initialize: Optional[Callable[..., torch.Tensor]] = None,
                 activation: Optional[ModuleType] = None,
                 device: Optional[Union[str, int, torch.device]] = None):
        assert isinstance(action_space, Discrete)
        super(COMAPolicy, self).__init__()
        self.device = device
        self.action_dim = action_space.n
        self.n_agents = n_agents
        self.representation = representation
        self.representation_info_shape = self.representation.output_shapes
        self.actor = ActorNet(representation.output_shapes['state'][0], self.action_dim, n_agents,
                              actor_hidden_size, normalize, initialize, activation, device)
        self.critic = COMA_CriticNet(state_dim, representation.output_shapes['state'][0], self.action_dim, n_agents,
                                     critic_hidden_size, normalize, initialize, activation, device)
        self.target_critic = copy.deepcopy(self.critic)
        self.parameters_critic = self.critic.parameters()
        self.parameters_actor = list(self.representation.parameters()) + list(self.actor.parameters())

    def build_critic_in(self, state, observations, actions_onehot, agent_ids, t=None):
        bs, act_dim = state.shape[0], actions_onehot.shape[-1]
        step_len = state.shape[1] if t is None else 1
        ts = slice(None) if t is None else slice(t, t + 1)
        obs_encode = self.representation(observations)['state']
        inputs = [state[:, ts], obs_encode[:, ts]]
        # counterfactual actions inputs
        actions_joint = actions_onehot[:, ts].view(bs, step_len, 1, -1).repeat(1, 1, self.n_agents, 1)
        agent_mask = (1 - torch.eye(self.n_agents)).view(-1, 1).repeat(1, act_dim).view(self.n_agents, -1)
        agent_mask = agent_mask.unsqueeze(0).unsqueeze(0).to(self.device)
        inputs.append(actions_joint * agent_mask)
        inputs.append(agent_ids[:, ts])
        return torch.concat(inputs, dim=-1)

    def forward(self, observation: torch.Tensor, agent_ids: torch.Tensor):
        outputs = self.representation(observation)
        input_with_id = torch.concat([outputs['state'], agent_ids], dim=-1)
        act_dist = self.actor(input_with_id)
        return outputs, act_dist

    def copy_target(self):
        for ep, tp in zip(self.critic.parameters(), self.target_critic.parameters()):
            tp.data.copy_(ep)
