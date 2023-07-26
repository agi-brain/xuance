import torch.distributions

from xuanpolicy.torch.policies import *
from xuanpolicy.torch.utils import *
from xuanpolicy.torch.representations import Basic_Identical
from gymnasium.spaces.box import Box as Box_pettingzoo


class BasicQhead(nn.Module):
    def __init__(self,
                 state_dim: int,
                 action_dim: int,
                 n_agents: int,
                 hidden_sizes: Sequence[int],
                 normalize: Optional[ModuleType] = None,
                 initialize: Optional[Callable[..., torch.Tensor]] = None,
                 activation: Optional[ModuleType] = None,
                 device: Optional[Union[str, int, torch.device]] = None):
        super(BasicQhead, self).__init__()
        layers_ = []
        input_shape = (state_dim + n_agents,)
        for h in hidden_sizes:
            mlp, input_shape = mlp_block(input_shape[0], h, normalize, activation, initialize, device)
            layers_.extend(mlp)
        layers_.extend(mlp_block(input_shape[0], action_dim, None, None, None, device)[0])
        self.model = nn.Sequential(*layers_)

    def forward(self, x: torch.Tensor):
        return self.model(x)


class BasicQnetwork(nn.Module):
    def __init__(self,
                 action_space: Discrete,
                 n_agents: int,
                 representation: Optional[Basic_Identical],
                 hidden_size: Sequence[int] = None,
                 normalize: Optional[ModuleType] = None,
                 initialize: Optional[Callable[..., torch.Tensor]] = None,
                 activation: Optional[ModuleType] = None,
                 device: Optional[Union[str, int, torch.device]] = None):
        super(BasicQnetwork, self).__init__()
        self.action_dim = action_space.n
        self.representation = representation
        self.representation_info_shape = self.representation.output_shapes

        self.eval_Qhead = BasicQhead(self.representation.output_shapes['state'][0], self.action_dim, n_agents,
                                     hidden_size, normalize, initialize, activation, device)
        self.target_Qhead = copy.deepcopy(self.eval_Qhead)

    def forward(self, observation: torch.Tensor, agent_ids: torch.Tensor):
        outputs = self.representation(observation)
        q_inputs = torch.concat([outputs['state'], agent_ids], dim=-1)
        evalQ = self.eval_Qhead(q_inputs)
        argmax_action = evalQ.argmax(dim=-1, keepdim=False)
        return outputs, argmax_action, evalQ

    def target_Q(self, observation: torch.Tensor, agent_ids: torch.Tensor):
        outputs = self.representation(observation)
        q_inputs = torch.concat([outputs['state'], agent_ids], dim=-1)
        return self.target_Qhead(q_inputs)

    def copy_target(self):
        for ep, tp in zip(self.eval_Qhead.parameters(), self.target_Qhead.parameters()):
            tp.data.copy_(ep)



class ActorNet(nn.Module):
    def __init__(self,
                 state_dim: int,
                 n_agents: int,
                 action_dim: int,
                 hidden_sizes: Sequence[int],
                 normalize: Optional[ModuleType] = None,
                 initialize: Optional[Callable[..., torch.Tensor]] = None,
                 activation: Optional[ModuleType] = None,
                 device: Optional[Union[str, int, torch.device]] = None):
        super(ActorNet, self).__init__()
        self.device = device
        layers = []
        input_shape = (state_dim + n_agents,)
        for h in hidden_sizes:
            mlp, input_shape = mlp_block(input_shape[0], h, normalize, activation, initialize, device)
            layers.extend(mlp)
        # layers.extend(mlp_block(input_shape[0], action_dim, None, nn.ReLU, initialize, device)[0])
        # self.mu = nn.Sequential(*layers)
        # self.logstd = nn.Sequential(*layers)
        self.output = nn.Sequential(*layers)
        self.out_mu = nn.Linear(hidden_sizes[0], action_dim, device=device)
        self.out_std = nn.Linear(hidden_sizes[0], action_dim, device=device)

    def forward(self, x: torch.Tensor):
        output = self.output(x)
        mu = torch.sigmoid(self.out_mu(output))
        # std = torch.tanh(self.out_std(output))
        std = torch.clamp(self.out_std(output), -20, 1)
        std = std.exp()
        dia_std = torch.diag_embed(std)
        self.dist = torch.distributions.multivariate_normal.MultivariateNormal(mu, dia_std)

        return self.dist


class CriticNet(nn.Module):
    def __init__(self,
                 independent: bool,
                 state_dim: int,
                 n_agents: int,
                 action_dim: int,
                 hidden_sizes: Sequence[int],
                 normalize: Optional[ModuleType] = None,
                 initialize: Optional[Callable[..., torch.Tensor]] = None,
                 activation: Optional[ModuleType] = None,
                 device: Optional[Union[str, int, torch.device]] = None
                 ):
        super(CriticNet, self).__init__()
        layers = []
        if independent:
            input_shape = (state_dim + action_dim + n_agents,)
        else:
            input_shape = (state_dim * n_agents + action_dim * n_agents + n_agents,)
        for h in hidden_sizes:
            mlp, input_shape = mlp_block(input_shape[0], h, normalize, activation, initialize, device)
            layers.extend(mlp)
        layers.extend(mlp_block(input_shape[0], 1, None, None, initialize, device)[0])
        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.tensor):
        return self.model(x)


class Basic_ISAC_policy(nn.Module):
    def __init__(self,
                 action_space: Space,
                 n_agents: int,
                 representation: Optional[Basic_Identical],
                 actor_hidden_size: Sequence[int],
                 critic_hidden_size: Sequence[int],
                 normalize: Optional[ModuleType] = None,
                 initialize: Optional[Callable[..., torch.Tensor]] = None,
                 activation: Optional[ModuleType] = None,
                 device: Optional[Union[str, int, torch.device]] = None
                 ):
        super(Basic_ISAC_policy, self).__init__()
        self.action_dim = action_space.shape[0]
        self.n_agents = n_agents
        self.representation = representation
        self.representation_info_shape = self.representation.output_shapes

        self.actor_net = ActorNet(representation.output_shapes['state'][0], n_agents, self.action_dim,
                                  actor_hidden_size, normalize, initialize, activation, device)
        self.critic_net = CriticNet(True, representation.output_shapes['state'][0], n_agents, self.action_dim,
                                    critic_hidden_size, normalize, initialize, activation, device)
        self.target_actor_net = copy.deepcopy(self.actor_net)
        self.target_critic_net = copy.deepcopy(self.critic_net)
        self.parameters_actor = list(self.representation.parameters()) + list(self.actor_net.parameters())
        self.parameters_critic = self.critic_net.parameters()

    def forward(self, observation: torch.Tensor, agent_ids: torch.Tensor):
        outputs = self.representation(observation)
        actor_in = torch.concat([outputs['state'], agent_ids], dim=-1)
        act = self.actor_net(actor_in)
        return outputs, act

    def critic(self, observation: torch.Tensor, actions: torch.Tensor, agent_ids: torch.Tensor):
        outputs = self.representation(observation)
        critic_in = torch.concat([outputs['state'], actions, agent_ids], dim=-1)
        return self.critic_net(critic_in)

    def target_critic(self, observation: torch.Tensor, actions: torch.Tensor, agent_ids: torch.Tensor):
        outputs = self.representation(observation)
        critic_in = torch.concat([outputs['state'], actions, agent_ids], dim=-1)
        return self.target_critic_net(critic_in)

    def target_actor(self, observation: torch.Tensor, agent_ids: torch.Tensor):
        outputs = self.representation(observation)
        actor_in = torch.concat([outputs['state'], agent_ids], dim=-1)
        return self.target_actor_net(actor_in)

    def soft_update(self, tau=0.005):
        for ep, tp in zip(self.actor_net.parameters(), self.target_actor_net.parameters()):
            tp.data.mul_(1 - tau)
            tp.data.add_(tau * ep.data)
        for ep, tp in zip(self.critic_net.parameters(), self.target_critic_net.parameters()):
            tp.data.mul_(1 - tau)
            tp.data.add_(tau * ep.data)


class MASAC_policy(Basic_ISAC_policy):
    def __init__(self,
                 action_space: Space,
                 n_agents: int,
                 representation: Optional[Basic_Identical],
                 actor_hidden_size: Sequence[int],
                 critic_hidden_size: Sequence[int],
                 normalize: Optional[ModuleType] = None,
                 initialize: Optional[Callable[..., torch.Tensor]] = None,
                 activation: Optional[ModuleType] = None,
                 device: Optional[Union[str, int, torch.device]] = None
                 ):
        super(MASAC_policy, self).__init__(action_space, n_agents, representation,
                                            actor_hidden_size, critic_hidden_size,
                                            normalize, initialize, activation, device)
        self.critic_net = CriticNet(False, representation.output_shapes['state'][0], n_agents, self.action_dim,
                                    critic_hidden_size, normalize, initialize, activation, device)
        self.target_critic_net = copy.deepcopy(self.critic_net)
        self.parameters_critic = self.critic_net.parameters()

    def critic(self, observation: torch.Tensor, actions: torch.Tensor, agent_ids: torch.Tensor):
        bs = observation.shape[0]
        outputs_n = self.representation(observation)['state'].view(bs, 1, -1).expand(-1, self.n_agents, -1)
        actions_n = actions.view(bs, 1, -1).expand(-1, self.n_agents, -1)
        critic_in = torch.concat([outputs_n, actions_n, agent_ids], dim=-1)
        return self.critic_net(critic_in)

    def target_critic(self, observation: torch.Tensor, actions: torch.Tensor, agent_ids: torch.Tensor):
        bs = observation.shape[0]
        outputs_n = self.representation(observation)['state'].view(bs, 1, -1).expand(-1, self.n_agents, -1)
        actions_n = actions.view(bs, 1, -1).expand(-1, self.n_agents, -1)
        critic_in = torch.concat([outputs_n, actions_n, agent_ids], dim=-1)
        return self.target_critic_net(critic_in)
