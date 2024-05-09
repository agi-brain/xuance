import copy

from xuance.torch.policies import *
from xuance.torch.utils import *


class ActorNet(nn.Module):
    def __init__(self,
                 state_dim: int,
                 n_agents: int,
                 action_dim: int,
                 hidden_sizes: Sequence[int],
                 normalize: Optional[ModuleType] = None,
                 initialize: Optional[Callable[..., torch.Tensor]] = None,
                 activation: Optional[ModuleType] = None,
                 activation_action: Optional[ModuleType] = None,
                 device: Optional[Union[str, int, torch.device]] = None):
        super(ActorNet, self).__init__()
        self.device = device
        layers = []
        input_shape = (state_dim + n_agents,)
        for h in hidden_sizes:
            mlp, input_shape = mlp_block(input_shape[0], h, normalize, activation, initialize, device)
            layers.extend(mlp)
        layers.extend(mlp_block(input_shape[0], action_dim, activation=activation_action, device=device)[0])
        self.mu = nn.Sequential(*layers)
        self.log_std = nn.Parameter(-torch.ones((action_dim,), device=device))
        self.dist = DiagGaussianDistribution(action_dim)

    def forward(self, x: torch.Tensor):
        self.dist.set_param(self.mu(x), self.log_std.exp())
        return self.dist


class ActorNet_SAC(nn.Module):
    def __init__(self,
                 state_dim: int,
                 n_agents: int,
                 action_dim: int,
                 hidden_sizes: Sequence[int],
                 normalize: Optional[ModuleType] = None,
                 initialize: Optional[Callable[..., torch.Tensor]] = None,
                 activation: Optional[ModuleType] = None,
                 activation_action: Optional[ModuleType] = None,
                 device: Optional[Union[str, int, torch.device]] = None):
        super(ActorNet_SAC, self).__init__()
        self.device = device
        layers = []
        input_shape = (state_dim + n_agents,)
        for h in hidden_sizes:
            mlp, input_shape = mlp_block(input_shape[0], h, normalize, activation, initialize, device)
            layers.extend(mlp)
        self.output = nn.Sequential(*layers)
        self.mu = nn.Linear(input_shape[0], action_dim, device=device)
        self.log_std = nn.Linear(input_shape[0], action_dim, device=device)
        self.dist = ActivatedDiagGaussianDistribution(action_dim, activation_action, device)

    def forward(self, x: torch.Tensor):
        output = self.output(x)
        mu = self.mu(output)
        log_std = torch.clamp(self.log_std(output), -20, 2)
        std = log_std.exp()
        self.dist.set_param(mu, std)
        return self.dist


class CriticNet(nn.Module):
    def __init__(self,
                 state_dim: int,
                 n_agents: int,
                 hidden_sizes: Sequence[int],
                 normalize: Optional[ModuleType] = None,
                 initialize: Optional[Callable[..., torch.Tensor]] = None,
                 activation: Optional[ModuleType] = None,
                 device: Optional[Union[str, int, torch.device]] = None
                 ):
        super(CriticNet, self).__init__()
        layers = []
        input_shape = (state_dim + n_agents,)
        for h in hidden_sizes:
            mlp, input_shape = mlp_block(input_shape[0], h, normalize, activation, initialize, device)
            layers.extend(mlp)
        layers.extend(mlp_block(input_shape[0], 1, None, None, initialize, device)[0])
        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.tensor):
        return self.model(x)


class MAAC_Policy(nn.Module):
    """
    MAAC_Policy: Multi-Agent Actor-Critic Policy with Gaussian policies
    """

    def __init__(self,
                 action_space: Discrete,
                 n_agents: int,
                 representation: nn.Module,
                 mixer: Optional[VDN_mixer] = None,
                 actor_hidden_size: Sequence[int] = None,
                 critic_hidden_size: Sequence[int] = None,
                 normalize: Optional[ModuleType] = None,
                 initialize: Optional[Callable[..., torch.Tensor]] = None,
                 activation: Optional[ModuleType] = None,
                 activation_action: Optional[ModuleType] = None,
                 device: Optional[Union[str, int, torch.device]] = None,
                 **kwargs):
        super(MAAC_Policy, self).__init__()
        self.device = device
        self.action_dim = action_space.shape[0]
        self.n_agents = n_agents
        self.representation = representation[0]
        self.representation_critic = representation[1]
        self.representation_info_shape = self.representation.output_shapes
        self.lstm = True if kwargs["rnn"] == "LSTM" else False
        self.use_rnn = True if kwargs["use_recurrent"] else False
        self.actor = ActorNet(self.representation.output_shapes['state'][0], n_agents, self.action_dim,
                              actor_hidden_size, normalize, initialize, activation, activation_action, device)
        dim_input_critic = self.representation_critic.output_shapes['state'][0]
        self.critic = CriticNet(dim_input_critic, n_agents, critic_hidden_size,
                                normalize, initialize, activation, device)
        self.mixer = mixer
        self.pi_dist = None

    def forward(self, observation: torch.Tensor, agent_ids: torch.Tensor,
                *rnn_hidden: torch.Tensor, **kwargs):
        if self.use_rnn:
            outputs = self.representation(observation, *rnn_hidden)
            rnn_hidden = (outputs['rnn_hidden'], outputs['rnn_cell'])
        else:
            outputs = self.representation(observation)
            rnn_hidden = None
        actor_input = torch.concat([outputs['state'], agent_ids], dim=-1)
        self.pi_dist = self.actor(actor_input)
        return rnn_hidden, self.pi_dist

    def get_values(self, critic_in: torch.Tensor, agent_ids: torch.Tensor,
                   *rnn_hidden: torch.Tensor, **kwargs):
        shape_input = critic_in.shape
        # get representation features
        if self.use_rnn:
            batch_size, n_agent, episode_length, dim_input = tuple(shape_input)
            outputs = self.representation_critic(critic_in.reshape(-1, episode_length, dim_input), *rnn_hidden)
            outputs['state'] = outputs['state'].reshape(batch_size, n_agent, episode_length, -1)
            rnn_hidden = (outputs['rnn_hidden'], outputs['rnn_cell'])
        else:
            batch_size, n_agent, dim_input = tuple(shape_input)
            outputs = self.representation_critic(critic_in.reshape(-1, dim_input))
            outputs['state'] = outputs['state'].reshape(batch_size, n_agent, -1)
            rnn_hidden = None
        # get critic values
        critic_in = torch.concat([outputs['state'], agent_ids], dim=-1)
        v = self.critic(critic_in)
        return rnn_hidden, v

    def value_tot(self, values_n: torch.Tensor, global_state=None):
        if global_state is not None:
            global_state = torch.as_tensor(global_state).to(self.device)
        return values_n if self.mixer is None else self.mixer(values_n, global_state)


class Basic_ISAC_policy(nn.Module):
    def __init__(self,
                 action_space: Space,
                 n_agents: int,
                 representation: nn.Module,
                 actor_hidden_size: Sequence[int],
                 critic_hidden_size: Sequence[int],
                 normalize: Optional[ModuleType] = None,
                 initialize: Optional[Callable[..., torch.Tensor]] = None,
                 activation: Optional[ModuleType] = None,
                 activation_action: Optional[ModuleType] = None,
                 device: Optional[Union[str, int, torch.device]] = None
                 ):
        super(Basic_ISAC_policy, self).__init__()
        self.action_dim = action_space.shape[0]
        self.activation_action = activation_action
        self.n_agents = n_agents
        self.representation_info_shape = representation.output_shapes
        dim_input_actor = representation.output_shapes['state'][0]
        dim_input_critic = representation.output_shapes['state'][0] + self.action_dim

        self.actor_representation = representation
        self.actor = ActorNet_SAC(dim_input_actor, n_agents, self.action_dim, actor_hidden_size,
                                  normalize, initialize, activation, activation_action, device)

        self.critic_1_representation = copy.deepcopy(representation)
        self.critic_1 = CriticNet(dim_input_critic, n_agents, critic_hidden_size,
                                  normalize, initialize, activation, device)
        self.critic_2_representation = copy.deepcopy(representation)
        self.critic_2 = CriticNet(dim_input_critic, n_agents, critic_hidden_size,
                                  normalize, initialize, activation, device)
        self.target_critic_1_representation = copy.deepcopy(self.critic_1_representation)
        self.target_critic_1 = copy.deepcopy(self.critic_1)
        self.target_critic_2_representation = copy.deepcopy(self.critic_2_representation)
        self.target_critic_2 = copy.deepcopy(self.critic_2)

        self.parameters_actor = list(self.actor_representation.parameters()) + list(self.actor.parameters())
        self.parameters_critic = list(self.critic_1_representation.parameters()) + list(
            self.critic_1.parameters()) + list(self.critic_2_representation.parameters()) + list(
            self.critic_2.parameters())

    def forward(self, observation: torch.Tensor, agent_ids: torch.Tensor):
        outputs_actor = self.actor_representation(observation)
        actor_in = torch.concat([outputs_actor['state'], agent_ids], dim=-1)
        act_dist = self.actor(actor_in)
        act_sample = act_dist.activated_rsample()
        return outputs_actor, act_sample

    def Qpolicy(self, observation: torch.Tensor, agent_ids: torch.Tensor):
        outputs_actor = self.actor_representation(observation)
        outputs_critic_1 = self.critic_1_representation(observation)
        outputs_critic_2 = self.critic_2_representation(observation)

        actor_in = torch.concat([outputs_actor['state'], agent_ids], dim=-1)
        act_dist = self.actor(actor_in)
        act_sample, act_log = act_dist.activated_rsample_and_logprob()

        critic_1_in = torch.concat([outputs_critic_1['state'], act_sample, agent_ids], dim=-1)
        critic_2_in = torch.concat([outputs_critic_2['state'], act_sample, agent_ids], dim=-1)
        q_1, q_2 = self.critic_1(critic_1_in), self.critic_2(critic_2_in)
        return act_log, q_1, q_2

    def Qtarget(self, observation: torch.Tensor, agent_ids: torch.Tensor):
        outputs_actor = self.actor_representation(observation)
        outputs_critic_1 = self.target_critic_1_representation(observation)
        outputs_critic_2 = self.target_critic_2_representation(observation)

        actor_in = torch.concat([outputs_actor['state'], agent_ids], dim=-1)
        new_act_dist = self.actor(actor_in)
        new_act_sample, new_act_log = new_act_dist.activated_rsample_and_logprob()

        critic_1_in = torch.concat([outputs_critic_1['state'], new_act_sample, agent_ids], dim=-1)
        critic_2_in = torch.concat([outputs_critic_2['state'], new_act_sample, agent_ids], dim=-1)
        target_q_1, target_q_2 = self.target_critic_1(critic_1_in), self.target_critic_2(critic_2_in)
        target_q = torch.min(target_q_1, target_q_2)
        return new_act_log, target_q

    def Qaction(self, observation: torch.Tensor, actions: torch.Tensor, agent_ids: torch.Tensor):
        outputs_critic_1 = self.critic_1_representation(observation)
        outputs_critic_2 = self.critic_2_representation(observation)
        critic_1_in = torch.concat([outputs_critic_1['state'], actions, agent_ids], dim=-1)
        critic_2_in = torch.concat([outputs_critic_2['state'], actions, agent_ids], dim=-1)
        q_1, q_2 = self.critic_1(critic_1_in), self.critic_2(critic_2_in)
        return q_1, q_2

    def soft_update(self, tau=0.005):
        for ep, tp in zip(self.critic_1_representation.parameters(), self.target_critic_1_representation.parameters()):
            tp.data.mul_(1 - tau)
            tp.data.add_(tau * ep.data)
        for ep, tp in zip(self.critic_1.parameters(), self.target_critic_1.parameters()):
            tp.data.mul_(1 - tau)
            tp.data.add_(tau * ep.data)
        for ep, tp in zip(self.critic_2_representation.parameters(), self.target_critic_2_representation.parameters()):
            tp.data.mul_(1 - tau)
            tp.data.add_(tau * ep.data)
        for ep, tp in zip(self.critic_2.parameters(), self.target_critic_2.parameters()):
            tp.data.mul_(1 - tau)
            tp.data.add_(tau * ep.data)


class MASAC_policy(Basic_ISAC_policy, nn.Module):
    def __init__(self,
                 action_space: Space,
                 n_agents: int,
                 representation: nn.Module,
                 actor_hidden_size: Sequence[int],
                 critic_hidden_size: Sequence[int],
                 normalize: Optional[ModuleType] = None,
                 initialize: Optional[Callable[..., torch.Tensor]] = None,
                 activation: Optional[ModuleType] = None,
                 activation_action: Optional[ModuleType] = None,
                 device: Optional[Union[str, int, torch.device]] = None
                 ):
        nn.Module.__init__(self)
        self.action_dim = action_space.shape[0]
        self.activation_action = activation_action
        self.n_agents = n_agents
        self.representation_info_shape = representation.output_shapes
        dim_input_actor = representation.output_shapes['state'][0]
        dim_input_critic = (representation.output_shapes['state'][0] + self.action_dim) * self.n_agents

        self.actor_representation = representation
        self.actor = ActorNet_SAC(dim_input_actor, n_agents, self.action_dim, actor_hidden_size,
                                  normalize, initialize, activation, activation_action, device)

        self.critic_1_representation = copy.deepcopy(representation)
        self.critic_1 = CriticNet(dim_input_critic, n_agents, critic_hidden_size,
                                  normalize, initialize, activation, device)
        self.critic_2_representation = copy.deepcopy(representation)
        self.critic_2 = CriticNet(dim_input_critic, n_agents, critic_hidden_size,
                                  normalize, initialize, activation, device)
        self.target_critic_1_representation = copy.deepcopy(self.critic_1_representation)
        self.target_critic_1 = copy.deepcopy(self.critic_1)
        self.target_critic_2_representation = copy.deepcopy(self.critic_2_representation)
        self.target_critic_2 = copy.deepcopy(self.critic_2)

        self.parameters_actor = list(self.actor_representation.parameters()) + list(self.actor.parameters())
        self.parameters_critic = list(self.critic_1_representation.parameters()) + list(
            self.critic_1.parameters()) + list(self.critic_2_representation.parameters()) + list(
            self.critic_2.parameters())

    def Qpolicy(self, observation: torch.Tensor, agent_ids: torch.Tensor):
        bs = observation.shape[0]
        outputs_actor = self.actor_representation(observation)
        outputs_critic_1 = self.critic_1_representation(observation)
        outputs_critic_2 = self.critic_2_representation(observation)

        actor_in = torch.concat([outputs_actor['state'], agent_ids], dim=-1)
        act_dist = self.actor(actor_in)
        act_sample, act_log = act_dist.activated_rsample_and_logprob()

        critic_1_in = torch.concat([outputs_critic_1['state'].view(bs, 1, -1).expand(-1, self.n_agents, -1),
                                    act_sample.view(bs, 1, -1).expand(-1, self.n_agents, -1),
                                    agent_ids], dim=-1)
        critic_2_in = torch.concat([outputs_critic_2['state'].view(bs, 1, -1).expand(-1, self.n_agents, -1),
                                    act_sample.view(bs, 1, -1).expand(-1, self.n_agents, -1),
                                    agent_ids], dim=-1)
        q_1, q_2 = self.critic_1(critic_1_in), self.critic_2(critic_2_in)
        return act_log, q_1, q_2

    def Qtarget(self, observation: torch.Tensor, agent_ids: torch.Tensor):
        bs = observation.shape[0]
        outputs_actor = self.actor_representation(observation)
        outputs_critic_1 = self.target_critic_1_representation(observation)
        outputs_critic_2 = self.target_critic_2_representation(observation)

        actor_in = torch.concat([outputs_actor['state'], agent_ids], dim=-1)
        new_act_dist = self.actor(actor_in)
        new_act_sample, new_act_log = new_act_dist.activated_rsample_and_logprob()

        critic_1_in = torch.concat([outputs_critic_1['state'].view(bs, 1, -1).expand(-1, self.n_agents, -1),
                                    new_act_sample.view(bs, 1, -1).expand(-1, self.n_agents, -1),
                                    agent_ids], dim=-1)
        critic_2_in = torch.concat([outputs_critic_2['state'].view(bs, 1, -1).expand(-1, self.n_agents, -1),
                                    new_act_sample.view(bs, 1, -1).expand(-1, self.n_agents, -1),
                                    agent_ids], dim=-1)
        target_q_1, target_q_2 = self.target_critic_1(critic_1_in), self.target_critic_2(critic_2_in)
        target_q = torch.min(target_q_1, target_q_2)
        return new_act_log, target_q

    def Qaction(self, observation: torch.Tensor, actions: torch.Tensor, agent_ids: torch.Tensor):
        bs = observation.shape[0]
        outputs_critic_1 = self.critic_1_representation(observation)
        outputs_critic_2 = self.critic_2_representation(observation)

        critic_1_in = torch.concat([outputs_critic_1['state'].view(bs, 1, -1).expand(-1, self.n_agents, -1),
                                    actions.view(bs, 1, -1).expand(-1, self.n_agents, -1),
                                    agent_ids], dim=-1)
        critic_2_in = torch.concat([outputs_critic_2['state'].view(bs, 1, -1).expand(-1, self.n_agents, -1),
                                    actions.view(bs, 1, -1).expand(-1, self.n_agents, -1),
                                    agent_ids], dim=-1)
        q_1, q_2 = self.critic_1(critic_1_in), self.critic_2(critic_2_in)
        return q_1, q_2
