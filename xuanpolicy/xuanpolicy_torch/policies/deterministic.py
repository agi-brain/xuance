from xuanpolicy.xuanpolicy_torch.policies import *
from xuanpolicy.xuanpolicy_torch.utils import *
from xuanpolicy.xuanpolicy_torch.representations import Basic_Identical
from xuanpolicy.xuanpolicy_torch.representations import C_DQN, L_DQN, CL_DQN


class BasicQhead(nn.Module):
    def __init__(self,
                 state_dim: int,
                 action_dim: int,
                 hidden_sizes: Sequence[int],
                 normalize: Optional[ModuleType] = None,
                 initialize: Optional[Callable[..., torch.Tensor]] = None,
                 activation: Optional[ModuleType] = None,
                 device: Optional[Union[str, int, torch.device]] = None):
        super(BasicQhead, self).__init__()
        layers = []
        input_shape = (state_dim,)
        for h in hidden_sizes:
            mlp, input_shape = mlp_block(input_shape[0], h, normalize, activation, initialize, device)
            layers.extend(mlp)
        layers.extend(mlp_block(input_shape[0], action_dim, None, None, None, device)[0])
        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor):
        return self.model(x)


class DuelQhead(nn.Module):
    def __init__(self,
                 state_dim: int,
                 action_dim: int,
                 hidden_sizes: Sequence[int],
                 normalize: Optional[ModuleType] = None,
                 initialize: Optional[Callable[..., torch.Tensor]] = None,
                 activation: Optional[ModuleType] = None,
                 device: Optional[Union[str, int, torch.device]] = None):
        super(DuelQhead, self).__init__()
        v_layers = []
        input_shape = (state_dim,)
        for h in hidden_sizes:
            v_mlp, input_shape = mlp_block(input_shape[0], h // 2, normalize, activation, initialize, device)
            v_layers.extend(v_mlp)
        v_layers.extend(mlp_block(input_shape[0], 1, None, None, None, device)[0])
        a_layers = []
        input_shape = (state_dim,)
        for h in hidden_sizes:
            a_mlp, input_shape = mlp_block(input_shape[0], h // 2, normalize, activation, initialize, device)
            a_layers.extend(a_mlp)
        a_layers.extend(mlp_block(input_shape[0], action_dim, None, None, None, device)[0])
        self.a_model = nn.Sequential(*a_layers)
        self.v_model = nn.Sequential(*v_layers)

    def forward(self, x: torch.Tensor):
        v = self.v_model(x)
        a = self.a_model(x)
        q = v + (a - a.mean(dim=-1).unsqueeze(dim=-1))
        return q


class C51Qhead(nn.Module):
    def __init__(self,
                 state_dim: int,
                 action_dim: int,
                 atom_num: int,
                 hidden_sizes: Sequence[int],
                 normalize: Optional[ModuleType] = None,
                 initialize: Optional[Callable[..., torch.Tensor]] = None,
                 activation: Optional[ModuleType] = None,
                 device: Optional[Union[str, int, torch.device]] = None):
        super(C51Qhead, self).__init__()
        self.action_dim = action_dim
        self.atom_num = atom_num
        layers = []
        input_shape = (state_dim,)
        for h in hidden_sizes:
            mlp, input_shape = mlp_block(input_shape[0], h, normalize, activation, initialize, device)
            layers.extend(mlp)
        layers.extend(mlp_block(input_shape[0], action_dim * atom_num, None, None, None, device)[0])
        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor):
        dist_logits = self.model(x).view(-1, self.action_dim, self.atom_num)
        dist_probs = F.softmax(dist_logits, dim=-1)
        return dist_probs


class QRDQNhead(nn.Module):
    def __init__(self,
                 state_dim: int,
                 action_dim: int,
                 atom_num: int,
                 hidden_sizes: Sequence[int],
                 normalize: Optional[ModuleType] = None,
                 initialize: Optional[Callable[..., torch.Tensor]] = None,
                 activation: Optional[ModuleType] = None,
                 device: Optional[Union[str, int, torch.device]] = None):
        super(QRDQNhead, self).__init__()
        self.action_dim = action_dim
        self.atom_num = atom_num
        layers = []
        input_shape = (state_dim,)
        for h in hidden_sizes:
            mlp, input_shape = mlp_block(input_shape[0], h, normalize, activation, initialize, device)
            layers.extend(mlp)
        layers.extend(mlp_block(input_shape[0], action_dim * atom_num, None, None, None, device)[0])
        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor):
        quantiles = self.model(x).view(-1, self.action_dim, self.atom_num)
        return quantiles


class BasicQnetwork(nn.Module):
    def __init__(self,
                 action_space: Discrete,
                 representation: Basic_Identical,
                 hidden_size: Sequence[int] = None,
                 normalize: Optional[ModuleType] = None,
                 initialize: Optional[Callable[..., torch.Tensor]] = None,
                 activation: Optional[ModuleType] = None,
                 device: Optional[Union[str, int, torch.device]] = None):
        super(BasicQnetwork, self).__init__()
        self.action_dim = action_space.n
        self.representation = representation
        self.representation_info_shape = self.representation.output_shapes
        self.eval_Qhead = BasicQhead(self.representation.output_shapes['state'][0], self.action_dim, hidden_size,
                                     normalize, initialize, activation, device)
        self.target_Qhead = copy.deepcopy(self.eval_Qhead)

    def forward(self, observation: Union[np.ndarray, dict]):
        outputs = self.representation(observation)
        evalQ = self.eval_Qhead(outputs['state'])
        targetQ = self.target_Qhead(outputs['state'])
        argmax_action = evalQ.argmax(dim=-1)
        return outputs, argmax_action, evalQ, targetQ.detach()

    def copy_target(self):
        for ep, tp in zip(self.eval_Qhead.parameters(), self.target_Qhead.parameters()):
            tp.data.copy_(ep)


class DuelQnetwork(nn.Module):
    def __init__(self,
                 action_space: Space,
                 representation: Basic_Identical,
                 hidden_size: Sequence[int] = None,
                 normalize: Optional[ModuleType] = None,
                 initialize: Optional[Callable[..., torch.Tensor]] = None,
                 activation: Optional[ModuleType] = None,
                 device: Optional[Union[str, int, torch.device]] = None):
        super(DuelQnetwork, self).__init__()
        self.action_dim = action_space.n
        self.representation = representation
        self.representation_info_shape = self.representation.output_shapes
        self.eval_Qhead = DuelQhead(self.representation.output_shapes['state'][0], self.action_dim, hidden_size,
                                    normalize, initialize, activation, device)
        self.target_Qhead = copy.deepcopy(self.eval_Qhead)

    def forward(self, observation: Union[np.ndarray, dict]):
        outputs = self.representation(observation)
        evalQ = self.eval_Qhead(outputs['state'])
        targetQ = self.target_Qhead(outputs['state'])
        argmax_action = evalQ.argmax(dim=-1)
        return outputs, argmax_action, evalQ, targetQ

    def copy_target(self):
        for ep, tp in zip(self.eval_Qhead.parameters(), self.target_Qhead.parameters()):
            tp.data.copy_(ep)


class NoisyQnetwork(nn.Module):
    def __init__(self,
                 action_space: Discrete,
                 representation: Basic_Identical,
                 hidden_size: Sequence[int] = None,
                 normalize: Optional[ModuleType] = None,
                 initialize: Optional[Callable[..., torch.Tensor]] = None,
                 activation: Optional[ModuleType] = None,
                 device: Optional[Union[str, int, torch.device]] = None):
        super(NoisyQnetwork, self).__init__()
        self.action_dim = action_space.n
        self.representation = representation
        self.representation_info_shape = self.representation.output_shapes
        self.eval_Qhead = BasicQhead(self.representation.output_shapes['state'][0], self.action_dim, hidden_size,
                                     normalize, initialize, activation, device)
        self.target_Qhead = copy.deepcopy(self.eval_Qhead)

    def update_noise(self, noisy_bound: float = 0.0):
        self.eval_noise_parameter = []
        self.target_noise_parameter = []
        for parameter in self.eval_Qhead.parameters():
            self.eval_noise_parameter.append(torch.randn_like(parameter) * noisy_bound)
            self.target_noise_parameter.append(torch.randn_like(parameter) * noisy_bound)

    def forward(self, observation: Union[np.ndarray, dict]):
        outputs = self.representation(observation)
        for parameter, noise_param in zip(self.eval_Qhead.parameters(), self.eval_noise_parameter):
            parameter.data.copy_(parameter.data + noise_param)
        for parameter, noise_param in zip(self.target_Qhead.parameters(), self.target_noise_parameter):
            parameter.data.copy_(parameter.data + noise_param)
        evalQ = self.eval_Qhead(outputs['state'])
        targetQ = self.target_Qhead(outputs['state'])
        argmax_action = evalQ.argmax(dim=-1)
        return outputs, argmax_action, evalQ, targetQ.detach()

    def copy_target(self):
        for ep, tp in zip(self.eval_Qhead.parameters(), self.target_Qhead.parameters()):
            tp.data.copy_(ep)


class C51Qnetwork(nn.Module):
    def __init__(self,
                 action_space: Discrete,
                 atom_num: int,
                 vmin: float,
                 vmax: float,
                 representation: Basic_Identical,
                 hidden_size: Sequence[int] = None,
                 normalize: Optional[ModuleType] = None,
                 initialize: Optional[Callable[..., torch.Tensor]] = None,
                 activation: Optional[ModuleType] = None,
                 device: Optional[Union[str, int, torch.device]] = None):
        assert isinstance(action_space, Discrete)
        super(C51Qnetwork, self).__init__()
        self.action_dim = action_space.n
        self.atom_num = atom_num
        self.vmin = vmin
        self.vmax = vmax
        self.representation = representation
        self.representation_info_shape = self.representation.output_shapes
        self.eval_Zhead = C51Qhead(self.representation.output_shapes['state'][0], self.action_dim, self.atom_num,
                                   hidden_size,
                                   normalize, initialize, activation, device)
        self.target_Zhead = copy.deepcopy(self.eval_Zhead)
        self.supports = torch.nn.Parameter(torch.linspace(self.vmin, self.vmax, self.atom_num), requires_grad=False).to(device)
        self.deltaz = (vmax - vmin) / (atom_num - 1)

    def forward(self, observation: Union[np.ndarray, dict]):
        outputs = self.representation(observation)
        eval_Z = self.eval_Zhead(outputs['state'])
        eval_Q = (self.supports * eval_Z).sum(-1)
        argmax_action = eval_Q.argmax(dim=-1)
        target_Z = self.target_Zhead(outputs['state'])
        return outputs, argmax_action, eval_Z, target_Z

    def copy_target(self):
        for ep, tp in zip(self.eval_Zhead.parameters(), self.target_Zhead.parameters()):
            tp.data.copy_(ep)


class QRDQN_Network(nn.Module):
    def __init__(self,
                 action_space: Discrete,
                 quantile_num: int,
                 representation: Basic_Identical,
                 hidden_size: Sequence[int] = None,
                 normalize: Optional[ModuleType] = None,
                 initialize: Optional[Callable[..., torch.Tensor]] = None,
                 activation: Optional[ModuleType] = None,
                 device: Optional[Union[str, int, torch.device]] = None):
        assert isinstance(action_space, Discrete)
        super(QRDQN_Network, self).__init__()
        self.action_dim = action_space.n
        self.quantile_num = quantile_num
        self.representation = representation
        self.representation_info_shape = self.representation.output_shapes
        self.eval_Zhead = QRDQNhead(self.representation.output_shapes['state'][0], self.action_dim, self.quantile_num,
                                    hidden_size,
                                    normalize, initialize, activation, device)
        self.target_Zhead = copy.deepcopy(self.eval_Zhead)

    def forward(self, observation: Union[np.ndarray, dict]):
        outputs = self.representation(observation)
        eval_Z = self.eval_Zhead(outputs['state'])
        eval_Q = eval_Z.mean(dim=-1)
        argmax_action = eval_Q.argmax(dim=-1)
        target_Z = self.target_Zhead(outputs['state'])
        return outputs, argmax_action, eval_Z, target_Z

    def copy_target(self):
        for ep, tp in zip(self.eval_Zhead.parameters(), self.target_Zhead.parameters()):
            tp.data.copy_(ep)


class ActorNet(nn.Module):
    def __init__(self,
                 state_dim: int,
                 action_dim: int,
                 hidden_sizes: Sequence[int],
                 initialize: Optional[Callable[..., torch.Tensor]] = None,
                 activation: Optional[ModuleType] = None,
                 device: Optional[Union[str, int, torch.device]] = None):
        super(ActorNet, self).__init__()
        layers = []
        input_shape = (state_dim,)
        for h in hidden_sizes:
            mlp, input_shape = mlp_block(input_shape[0], h, None, activation, initialize, device)
            layers.extend(mlp)
        layers.extend(mlp_block(input_shape[0], action_dim, None, nn.Tanh, initialize, device)[0])
        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.tensor):
        return self.model(x)


class CriticNet(nn.Module):
    def __init__(self,
                 state_dim: int,
                 action_dim: int,
                 hidden_sizes: Sequence[int],
                 initialize: Optional[Callable[..., torch.Tensor]] = None,
                 activation: Optional[ModuleType] = None,
                 device: Optional[Union[str, int, torch.device]] = None):
        super(CriticNet, self).__init__()
        layers = []
        input_shape = (state_dim + action_dim,)
        for h in hidden_sizes:
            mlp, input_shape = mlp_block(input_shape[0], h, None, activation, initialize, device)
            layers.extend(mlp)
        layers.extend(mlp_block(input_shape[0], 1, None, None, initialize, device)[0])
        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.tensor, a: torch.tensor):
        return self.model(torch.concat((x, a), dim=-1))[:, 0]


class DDPGPolicy(nn.Module):
    def __init__(self,
                 action_space: Space,
                 representation: Basic_Identical,
                 actor_hidden_size: Sequence[int],
                 critic_hidden_size: Sequence[int],
                 initialize: Optional[Callable[..., torch.Tensor]] = None,
                 activation: Optional[ModuleType] = None,
                 device: Optional[Union[str, int, torch.device]] = None):
        assert isinstance(action_space, Box)
        super(DDPGPolicy, self).__init__()
        self.action_dim = action_space.shape[0]
        self.representation = representation
        self.representation_info_shape = self.representation.output_shapes

        self.actor = ActorNet(representation.output_shapes['state'][0], self.action_dim, actor_hidden_size, initialize,
                              activation, device)
        self.critic = CriticNet(representation.output_shapes['state'][0], self.action_dim, critic_hidden_size,
                                initialize, activation, device)
        self.target_actor = copy.deepcopy(self.actor)
        self.target_critic = copy.deepcopy(self.critic)

    def action(self, observation: Union[np.ndarray, dict], noise_scale: float):
        outputs = self.representation(observation)
        act = self.actor(outputs['state'])
        noise = torch.randn_like(act) * noise_scale
        return outputs, (act + noise).clamp(-1, 1)

    def Qtarget(self, observation: Union[np.ndarray, dict]):
        outputs = self.representation(observation)
        act = self.target_actor(outputs['state'])
        return outputs, self.target_critic(outputs['state'], act)

    def Qaction(self, observation: Union[np.ndarray, dict], action: torch.Tensor):
        outputs = self.representation(observation)
        return outputs, self.critic(outputs['state'], action)

    def Qpolicy(self, observation: Union[np.ndarray, dict]):
        outputs = self.representation(observation)
        return outputs, self.critic(outputs['state'], self.actor(outputs['state']))

    def forward(self):
        return super().forward()

    def soft_update(self, tau=0.005):
        for ep, tp in zip(self.actor.parameters(), self.target_actor.parameters()):
            tp.data.mul_(1 - tau)
            tp.data.add_(tau * ep.data)
        for ep, tp in zip(self.critic.parameters(), self.target_critic.parameters()):
            tp.data.mul_(1 - tau)
            tp.data.add_(tau * ep.data)


class TD3Policy(nn.Module):
    def __init__(self,
                 action_space: Space,
                 representation: Basic_Identical,
                 actor_hidden_size: Sequence[int],
                 critic_hidden_size: Sequence[int],
                 normalize: Optional[ModuleType] = None,
                 initialize: Optional[Callable[..., torch.Tensor]] = None,
                 activation: Optional[ModuleType] = None,
                 device: Optional[Union[str, int, torch.device]] = None):
        assert isinstance(action_space, Box)
        super(TD3Policy, self).__init__()
        self.action_dim = action_space.shape[0]
        self.representation = representation
        self.representation_info_shape = self.representation.output_shapes
        self.actor = ActorNet(representation.output_shapes['state'][0], self.action_dim, actor_hidden_size,
                              initialize, activation, device)
        self.criticA = CriticNet(representation.output_shapes['state'][0], self.action_dim, critic_hidden_size,
                                 initialize, activation, device)
        self.criticB = CriticNet(representation.output_shapes['state'][0], self.action_dim, critic_hidden_size,
                                 initialize, activation, device)
        self.target_actor = copy.deepcopy(self.actor)
        self.target_criticA = copy.deepcopy(self.criticA)
        self.target_criticB = copy.deepcopy(self.criticB)

    def action(self, observation: Union[np.ndarray, dict], noise_scale: float):
        outputs = self.representation(observation)
        act = self.actor(outputs['state'])
        noise = torch.randn_like(act).clamp(-1, 1) * noise_scale
        return outputs, (act + noise).clamp(-1, 1)

    def Qtarget(self, observation: Union[np.ndarray, dict]):
        outputs = self.representation(observation)
        act = self.target_actor(outputs['state'])
        noise = torch.randn_like(act).clamp(-1, 1) * 0.1
        act = (act + noise).clamp(-1, 1)
        qa = self.target_criticA(outputs['state'], act).unsqueeze(dim=1)
        qb = self.target_criticB(outputs['state'], act).unsqueeze(dim=1)
        mim_q = torch.minimum(qa, qb)
        return outputs, mim_q

    def Qaction(self, observation: Union[np.ndarray, dict], action: torch.Tensor):
        outputs = self.representation(observation)
        qa = self.criticA(outputs['state'], action).unsqueeze(dim=1)
        qb = self.criticB(outputs['state'], action).unsqueeze(dim=1)
        return outputs, torch.cat((qa, qb), axis=-1)

    def Qpolicy(self, observation: Union[np.ndarray, dict]):
        outputs = self.representation(observation)
        act = self.actor(outputs['state'])
        qa = self.criticA(outputs['state'], act).unsqueeze(dim=1)
        qb = self.criticB(outputs['state'], act).unsqueeze(dim=1)
        return outputs, (qa + qb) / 2.0

    def soft_update(self, tau=0.005):
        for ep, tp in zip(self.actor.parameters(), self.target_actor.parameters()):
            tp.data.mul_(1 - tau)
            tp.data.add_(tau * ep.data)
        for ep, tp in zip(self.criticA.parameters(), self.target_criticA.parameters()):
            tp.data.mul_(1 - tau)
            tp.data.add_(tau * ep.data)
        for ep, tp in zip(self.criticB.parameters(), self.target_criticB.parameters()):
            tp.data.mul_(1 - tau)
            tp.data.add_(tau * ep.data)


class PDQNPolicy(nn.Module):
    def __init__(self,
                 observation_space,
                 action_space,
                 representation: Basic_Identical,
                 conactor_hidden_size: Sequence[int],
                 qnetwork_hidden_size: Sequence[int],
                 normalize: Optional[ModuleType] = None,
                 initialize: Optional[Callable[..., torch.Tensor]] = None,
                 activation: Optional[ModuleType] = None,
                 device: Optional[Union[str, int, torch.device]] = None):
        super(PDQNPolicy, self).__init__()
        self.representation = representation
        self.observation_space = observation_space
        self.action_space = action_space
        self.num_disact = self.action_space.spaces[0].n
        self.conact_sizes = np.array([self.action_space.spaces[i].shape[0] for i in range(1, self.num_disact + 1)])
        self.conact_size = int(self.conact_sizes.sum())

        self.qnetwork = BasicQhead(self.observation_space.shape[0] + self.conact_size, self.num_disact,
                                   qnetwork_hidden_size, normalize,
                                   initialize, torch.nn.modules.activation.ReLU, device)
        self.conactor = ActorNet(self.observation_space.shape[0], self.conact_size, conactor_hidden_size,
                                 initialize, torch.nn.modules.activation.ReLU, device)
        self.target_conactor = copy.deepcopy(self.conactor)
        self.target_qnetwork = copy.deepcopy(self.qnetwork)

    def Atarget(self, state):
        target_conact = self.target_conactor(state)
        return target_conact

    def con_action(self, state):
        conaction = self.conactor(state)
        return conaction

    def Qtarget(self, state, action):
        input_q = torch.cat((state, action), dim=1)
        target_q = self.target_qnetwork(input_q)
        return target_q

    def Qeval(self, state, action):
        input_q = torch.cat((state, action), dim=1)
        eval_q = self.qnetwork(input_q)
        return eval_q

    def Qpolicy(self, state):
        conact = self.conactor(state)
        input_q = torch.cat((state, conact), dim=1)
        policy_q = torch.sum(self.qnetwork(input_q))
        return policy_q

    def soft_update(self, tau=0.005):
        for ep, tp in zip(self.conactor.parameters(), self.target_conactor.parameters()):
            tp.data.mul_(1 - tau)
            tp.data.add_(tau * ep.data)
        for ep, tp in zip(self.qnetwork.parameters(), self.target_qnetwork.parameters()):
            tp.data.mul_(1 - tau)
            tp.data.add_(tau * ep.data)


class MPDQNPolicy(nn.Module):
    def __init__(self,
                 observation_space,
                 action_space,
                 representation: Basic_Identical,
                 conactor_hidden_size: Sequence[int],
                 qnetwork_hidden_size: Sequence[int],
                 normalize: Optional[ModuleType] = None,
                 initialize: Optional[Callable[..., torch.Tensor]] = None,
                 activation: Optional[ModuleType] = None,
                 device: Optional[Union[str, int, torch.device]] = None):
        super(MPDQNPolicy, self).__init__()
        self.representation = representation
        self.observation_space = observation_space
        self.obs_size = self.observation_space.shape[0]
        self.action_space = action_space
        self.num_disact = self.action_space.spaces[0].n
        self.conact_sizes = np.array([self.action_space.spaces[i].shape[0] for i in range(1, self.num_disact + 1)])
        self.conact_size = int(self.conact_sizes.sum())

        self.qnetwork = BasicQhead(self.observation_space.shape[0] + self.conact_size, self.num_disact,
                                   qnetwork_hidden_size, normalize,
                                   initialize, torch.nn.modules.activation.ReLU, device)
        self.conactor = ActorNet(self.observation_space.shape[0], self.conact_size, conactor_hidden_size,
                                 initialize, torch.nn.modules.activation.ReLU, device)
        self.target_conactor = copy.deepcopy(self.conactor)
        self.target_qnetwork = copy.deepcopy(self.qnetwork)

        self.offsets = self.conact_sizes.cumsum()
        self.offsets = np.insert(self.offsets, 0, 0)

    def Atarget(self, state):
        target_conact = self.target_conactor(state)
        return target_conact

    def con_action(self, state):
        conaction = self.conactor(state)
        return conaction

    def Qtarget(self, state, action):
        batch_size = state.shape[0]
        Q = []
        input_q = torch.cat((state, torch.zeros_like(action)), dim=1)
        input_q = input_q.repeat(self.num_disact, 1)
        for i in range(self.num_disact):
            input_q[i * batch_size:(i + 1) * batch_size,
            self.obs_size + self.offsets[i]: self.obs_size + self.offsets[i + 1]] \
                = action[:, self.offsets[i]:self.offsets[i + 1]]
        eval_qall = self.target_qnetwork(input_q)
        for i in range(self.num_disact):
            eval_q = eval_qall[i * batch_size:(i + 1) * batch_size, i]
            if len(eval_q.shape) == 1:
                eval_q = eval_q.unsqueeze(1)
            Q.append(eval_q)
        Q = torch.cat(Q, dim=1)
        return Q

    def Qeval(self, state, action):
        batch_size = state.shape[0]
        Q = []
        input_q = torch.cat((state, torch.zeros_like(action)), dim=1)
        input_q = input_q.repeat(self.num_disact, 1)
        for i in range(self.num_disact):
            input_q[i * batch_size:(i + 1) * batch_size,
            self.obs_size + self.offsets[i]: self.obs_size + self.offsets[i + 1]] \
                = action[:, self.offsets[i]:self.offsets[i + 1]]
        eval_qall = self.qnetwork(input_q)
        for i in range(self.num_disact):
            eval_q = eval_qall[i * batch_size:(i + 1) * batch_size, i]
            if len(eval_q.shape) == 1:
                eval_q = eval_q.unsqueeze(1)
            Q.append(eval_q)
        Q = torch.cat(Q, dim=1)
        return Q

    def Qpolicy(self, state):
        conact = self.conactor(state)
        batch_size = state.shape[0]
        Q = []
        input_q = torch.cat((state, torch.zeros_like(conact)), dim=1)
        input_q = input_q.repeat(self.num_disact, 1)
        for i in range(self.num_disact):
            input_q[i * batch_size:(i + 1) * batch_size,
            self.obs_size + self.offsets[i]: self.obs_size + self.offsets[i + 1]] \
                = conact[:, self.offsets[i]:self.offsets[i + 1]]
        eval_qall = self.qnetwork(input_q)
        for i in range(self.num_disact):
            eval_q = eval_qall[i * batch_size:(i + 1) * batch_size, i]
            if len(eval_q.shape) == 1:
                eval_q = eval_q.unsqueeze(1)
            Q.append(eval_q)
        Q = torch.cat(Q, dim=1)
        return Q

    def soft_update(self, tau=0.005):
        for ep, tp in zip(self.conactor.parameters(), self.target_conactor.parameters()):
            tp.data.mul_(1 - tau)
            tp.data.add_(tau * ep.data)
        for ep, tp in zip(self.qnetwork.parameters(), self.target_qnetwork.parameters()):
            tp.data.mul_(1 - tau)
            tp.data.add_(tau * ep.data)


class SPDQNPolicy(nn.Module):
    def __init__(self,
                 observation_space,
                 action_space,
                 representation: Basic_Identical,
                 conactor_hidden_size: Sequence[int],
                 qnetwork_hidden_size: Sequence[int],
                 normalize: Optional[ModuleType] = None,
                 initialize: Optional[Callable[..., torch.Tensor]] = None,
                 activation: Optional[ModuleType] = None,
                 device: Optional[Union[str, int, torch.device]] = None):
        super(SPDQNPolicy, self).__init__()
        self.representation = representation
        self.observation_space = observation_space
        self.action_space = action_space
        self.num_disact = self.action_space.spaces[0].n
        self.conact_sizes = np.array([self.action_space.spaces[i].shape[0] for i in range(1, self.num_disact + 1)])
        self.conact_size = int(self.conact_sizes.sum())
        self.qnetwork = nn.ModuleList()
        for k in range(self.num_disact):
            self.qnetwork.append(
                BasicQhead(self.observation_space.shape[0] + self.conact_sizes[k], 1, qnetwork_hidden_size, normalize,
                           initialize, torch.nn.modules.activation.ReLU, device))
        self.conactor = ActorNet(self.observation_space.shape[0], self.conact_size, conactor_hidden_size,
                                 initialize, torch.nn.modules.activation.ReLU, device)
        self.target_conactor = copy.deepcopy(self.conactor)
        self.target_qnetwork = copy.deepcopy(self.qnetwork)

        self.offsets = self.conact_sizes.cumsum()
        self.offsets = np.insert(self.offsets, 0, 0)

    def Atarget(self, state):
        target_conact = self.target_conactor(state)
        return target_conact

    def con_action(self, state):
        conaction = self.conactor(state)
        return conaction

    def Qtarget(self, state, action):
        target_Q = []
        for i in range(self.num_disact):
            conact = action[:, self.offsets[i]:self.offsets[i + 1]]
            input_q = torch.cat((state, conact), dim=1)
            eval_q = self.target_qnetwork[i](input_q)
            target_Q.append(eval_q)
        target_Q = torch.cat(target_Q, dim=1)
        return target_Q

    def Qeval(self, state, action):
        Q = []
        for i in range(self.num_disact):
            conact = action[:, self.offsets[i]:self.offsets[i + 1]]
            input_q = torch.cat((state, conact), dim=1)
            eval_q = self.qnetwork[i](input_q)
            Q.append(eval_q)
        Q = torch.cat(Q, dim=1)
        return Q

    def Qpolicy(self, state):
        conacts = self.conactor(state)
        Q = []
        for i in range(self.num_disact):
            conact = conacts[:, self.offsets[i]:self.offsets[i + 1]]
            input_q = torch.cat((state, conact), dim=1)
            eval_q = self.qnetwork[i](input_q)
            Q.append(eval_q)
        Q = torch.cat(Q, dim=1)
        return Q

    def soft_update(self, tau=0.005):
        for ep, tp in zip(self.conactor.parameters(), self.target_conactor.parameters()):
            tp.data.mul_(1 - tau)
            tp.data.add_(tau * ep.data)
        for ep, tp in zip(self.qnetwork.parameters(), self.target_qnetwork.parameters()):
            tp.data.mul_(1 - tau)
            tp.data.add_(tau * ep.data)


class CDQNPolicy(nn.Module):
    def __init__(self,
                 action_space: Discrete,
                 representation: C_DQN,
                 hidden_size: Sequence[int] = None,
                 normalize: Optional[ModuleType] = None,
                 initialize: Optional[Callable[..., torch.Tensor]] = None,
                 activation: Optional[ModuleType] = None,
                 device: Optional[Union[str, int, torch.device]] = None):
        super(CDQNPolicy, self).__init__()
        self.action_dim = action_space.n
        self.representation = representation
        self.representation_info_shape = self.representation.output_shapes
        self.eval_Qhead = BasicQhead(self.representation.output_shapes['state'][0], self.action_dim, hidden_size,
                                     normalize, initialize, activation, device)
        self.target_Qhead = copy.deepcopy(self.eval_Qhead)

    def forward(self, observation: Union[np.ndarray, dict]):
        outputs = self.representation(observation)
        evalQ = self.eval_Qhead(outputs['state'])
        targetQ = self.target_Qhead(outputs['state'])
        argmax_action = evalQ.argmax(dim=-1)
        return outputs, argmax_action, evalQ, targetQ.detach()

    def copy_target(self):
        for ep, tp in zip(self.eval_Qhead.parameters(), self.target_Qhead.parameters()):
            tp.data.copy_(ep)


class LDQNPolicy(nn.Module):
    def __init__(self,
                 action_space: Discrete,
                 representation: L_DQN,
                 hidden_size: Sequence[int] = None,
                 normalize: Optional[ModuleType] = None,
                 initialize: Optional[Callable[..., torch.Tensor]] = None,
                 activation: Optional[ModuleType] = None,
                 device: Optional[Union[str, int, torch.device]] = None):
        super(LDQNPolicy, self).__init__()
        self.action_dim = action_space.n
        self.representation = representation
        self.representation_info_shape = self.representation.output_shapes
        self.eval_Qhead = BasicQhead(self.representation.output_shapes['state'][0], self.action_dim, hidden_size,
                                     normalize, initialize, activation, device)
        self.target_Qhead = copy.deepcopy(self.eval_Qhead)

    def forward(self, observation: Union[np.ndarray, dict]):
        outputs = self.representation(observation)
        evalQ = self.eval_Qhead(outputs['state'])
        targetQ = self.target_Qhead(outputs['state'])
        argmax_action = evalQ.argmax(dim=-1)
        return outputs, argmax_action, evalQ, targetQ.detach()

    def copy_target(self):
        for ep, tp in zip(self.eval_Qhead.parameters(), self.target_Qhead.parameters()):
            tp.data.copy_(ep)


class CLDQNPolicy(nn.Module):
    def __init__(self,
                 action_space: Discrete,
                 representation: CL_DQN,
                 hidden_size: Sequence[int] = None,
                 normalize: Optional[ModuleType] = None,
                 initialize: Optional[Callable[..., torch.Tensor]] = None,
                 activation: Optional[ModuleType] = None,
                 device: Optional[Union[str, int, torch.device]] = None):
        super(CLDQNPolicy, self).__init__()
        self.action_dim = action_space.n
        self.representation = representation
        self.representation_info_shape = self.representation.output_shapes
        self.eval_Qhead = BasicQhead(self.representation.output_shapes['state'][0], self.action_dim, hidden_size,
                                     normalize, initialize, activation, device)
        self.target_Qhead = copy.deepcopy(self.eval_Qhead)

    def forward(self, observation: Union[np.ndarray, dict]):
        outputs = self.representation(observation)
        evalQ = self.eval_Qhead(outputs['state'])
        targetQ = self.target_Qhead(outputs['state'])
        argmax_action = evalQ.argmax(dim=-1)
        return outputs, argmax_action, evalQ, targetQ.detach()

    def copy_target(self):
        for ep, tp in zip(self.eval_Qhead.parameters(), self.target_Qhead.parameters()):
            tp.data.copy_(ep)
