from xuanpolicy.mindspore.policies import *
from xuanpolicy.mindspore.utils import *
import copy
from gym.spaces import Space, Box, Discrete, Dict

class BasicQhead(nn.Cell):
    def __init__(self,
                 state_dim: int,
                 action_dim: int,
                 hidden_sizes: Sequence[int],
                 normalize: Optional[ModuleType] = None,
                 initialize: Optional[Callable[..., ms.Tensor]] = None,
                 activation: Optional[ModuleType] = None
                 ):
        super(BasicQhead, self).__init__()
        layers = []
        input_shape = (state_dim,)
        for h in hidden_sizes:
            mlp, input_shape = mlp_block(input_shape[0], h, normalize, activation, initialize)
            layers.extend(mlp)
        layers.extend(mlp_block(input_shape[0], action_dim, None, None, None)[0])
        self.model = nn.SequentialCell(*layers)

    def construct(self, x: ms.tensor):
        return self.model(x)


class DuelQhead(nn.Cell):
    def __init__(self,
                 state_dim: int,
                 action_dim: int,
                 hidden_sizes: Sequence[int],
                 normalize: Optional[ModuleType] = None,
                 initialize: Optional[Callable[..., ms.Tensor]] = None,
                 activation: Optional[ModuleType] = None
                 ):
        super(DuelQhead, self).__init__()
        v_layers = []
        input_shape = (state_dim,)
        for h in hidden_sizes:
            v_mlp, input_shape = mlp_block(input_shape[0], h // 2, normalize, activation, initialize)
            v_layers.extend(v_mlp)
        v_layers.extend(mlp_block(input_shape[0], 1, None, None, None)[0])

        a_layers = []
        input_shape = (state_dim,)
        for h in hidden_sizes:
            a_mlp, input_shape = mlp_block(input_shape[0], h // 2, normalize, activation, initialize)
            a_layers.extend(a_mlp)
        a_layers.extend(mlp_block(input_shape[0], action_dim, None, None, None)[0])

        self.a_model = nn.SequentialCell(*a_layers)
        self.v_model = nn.SequentialCell(*v_layers)

        self._mean = ms.ops.ReduceMean(keep_dims=True)

    def construct(self, x: ms.tensor):
        v = self.v_model(x)
        a = self.a_model(x)
        q = v + (a - self._mean(a))
        return q


class C51Qhead(nn.Cell):
    def __init__(self,
                 state_dim: int,
                 action_dim: int,
                 atom_num: int,
                 hidden_sizes: Sequence[int],
                 normalize: Optional[ModuleType] = None,
                 initialize: Optional[Callable[..., ms.Tensor]] = None,
                 activation: Optional[ModuleType] = None
                 ):
        super(C51Qhead, self).__init__()
        self.action_dim = action_dim
        self.atom_num = atom_num
        layers = []
        input_shape = (state_dim,)
        for h in hidden_sizes:
            mlp, input_shape = mlp_block(input_shape[0], h, normalize, activation, initialize)
            layers.extend(mlp)
        layers.extend(mlp_block(input_shape[0], action_dim * atom_num, None, None, None)[0])
        self.model = nn.SequentialCell(*layers)
        self._softmax = ms.ops.Softmax(axis=-1)

    def construct(self, x: ms.tensor):
        dist_logits = self.model(x).view(-1, self.action_dim, self.atom_num)
        dist_probs = self._softmax(dist_logits)
        return dist_probs

class QRDQNhead(nn.Cell):
    def __init__(self,
                 state_dim: int,
                 action_dim: int,
                 atom_num: int,
                 hidden_sizes: Sequence[int],
                 normalize: Optional[ModuleType] = None,
                 initialize: Optional[Callable[..., ms.Tensor]] = None,
                 activation: Optional[ModuleType] = None
                 ):
        super(QRDQNhead, self).__init__()
        self.action_dim = action_dim
        self.atom_num = atom_num
        layers = []
        input_shape = (state_dim,)
        for h in hidden_sizes:
            mlp, input_shape = mlp_block(input_shape[0], h, normalize, activation, initialize)
            layers.extend(mlp)
        layers.extend(mlp_block(input_shape[0], action_dim * atom_num, None, None, None)[0])
        self.model = nn.SequentialCell(*layers)

    def construct(self, x: ms.tensor):
        return self.model(x).view(-1, self.action_dim, self.atom_num)

class BasicQnetwork(nn.Cell):
    def __init__(self,
                 action_space: Space,
                 representation: ModuleType,
                 hidden_size: Sequence[int] = None,
                 normalize: Optional[ModuleType] = None,
                 initialize: Optional[Callable[..., ms.Tensor]] = None,
                 activation: Optional[ModuleType] = None
                 ):
        super(BasicQnetwork, self).__init__()
        self.action_dim = action_space.n
        self.representation = representation
        self.representation_info_shape = self.representation.output_shapes
        self.eval_Qhead = BasicQhead(self.representation.output_shapes['state'][0], self.action_dim, hidden_size,
                                     normalize, initialize, activation)
        self.target_Qhead = copy.deepcopy(self.eval_Qhead)

    def construct(self, observation: ms.tensor):
        outputs = self.representation(observation)
        evalQ = self.eval_Qhead(outputs['state'])
        targetQ = self.target_Qhead(outputs['state'])
        argmax_action = evalQ.argmax(axis=-1)
        return outputs, argmax_action, evalQ, targetQ

    def trainable_params(self, recurse=True):
        return self.representation.trainable_params() + self.eval_Qhead.trainable_params()

    def copy_target(self):
        for ep, tp in zip(self.eval_Qhead.trainable_params(), self.target_Qhead.trainable_params()):
            tp.assign_value(ep)


class DuelQnetwork(nn.Cell):
    def __init__(self,
                 action_space: Space,
                 representation: ModuleType,
                 hidden_size: Sequence[int] = None,
                 normalize: Optional[ModuleType] = None,
                 initialize: Optional[Callable[..., ms.Tensor]] = None,
                 activation: Optional[ModuleType] = None
                 ):
        super(DuelQnetwork, self).__init__()
        self.action_dim = action_space.n
        self.representation = representation
        self.representation_info_shape = self.representation.output_shapes
        self.eval_Qhead = DuelQhead(self.representation.output_shapes['state'][0], self.action_dim, hidden_size,
                                    normalize, initialize, activation)
        self.target_Qhead = copy.deepcopy(self.eval_Qhead)

    def construct(self, observation: ms.tensor):
        outputs = self.representation(observation)
        evalQ = self.eval_Qhead(outputs['state'])
        targetQ = self.target_Qhead(outputs['state'])
        argmax_action = evalQ.argmax(axis=-1)
        return outputs, argmax_action, evalQ, targetQ

    def trainable_params(self, recurse=True):
        return self.representation.trainable_params() + self.eval_Qhead.trainable_params()

    def copy_target(self):
        for ep, tp in zip(self.eval_Qhead.trainable_params(), self.target_Qhead.trainable_params()):
            tp.assign_value(ep)

class NoisyQnetwork(nn.Cell):
    def __init__(self,
                 action_space: Discrete,
                 representation: ModuleType,
                 hidden_size: Sequence[int] = None,
                 normalize: Optional[ModuleType] = None,
                 initialize: Optional[Callable[..., ms.Tensor]] = None,
                 activation: Optional[ModuleType] = None
                 ):
        super(NoisyQnetwork, self).__init__()
        self.action_dim = action_space.n
        self.representation = representation
        self.representation_info_shape = self.representation.output_shapes
        self.eval_Qhead = BasicQhead(self.representation.output_shapes['state'][0], self.action_dim, hidden_size,
                                    normalize, initialize, activation)
        self.target_Qhead = copy.deepcopy(self.eval_Qhead)

        self._stdnormal = ms.ops.StandardNormal()
        self._assign = ms.ops.Assign()

    def update_noise(self,noisy_bound:float=0.0):
        self.eval_noise_parameter = []
        self.target_noise_parameter = []
        for parameter in self.eval_Qhead.trainable_params():
            self.eval_noise_parameter.append(self._stdnormal(parameter.shape)*noisy_bound)
            self.target_noise_parameter.append(self._stdnormal(parameter.shape)*noisy_bound)
    
    def construct(self, observation: ms.tensor):
        outputs = self.representation(observation)
        for parameter, noise_param in zip(self.eval_Qhead.trainable_params(), self.eval_noise_parameter):
            _ = self._assign(parameter, parameter + noise_param)
        for parameter, noise_param in zip(self.target_Qhead.trainable_params(), self.target_noise_parameter):
            _ = self._assign(parameter, parameter + noise_param)
        evalQ = self.eval_Qhead(outputs['state'])
        targetQ = self.target_Qhead(outputs['state'])
        argmax_action = evalQ.argmax(axis=-1)
        return outputs, argmax_action, evalQ, targetQ

    def trainable_params(self, recurse=True):
        return self.representation.trainable_params() + self.eval_Qhead.trainable_params()

    def copy_target(self):
        for ep, tp in zip(self.eval_Qhead.trainable_params(), self.target_Qhead.trainable_params()):
            tp.assign_value(ep)

class C51Qnetwork(nn.Cell):
    def __init__(self,
                 action_space: Discrete,
                 atom_num: int,
                 vmin: float,
                 vmax: float,
                 representation: ModuleType,
                 hidden_size: Sequence[int] = None,
                 normalize: Optional[ModuleType] = None,
                 initialize: Optional[Callable[..., ms.Tensor]] = None,
                 activation: Optional[ModuleType] = None
                 ):
        assert isinstance(action_space,Discrete)
        super(C51Qnetwork,self).__init__()
        self.action_dim = action_space.n
        self.atom_num = atom_num
        self.vmin = vmin
        self.vmax = vmax
        self.representation = representation
        self.representation_info_shape = self.representation.output_shapes
        self.eval_Zhead = C51Qhead(self.representation.output_shapes['state'][0],self.action_dim,self.atom_num,hidden_size,
                                   normalize,initialize,activation)
        self.target_Zhead = copy.deepcopy(self.eval_Zhead)
        self._LinSpace = ms.ops.LinSpace()
        self.supports = ms.Parameter(self._LinSpace(ms.Tensor(self.vmin, ms.float32),
                                                    ms.Tensor(self.vmax, ms.float32),
                                                    self.atom_num),
                                    requires_grad=False)
        self.deltaz = (vmax - vmin) / (atom_num - 1)
    
    def construct(self,observation: Union[np.ndarray, dict]):
        outputs = self.representation(observation)
        eval_Z = self.eval_Zhead(outputs['state'])
        eval_Q = (self.supports * eval_Z).sum(-1)
        argmax_action = eval_Q.argmax(axis=-1)
        target_Z = self.target_Zhead(outputs['state'])
        return outputs,argmax_action, eval_Z, target_Z
    
    def copy_target(self):
        for ep, tp in zip(self.eval_Zhead.trainable_params(), self.target_Zhead.trainable_params()):
            tp.assign_value(ep)

class QRDQN_Network(nn.Cell):
    def __init__(self,
                 action_space: Discrete,
                 quantile_num: int,
                 representation: ModuleType,
                 hidden_size: Sequence[int] = None,
                 normalize: Optional[ModuleType] = None,
                 initialize: Optional[Callable[..., ms.Tensor]] = None,
                 activation: Optional[ModuleType] = None
                 ):
        assert isinstance(action_space,Discrete)
        super(QRDQN_Network, self).__init__()
        self.action_dim = action_space.n
        self.quantile_num = quantile_num
        self.representation = representation
        self.representation_info_shape = self.representation.output_shapes
        self.eval_Qhead = QRDQNhead(self.representation.output_shapes['state'][0], self.action_dim, self.quantile_num, hidden_size,
                                     normalize, initialize, activation)
        self.target_Qhead = copy.deepcopy(self.eval_Qhead)

        self._mean = ms.ops.ReduceMean()

    def construct(self, observation: ms.tensor):
        outputs = self.representation(observation)
        evalZ = self.eval_Qhead(outputs['state'])
        evalQ = self._mean(evalZ, -1)
        targetQ = self.target_Qhead(outputs['state'])
        argmax_action = evalQ.argmax(axis=-1)
        return outputs, argmax_action, evalZ, targetQ

    def trainable_params(self, recurse=True):
        return self.representation.trainable_params() + self.eval_Qhead.trainable_params()

    def copy_target(self):
        for ep, tp in zip(self.eval_Qhead.trainable_params(), self.target_Qhead.trainable_params()):
            tp.assign_value(ep)


class ActorNet(nn.Cell):
    def __init__(self,
                 state_dim: int,
                 action_dim: int,
                 hidden_sizes: Sequence[int],
                 initialize: Optional[Callable[..., ms.Tensor]] = None,
                 activation: Optional[ModuleType] = None
                 ):
        super(ActorNet, self).__init__()
        layers = []
        input_shape = (state_dim,)
        for h in hidden_sizes:
            mlp, input_shape = mlp_block(input_shape[0], h, None, activation, initialize)
            layers.extend(mlp)
        layers.extend(mlp_block(input_shape[0], action_dim, None, nn.Tanh, initialize)[0])
        self.model = nn.SequentialCell(*layers)

    def construct(self, x: ms.tensor):
        return self.model(x)


class CriticNet(nn.Cell):
    def __init__(self,
                 state_dim: int,
                 action_dim: int,
                 hidden_sizes: Sequence[int],
                 initialize: Optional[Callable[..., ms.Tensor]] = None,
                 activation: Optional[ModuleType] = None
                 ):
        super(CriticNet, self).__init__()
        layers = []
        input_shape = (state_dim + action_dim,)
        for h in hidden_sizes:
            mlp, input_shape = mlp_block(input_shape[0], h, None, activation, initialize)
            layers.extend(mlp)
        layers.extend(mlp_block(input_shape[0], 1, None, None, initialize)[0])
        self._concat = ms.ops.Concat(axis=-1)
        self.model = nn.SequentialCell(*layers)

    def construct(self, x: ms.tensor, a: ms.tensor):
        return self.model(self._concat((x, a)))[:, 0]


class DDPGPolicy(nn.Cell):
    def __init__(self,
                 action_space: Space,
                 representation: ModuleType,
                 actor_hidden_size: Sequence[int],
                 critic_hidden_size: Sequence[int],
                 initialize: Optional[Callable[..., ms.Tensor]] = None,
                 activation: Optional[ModuleType] = None
                 ):
        assert isinstance(action_space, Box)
        super(DDPGPolicy, self).__init__()
        self.action_dim = action_space.shape[0]
        self.representation = representation
        self.representation_info_shape = self.representation.output_shapes

        self.actor = ActorNet(representation.output_shapes['state'][0], self.action_dim, actor_hidden_size, initialize,
                              activation)
        self.critic = CriticNet(representation.output_shapes['state'][0], self.action_dim, critic_hidden_size,
                                initialize, activation)
        self.target_actor = copy.deepcopy(self.actor)
        self.target_critic = copy.deepcopy(self.critic)
        # options
        self._standard_normal = ms.ops.StandardNormal()
        self._min_act, self._max_act = ms.Tensor(-1.0), ms.Tensor(1.0)

    def action(self, observation: ms.tensor, noise_scale: ms.float32):
        outputs = self.representation(observation)
        act = self.actor(outputs[0])
        noise = self._standard_normal(act.shape) * noise_scale
        return outputs, ms.ops.clip_by_value(act + noise, self._min_act, self._max_act)

    def Qtarget(self, observation: ms.tensor):
        outputs = self.representation(observation)
        act = self.target_actor(outputs[0])
        return outputs, self.target_critic(outputs[0], act)

    def Qaction(self, observation: ms.tensor, action: ms.tensor):
        outputs = self.representation(observation)
        return outputs, self.critic(outputs['state'], action)

    def Qpolicy(self, observation: ms.tensor):
        outputs = self.representation(observation)
        return outputs, self.critic(outputs['state'], self.actor(outputs['state']))

    def construct(self):
        return super().construct()

    def soft_update(self, tau=0.005):
        for ep, tp in zip(self.actor.trainable_params(), self.target_actor.trainable_params()):
            tp.assign_value((tau*ep.data+(1-tau)*tp.data))
        for ep, tp in zip(self.critic.trainable_params(), self.target_critic.trainable_params()):
            tp.assign_value((tau*ep.data+(1-tau)*tp.data))


class TD3Policy(nn.Cell):
    def __init__(self,
                 action_space: Space,
                 representation: ModuleType,
                 actor_hidden_size: Sequence[int],
                 critic_hidden_size: Sequence[int],
                 normalize: Optional[ModuleType] = None,
                 initialize: Optional[Callable[..., ms.Tensor]] = None,
                 activation: Optional[ModuleType] = None
                 ):
        assert isinstance(action_space, Box)
        super(TD3Policy, self).__init__()
        self.action_dim = action_space.shape[0]
        self.representation = representation
        self.representation_info_shape = self.representation.output_shapes
        try:
            self.representation_params = self.representation.trainable_params()
        except:
            self.representation_params = []
        self.actor = ActorNet(representation.output_shapes['state'][0], self.action_dim, actor_hidden_size,
                              initialize, activation)
        self.criticA = CriticNet(representation.output_shapes['state'][0], self.action_dim, critic_hidden_size,
                                 initialize, activation)
        self.criticB = CriticNet(representation.output_shapes['state'][0], self.action_dim, critic_hidden_size,
                                 initialize, activation)
        self.target_actor = copy.deepcopy(self.actor)
        self.target_criticA = copy.deepcopy(self.criticA)
        self.target_criticB = copy.deepcopy(self.criticB)
        self.actor_params = self.representation_params + self.actor.trainable_params()
        # options
        self._standard_normal = ms.ops.StandardNormal()
        self._min_act, self._max_act = ms.Tensor(-1.0), ms.Tensor(1.0)
        self._minimum = ms.ops.Minimum()
        self._concat = ms.ops.Concat(axis=-1)
        self._expand_dims = ms.ops.ExpandDims()

    def action(self, observation: ms.tensor, noise_scale: float):
        outputs = self.representation(observation)
        act = self.actor(outputs[0])
        noise = self._standard_normal(act.shape) * noise_scale
        return outputs, ms.ops.clip_by_value(act + noise, self._min_act, self._max_act)

    def Qtarget(self, observation: ms.tensor):
        outputs = self.representation(observation)
        act = self.target_actor(outputs['state'])
        noise = ms.ops.clip_by_value(self._standard_normal(act.shape), self._min_act, self._max_act) * 0.1
        act = ms.ops.clip_by_value(act + noise, self._min_act, self._max_act)
        qa = self._expand_dims(self.target_criticA(outputs['state'], act), 1)
        qb = self._expand_dims(self.target_criticB(outputs['state'], act), 1)
        mim_q = self._minimum(qa, qb)
        return outputs, mim_q

    def Qaction(self, observation: ms.tensor, action: ms.tensor):
        outputs = self.representation(observation)
        qa = self._expand_dims(self.criticA(outputs['state'], action), 1)
        qb = self._expand_dims(self.criticB(outputs['state'], action), 1)
        return outputs, self._concat((qa, qb))

    def Qpolicy(self, observation: ms.tensor):
        outputs = self.representation(observation)
        act = self.actor(outputs['state'])
        qa = self._expand_dims(self.criticA(outputs['state'], act), 1)
        qb = self._expand_dims(self.criticB(outputs['state'], act), 1)
        return outputs, (qa + qb) / 2.0

    def soft_update(self, tau=0.005):
        for ep, tp in zip(self.actor.trainable_params(), self.target_actor.trainable_params()):
            tp.assign_value((tau*ep.data+(1-tau)*tp.data))
        for ep, tp in zip(self.criticA.trainable_params(), self.target_criticA.trainable_params()):
            tp.assign_value((tau*ep.data+(1-tau)*tp.data))
        for ep, tp in zip(self.criticB.trainable_params(), self.target_criticB.trainable_params()):
            tp.assign_value((tau*ep.data+(1-tau)*tp.data))


class CDQNPolicy(nn.Cell):
    def __init__(self,
                 action_space: Space,
                 representation: ModuleType,
                 hidden_size: Sequence[int] = None,
                 normalize: Optional[ModuleType] = None,
                 initialize: Optional[Callable[..., ms.Tensor]] = None,
                 activation: Optional[ModuleType] = None
                 ):
        super(CDQNPolicy, self).__init__()
        self.action_dim = action_space.n
        self.representation = representation
        self.representation_info_shape = self.representation.output_shapes
        self.eval_Qhead = BasicQhead(self.representation.output_shapes['state'][0], self.action_dim, hidden_size,
                                     normalize, initialize, activation)
        self.target_Qhead = copy.deepcopy(self.eval_Qhead)

    def construct(self, observation: ms.tensor):
        outputs = self.representation(observation)
        evalQ = self.eval_Qhead(outputs['state'])
        targetQ = self.target_Qhead(outputs['state'])
        argmax_action = evalQ.argmax(axis=-1)
        return outputs, argmax_action, evalQ, targetQ

    def trainable_params(self, recurse=True):
        return self.representation.trainable_params() + self.eval_Qhead.trainable_params()

    def copy_target(self):
        for ep, tp in zip(self.eval_Qhead.trainable_params(), self.target_Qhead.trainable_params()):
            tp.assign_value(ep)


class LDQNPolicy(nn.Cell):
    def __init__(self,
                 action_space: Space,
                 representation: ModuleType,
                 hidden_size: Sequence[int] = None,
                 normalize: Optional[ModuleType] = None,
                 initialize: Optional[Callable[..., ms.Tensor]] = None,
                 activation: Optional[ModuleType] = None
                 ):
        super(LDQNPolicy, self).__init__()
        self.action_dim = action_space.n
        self.representation = representation
        self.representation_info_shape = self.representation.output_shapes
        self.eval_Qhead = BasicQhead(self.representation.output_shapes['state'][0], self.action_dim, hidden_size,
                                     normalize, initialize, activation)
        self.target_Qhead = copy.deepcopy(self.eval_Qhead)

    def construct(self, observation: ms.tensor):
        outputs = self.representation(observation)
        evalQ = self.eval_Qhead(outputs['state'])
        targetQ = self.target_Qhead(outputs['state'])
        argmax_action = evalQ.argmax(axis=-1)
        return outputs, argmax_action, evalQ, targetQ

    def trainable_params(self, recurse=True):
        return self.representation.trainable_params() + self.eval_Qhead.trainable_params()

    def copy_target(self):
        for ep, tp in zip(self.eval_Qhead.trainable_params(), self.target_Qhead.trainable_params()):
            tp.assign_value(ep)


class CLDQNPolicy(nn.Cell):
    def __init__(self,
                 action_space: Space,
                 representation: ModuleType,
                 hidden_size: Sequence[int] = None,
                 normalize: Optional[ModuleType] = None,
                 initialize: Optional[Callable[..., ms.Tensor]] = None,
                 activation: Optional[ModuleType] = None
                 ):
        super(CLDQNPolicy, self).__init__()
        self.action_dim = action_space.n
        self.representation = representation
        self.representation_info_shape = self.representation.output_shapes
        self.eval_Qhead = BasicQhead(self.representation.output_shapes['state'][0], self.action_dim, hidden_size,
                                     normalize, initialize, activation)
        self.target_Qhead = copy.deepcopy(self.eval_Qhead)

    def construct(self, observation: ms.tensor):
        outputs = self.representation(observation)
        evalQ = self.eval_Qhead(outputs['state'])
        targetQ = self.target_Qhead(outputs['state'])
        argmax_action = evalQ.argmax(axis=-1)
        return outputs, argmax_action, evalQ, targetQ

    def trainable_params(self, recurse=True):
        return self.representation.trainable_params() + self.eval_Qhead.trainable_params()

    def copy_target(self):
        for ep, tp in zip(self.eval_Qhead.trainable_params(), self.target_Qhead.trainable_params()):
            tp.assign_value(ep)


class PDQNPolicy(nn.Cell):
    def __init__(self,
                 observation_space,
                 action_space,
                 representation: ModuleType,
                 conactor_hidden_size: Sequence[int],
                 qnetwork_hidden_size: Sequence[int],
                 normalize: Optional[ModuleType] = None,
                 initialize: Optional[Callable[..., ms.Tensor]] = None,
                 activation: Optional[ModuleType] = None):
        super(PDQNPolicy, self).__init__()
        self.representation = representation
        self.observation_space = observation_space
        self.action_space = action_space
        self.num_disact = self.action_space.spaces[0].n
        self.conact_sizes = np.array([self.action_space.spaces[i].shape[0] for i in range(1, self.num_disact+1)])
        self.conact_size = int(self.conact_sizes.sum())

        self.qnetwork = BasicQhead(self.observation_space.shape[0]+self.conact_size, self.num_disact, qnetwork_hidden_size, normalize,
                                       initialize, nn.ReLU)
        self.conactor = ActorNet(self.observation_space.shape[0], self.conact_size, conactor_hidden_size,
                                  initialize, nn.ReLU)
        self.target_conactor = copy.deepcopy(self.conactor)
        self.target_qnetwork = copy.deepcopy(self.qnetwork)
        self._concat = ms.ops.Concat(1)

    def Atarget(self, state):
        target_conact = self.target_conactor(state)
        return target_conact

    def con_action(self, state):
        state = state.expand_dims(0).astype(ms.float32)
        conaction = self.conactor(state).squeeze()
        return conaction

    def Qtarget(self, state, action):
        input_q = self._concat((state, action))
        target_q = self.target_qnetwork(input_q)
        return target_q

    def Qeval(self, state, action):
        state = state.astype(ms.float32)
        input_q = self._concat((state, action))
        eval_q = self.qnetwork(input_q)
        return eval_q

    def Qpolicy(self, state):
        conact = self.conactor(state)
        input_q = self._concat((state, conact))
        policy_q = (self.qnetwork(input_q)).sum()
        return policy_q

    def construct(self):
        return super().construct()

    def soft_update(self, tau=0.005):
        for ep, tp in zip(self.conactor.trainable_params(), self.target_conactor.trainable_params()):
            tp.assign_value((tau*ep.data+(1-tau)*tp.data))
        for ep, tp in zip(self.qnetwork.trainable_params(), self.target_qnetwork.trainable_params()):
            tp.assign_value((tau*ep.data+(1-tau)*tp.data))


class MPDQNPolicy(nn.Cell):
    def __init__(self,
                 observation_space,
                 action_space,
                 representation: ModuleType,
                 conactor_hidden_size: Sequence[int],
                 qnetwork_hidden_size: Sequence[int],
                 normalize: Optional[ModuleType] = None,
                 initialize: Optional[Callable[..., ms.Tensor]] = None,
                 activation: Optional[ModuleType] = None):
        super(MPDQNPolicy, self).__init__()
        self.representation = representation
        self.observation_space = observation_space
        self.obs_size = self.observation_space.shape[0]
        self.action_space = action_space
        self.num_disact = self.action_space.spaces[0].n
        self.conact_sizes = np.array([self.action_space.spaces[i].shape[0] for i in range(1, self.num_disact+1)])
        self.conact_size = int(self.conact_sizes.sum())

        self.qnetwork = BasicQhead(self.observation_space.shape[0]+self.conact_size, self.num_disact, qnetwork_hidden_size, normalize,
                                       initialize, nn.ReLU)
        self.conactor = ActorNet(self.observation_space.shape[0], self.conact_size, conactor_hidden_size,
                                  initialize, nn.ReLU)
        self.target_conactor = copy.deepcopy(self.conactor)
        self.target_qnetwork = copy.deepcopy(self.qnetwork)

        self.offsets = self.conact_sizes.cumsum()
        self.offsets = np.insert(self.offsets, 0, 0)
        self.offsets = ms.Tensor(self.offsets)

        self._concat = ms.ops.Concat(1)
        self._zeroslike = ms.ops.ZerosLike()
        self._squeeze = ms.ops.Squeeze(1)

    def Atarget(self, state):
        target_conact = self.target_conactor(state)
        return target_conact

    def con_action(self, state):
        # conaction = self.conactor(state)
        state = state.expand_dims(0).astype(ms.float32)
        conaction = self.conactor(state).squeeze()
        return conaction

    def Qtarget(self, state, action):
        batch_size = state.shape[0]
        Q = []
        input_q = self._concat((state, self._zeroslike(action)))
        input_q = input_q.repeat(self.num_disact, 0)
        input_q = input_q.asnumpy()
        action = action.asnumpy()
        for i in range(self.num_disact):
            input_q[i * batch_size:(i + 1) * batch_size, self.obs_size + self.offsets[i]: self.obs_size + self.offsets[i + 1]] \
                = action[:, self.offsets[i]:self.offsets[i + 1]]
        input_q = ms.Tensor(input_q, dtype=ms.float32)
        eval_qall = self.target_qnetwork(input_q)
        for i in range(self.num_disact):
            eval_q = eval_qall[i * batch_size:(i + 1) * batch_size, i]
            if len(eval_q.shape) == 1:
                eval_q = eval_q.expand_dims(1)
            Q.append(eval_q)
        Q = self._concat(Q)
        return Q

    def Qeval(self, state, action,input_q):
        # state = state.astype(ms.float32)
        batch_size = state.shape[0]
        Q = []
        # input_q = self._concat((state, self._zeroslike(action)))
        # input_q = input_q.repeat(self.num_disact, 0)
        # input_q = input_q.asnumpy()
        # action = action.asnumpy()
        # for i in range(self.num_disact):
        #     input_q[i * batch_size:(i + 1) * batch_size, self.obs_size + self.offsets[i]: self.obs_size + self.offsets[i + 1]] \
        #         = action[:, self.offsets[i]:self.offsets[i + 1]]
        #         # = self._squeeze(action[:, self.offsets[i]:self.offsets[i + 1]])
        # input_q = ms.Tensor(input_q, dtype=ms.float32)
        eval_qall = self.qnetwork(input_q)
        for i in range(self.num_disact):
            eval_q = eval_qall[i * batch_size:(i+1) * batch_size, i]
            if len(eval_q.shape) == 1:
                eval_q = eval_q.expand_dims(1)
            Q.append(eval_q)
        Q = self._concat(Q)
        return Q

    def Qpolicy(self, state,input_q):
        # conact = self.conactor(state)
        batch_size = state.shape[0]
        Q = []
        # input_q = self._concat((state, self._zeroslike(conact)))
        # input_q = input_q.repeat(self.num_disact, 0)
        # for i in range(self.num_disact):
        #     input_q[i * batch_size:(i + 1) * batch_size,
        #     self.obs_size + self.offsets[i]: self.obs_size + self.offsets[i + 1]] \
        #         = conact[:, self.offsets[i]:self.offsets[i + 1]]
        eval_qall = self.qnetwork(input_q)
        for i in range(self.num_disact):
            eval_q = eval_qall[i * batch_size:(i + 1) * batch_size, i]
            if len(eval_q.shape) == 1:
                eval_q = eval_q.expand_dims(1)
            Q.append(eval_q)
        Q = self._concat(Q)
        return Q
    
    def construct(self):
        return super().construct()

    def soft_update(self, tau=0.005):
        for ep, tp in zip(self.conactor.trainable_params(), self.target_conactor.trainable_params()):
            tp.assign_value((tau*ep.data+(1-tau)*tp.data))
        for ep, tp in zip(self.qnetwork.trainable_params(), self.target_qnetwork.trainable_params()):
            tp.assign_value((tau*ep.data+(1-tau)*tp.data))


class SPDQNPolicy(nn.Cell):
    def __init__(self,
                 observation_space,
                 action_space,
                 representation: ModuleType,
                 conactor_hidden_size: Sequence[int],
                 qnetwork_hidden_size: Sequence[int],
                 normalize: Optional[ModuleType] = None,
                 initialize: Optional[Callable[..., ms.Tensor]] = None,
                 activation: Optional[ModuleType] = None):
        super(SPDQNPolicy, self).__init__()
        self.representation = representation
        self.observation_space = observation_space
        self.obs_size = self.observation_space.shape[0]
        self.action_space = action_space
        self.num_disact = self.action_space.spaces[0].n
        self.conact_sizes = np.array([self.action_space.spaces[i].shape[0] for i in range(1, self.num_disact + 1)])
        self.conact_size = int(self.conact_sizes.sum())

        self.qnetwork = BasicQhead(self.observation_space.shape[0] + self.conact_size, self.num_disact,
                                   qnetwork_hidden_size, normalize,
                                   initialize, nn.ReLU)
        self.conactor = ActorNet(self.observation_space.shape[0], self.conact_size, conactor_hidden_size,
                                 initialize, nn.ReLU)
        self.target_conactor = copy.deepcopy(self.conactor)
        self.target_qnetwork = copy.deepcopy(self.qnetwork)

        self.offsets = self.conact_sizes.cumsum()
        self.offsets = np.insert(self.offsets, 0, 0)
        self.offsets = ms.Tensor(self.offsets)

        self._concat = ms.ops.Concat(1)
        self._zeroslike = ms.ops.ZerosLike()
        self._squeeze = ms.ops.Squeeze(1)

    def Atarget(self, state):
        target_conact = self.target_conactor(state)
        return target_conact

    def con_action(self, state):
        state = state.expand_dims(0).astype(ms.float32)
        conaction = self.conactor(state).squeeze()
        return conaction

    def Qtarget(self, state, action):
        batch_size = state.shape[0]
        Q = []
        input_q = self._concat((state, self._zeroslike(action)))
        input_q = input_q.repeat(self.num_disact, 0)
        input_q = input_q.asnumpy()
        action = action.asnumpy()
        for i in range(self.num_disact):
            input_q[i * batch_size:(i + 1) * batch_size,
            self.obs_size + self.offsets[i]: self.obs_size + self.offsets[i + 1]] \
                = action[:, self.offsets[i]:self.offsets[i + 1]]
        input_q = ms.Tensor(input_q, dtype=ms.float32)
        eval_qall = self.target_qnetwork(input_q)
        for i in range(self.num_disact):
            eval_q = eval_qall[i * batch_size:(i + 1) * batch_size, i]
            if len(eval_q.shape) == 1:
                eval_q = eval_q.expand_dims(1)
            Q.append(eval_q)
        Q = self._concat(Q)
        return Q

    def Qeval(self, state, action, input_q):
        batch_size = state.shape[0]
        Q = []
        eval_qall = self.qnetwork(input_q)
        for i in range(self.num_disact):
            eval_q = eval_qall[i * batch_size:(i + 1) * batch_size, i]
            if len(eval_q.shape) == 1:
                eval_q = eval_q.expand_dims(1)
            Q.append(eval_q)
        Q = self._concat(Q)
        return Q

    def Qpolicy(self, state, input_q):
        batch_size = state.shape[0]
        Q = []
        eval_qall = self.qnetwork(input_q)
        for i in range(self.num_disact):
            eval_q = eval_qall[i * batch_size:(i + 1) * batch_size, i]
            if len(eval_q.shape) == 1:
                eval_q = eval_q.expand_dims(1)
            Q.append(eval_q)
        Q = self._concat(Q)
        return Q

    def construct(self):
        return super().construct()

    def soft_update(self, tau=0.005):
        for ep, tp in zip(self.conactor.trainable_params(), self.target_conactor.trainable_params()):
            tp.assign_value((tau * ep.data + (1 - tau) * tp.data))
        for ep, tp in zip(self.qnetwork.trainable_params(), self.target_qnetwork.trainable_params()):
            tp.assign_value((tau * ep.data + (1 - tau) * tp.data))
