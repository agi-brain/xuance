import mindspore.nn as nn
import numpy as np
from copy import deepcopy
from gym.spaces import Space, Discrete
from xuance.common import Sequence, Optional, Callable, Union
from xuance.mindspore import ms, Module, Tensor
from xuance.mindspore.utils import ModuleType
from .core import BasicQhead, BasicRecurrent, DuelQhead, C51Qhead, QRDQNhead, ActorNet, CriticNet


class BasicQnetwork(Module):
    def __init__(self,
                 action_space: Discrete,
                 representation: Module,
                 hidden_size: Sequence[int] = None,
                 normalize: Optional[ModuleType] = None,
                 initialize: Optional[Callable[..., Tensor]] = None,
                 activation: Optional[ModuleType] = None):
        super(BasicQnetwork, self).__init__()
        self.action_dim = action_space.n
        self.representation = representation
        self.target_representation = deepcopy(representation)
        self.representation_info_shape = self.representation.output_shapes
        self.eval_Qhead = BasicQhead(self.representation.output_shapes['state'][0], self.action_dim, hidden_size,
                                     normalize, initialize, activation)
        self.target_Qhead = deepcopy(self.eval_Qhead)

    def construct(self, observation: Tensor):
        outputs = self.representation(observation)
        evalQ = self.eval_Qhead(outputs['state'])
        argmax_action = evalQ.argmax(axis=-1)
        return outputs, argmax_action, evalQ

    def target(self, observation: Tensor):
        outputs_target = self.target_representation(observation)
        targetQ = self.target_Qhead(outputs_target['state'])
        argmax_action = targetQ.argmax(axis=-1)
        return outputs_target, argmax_action, targetQ

    def trainable_params(self, recurse=True):
        return self.representation.trainable_params() + self.eval_Qhead.trainable_params()

    def copy_target(self):
        for ep, tp in zip(self.representation.trainable_params(), self.target_representation.trainable_params()):
            tp.assign_value(ep)
        for ep, tp in zip(self.eval_Qhead.trainable_params(), self.target_Qhead.trainable_params()):
            tp.assign_value(ep)


class DuelQnetwork(Module):
    def __init__(self,
                 action_space: Space,
                 representation: ModuleType,
                 hidden_size: Sequence[int] = None,
                 normalize: Optional[ModuleType] = None,
                 initialize: Optional[Callable[..., Tensor]] = None,
                 activation: Optional[ModuleType] = None
                 ):
        super(DuelQnetwork, self).__init__()
        self.action_dim = action_space.n
        self.representation = representation
        self.target_representation = deepcopy(representation)
        self.representation_info_shape = self.representation.output_shapes
        self.eval_Qhead = DuelQhead(self.representation.output_shapes['state'][0], self.action_dim, hidden_size,
                                    normalize, initialize, activation)
        self.target_Qhead = deepcopy(self.eval_Qhead)

    def construct(self, observation: Tensor):
        outputs = self.representation(observation)
        evalQ = self.eval_Qhead(outputs['state'])
        argmax_action = evalQ.argmax(axis=-1)
        return outputs, argmax_action, evalQ

    def target(self, observation: Tensor):
        outputs = self.target_representation(observation)
        targetQ = self.target_Qhead(outputs['state'])
        argmax_action = targetQ.argmax(axis=-1)
        return outputs, argmax_action, targetQ

    def trainable_params(self, recurse=True):
        return self.representation.trainable_params() + self.eval_Qhead.trainable_params()

    def copy_target(self):
        for ep, tp in zip(self.representation.trainable_params(), self.target_representation.trainable_params()):
            tp.assign_value(ep)
        for ep, tp in zip(self.eval_Qhead.trainable_params(), self.target_Qhead.trainable_params()):
            tp.assign_value(ep)


class NoisyQnetwork(Module):
    def __init__(self,
                 action_space: Discrete,
                 representation: ModuleType,
                 hidden_size: Sequence[int] = None,
                 normalize: Optional[ModuleType] = None,
                 initialize: Optional[Callable[..., Tensor]] = None,
                 activation: Optional[ModuleType] = None
                 ):
        super(NoisyQnetwork, self).__init__()
        self.action_dim = action_space.n
        self.representation = representation
        self.target_representation = deepcopy(representation)
        self.representation_info_shape = self.representation.output_shapes
        self.eval_Qhead = BasicQhead(self.representation.output_shapes['state'][0], self.action_dim, hidden_size,
                                     normalize, initialize, activation)
        self.target_Qhead = deepcopy(self.eval_Qhead)

        self._stdnormal = ms.ops.StandardNormal()
        self._assign = ms.ops.Assign()

    def update_noise(self, noisy_bound: float = 0.0):
        self.eval_noise_parameter = []
        self.target_noise_parameter = []
        for parameter in self.eval_Qhead.trainable_params():
            self.eval_noise_parameter.append(self._stdnormal(parameter.shape) * noisy_bound)
            self.target_noise_parameter.append(self._stdnormal(parameter.shape) * noisy_bound)

    def noisy_parameters(self, is_target=False):
        self.update_noise(self.noise_scale)
        if is_target:
            for parameter, noise_param in zip(self.eval_Qhead.trainable_params(), self.eval_noise_parameter):
                _ = self._assign(parameter, parameter + noise_param)
        else:
            for parameter, noise_param in zip(self.target_Qhead.trainable_params(), self.target_noise_parameter):
                _ = self._assign(parameter, parameter + noise_param)

    def construct(self, observation: Tensor):
        outputs = self.representation(observation)
        evalQ = self.eval_Qhead(outputs['state'])
        argmax_action = evalQ.argmax(axis=-1)
        return outputs, argmax_action, evalQ

    def target(self, observation: Tensor):
        outputs = self.target_representation(observation)
        self.noisy_parameters(is_target=True)
        targetQ = self.target_Qhead(outputs['state'])
        argmax_action = targetQ.argmax(axis=-1)
        return outputs, argmax_action, targetQ

    def trainable_params(self, recurse=True):
        return self.representation.trainable_params() + self.eval_Qhead.trainable_params()

    def copy_target(self):
        for ep, tp in zip(self.representation.trainable_params(), self.target_representation.trainable_params()):
            tp.assign_value(ep)
        for ep, tp in zip(self.eval_Qhead.trainable_params(), self.target_Qhead.trainable_params()):
            tp.assign_value(ep)


class C51Qnetwork(Module):
    def __init__(self,
                 action_space: Discrete,
                 atom_num: int,
                 v_min: float,
                 v_max: float,
                 representation: ModuleType,
                 hidden_size: Sequence[int] = None,
                 normalize: Optional[ModuleType] = None,
                 initialize: Optional[Callable[..., Tensor]] = None,
                 activation: Optional[ModuleType] = None
                 ):
        assert isinstance(action_space, Discrete)
        super(C51Qnetwork, self).__init__()
        self.action_dim = action_space.n
        self.atom_num = atom_num
        self.v_min = v_min
        self.v_max = v_max
        self.representation = representation
        self.target_representation = deepcopy(representation)
        self.representation_info_shape = self.representation.output_shapes
        self.eval_Zhead = C51Qhead(self.representation.output_shapes['state'][0], self.action_dim, self.atom_num,
                                   hidden_size, normalize, initialize, activation)
        self.target_Zhead = deepcopy(self.eval_Zhead)
        self._LinSpace = ms.ops.LinSpace()
        self.supports = ms.Parameter(self._LinSpace(Tensor(self.v_min, ms.float32),
                                                    Tensor(self.v_max, ms.float32),
                                                    self.atom_num),
                                     requires_grad=False)
        self.deltaz = (v_max - v_min) / (atom_num - 1)

    def construct(self, observation: Union[np.ndarray, dict]):
        outputs = self.representation(observation)
        eval_Z = self.eval_Zhead(outputs['state'])
        eval_Q = (self.supports * eval_Z).sum(-1)
        argmax_action = eval_Q.argmax(axis=-1)
        return outputs, argmax_action, eval_Z

    def target(self, observation: Union[np.ndarray, dict]):
        outputs = self.target_representation(observation)
        target_Z = self.target_Zhead(outputs['state'])
        target_Q = (self.supports * target_Z).sum(-1)
        argmax_action = target_Q.argmax(dim=-1)
        return outputs, argmax_action, target_Z

    def copy_target(self):
        for ep, tp in zip(self.representation.trainable_params(), self.target_representation.trainable_params()):
            tp.assign_value(ep)
        for ep, tp in zip(self.eval_Zhead.trainable_params(), self.target_Zhead.trainable_params()):
            tp.assign_value(ep)


class QRDQN_Network(Module):
    def __init__(self,
                 action_space: Discrete,
                 quantile_num: int,
                 representation: ModuleType,
                 hidden_size: Sequence[int] = None,
                 normalize: Optional[ModuleType] = None,
                 initialize: Optional[Callable[..., Tensor]] = None,
                 activation: Optional[ModuleType] = None
                 ):
        super(QRDQN_Network, self).__init__()
        self.action_dim = action_space.n
        self.quantile_num = quantile_num
        self.representation = representation
        self.target_representation = deepcopy(representation)
        self.representation_info_shape = self.representation.output_shapes
        self.eval_Zhead = QRDQNhead(self.representation.output_shapes['state'][0], self.action_dim, self.quantile_num,
                                    hidden_size,
                                    normalize, initialize, activation)
        self.target_Zhead = deepcopy(self.eval_Zhead)

        self._mean = ms.ops.ReduceMean()

    def construct(self, observation: Tensor):
        outputs = self.representation(observation)
        evalZ = self.eval_Zhead(outputs['state'])
        evalQ = self._mean(evalZ, -1)
        argmax_action = evalQ.argmax(axis=-1)
        return outputs, argmax_action, evalZ

    def target(self, observation: Tensor):
        outputs = self.target_representation(observation)
        target_Z = self.target_Zhead(outputs['state'])
        target_Q = self._mean(target_Z, -1)
        argmax_action = target_Q.argmax(axis=-1)
        return outputs, argmax_action, target_Z

    def trainable_params(self, recurse=True):
        return self.representation.trainable_params() + self.eval_Zhead.trainable_params()

    def copy_target(self):
        for ep, tp in zip(self.representation.trainable_params(), self.target_representation.trainable_params()):
            tp.assign_value(ep)
        for ep, tp in zip(self.eval_Zhead.trainable_params(), self.target_Zhead.trainable_params()):
            tp.assign_value(ep)


class DDPGPolicy(Module):
    def __init__(self,
                 action_space: Space,
                 representation: ModuleType,
                 actor_hidden_size: Sequence[int],
                 critic_hidden_size: Sequence[int],
                 initialize: Optional[Callable[..., Tensor]] = None,
                 activation: Optional[ModuleType] = None
                 ):
        super(DDPGPolicy, self).__init__()
        self.action_dim = action_space.shape[0]
        self.representation = representation
        self.representation_info_shape = self.representation.output_shapes

        self.actor = ActorNet(representation.output_shapes['state'][0], self.action_dim, actor_hidden_size, initialize,
                              activation)
        self.critic = CriticNet(representation.output_shapes['state'][0], self.action_dim, critic_hidden_size,
                                initialize, activation)
        self.target_actor = deepcopy(self.actor)
        self.target_critic = deepcopy(self.critic)
        # options
        self._standard_normal = ms.ops.StandardNormal()
        self._min_act, self._max_act = Tensor(-1.0), Tensor(1.0)

    def construct(self, observation: Tensor):
        outputs = self.representation(observation)
        act = self.actor(outputs['state'])
        return outputs, act

    def Qtarget(self, observation: Tensor):
        outputs = self.representation(observation)
        act = self.target_actor(outputs['state'])
        return self.target_critic(outputs['state'], act)

    def Qaction(self, observation: Tensor, action: Tensor):
        outputs = self.representation(observation)
        return self.critic(outputs['state'], action)

    def Qpolicy(self, observation: Tensor):
        outputs = self.representation(observation)
        return self.critic(outputs['state'], self.actor(outputs['state']))

    def soft_update(self, tau=0.005):
        for ep, tp in zip(self.actor.trainable_params(), self.target_actor.trainable_params()):
            tp.assign_value((tau * ep.data + (1 - tau) * tp.data))
        for ep, tp in zip(self.critic.trainable_params(), self.target_critic.trainable_params()):
            tp.assign_value((tau * ep.data + (1 - tau) * tp.data))


class TD3Policy(Module):
    def __init__(self,
                 action_space: Space,
                 representation: ModuleType,
                 actor_hidden_size: Sequence[int],
                 critic_hidden_size: Sequence[int],
                 normalize: Optional[ModuleType] = None,
                 initialize: Optional[Callable[..., Tensor]] = None,
                 activation: Optional[ModuleType] = None
                 ):
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
        self.target_actor = deepcopy(self.actor)
        self.target_criticA = deepcopy(self.criticA)
        self.target_criticB = deepcopy(self.criticB)
        self.actor_params = self.representation_params + self.actor.trainable_params()
        # options
        self._standard_normal = ms.ops.StandardNormal()
        self._min_act, self._max_act = Tensor(-1.0), Tensor(1.0)
        self._minimum = ms.ops.Minimum()
        self._concat = ms.ops.Concat(axis=-1)
        self._expand_dims = ms.ops.ExpandDims()

    def action(self, observation: Tensor):
        outputs = self.representation(observation)
        act = self.actor(outputs['state'])
        return outputs, act

    def Qtarget(self, observation: Tensor):
        outputs = self.representation(observation)
        act = self.target_actor(outputs['state'])
        noise = ms.ops.clip_by_value(self._standard_normal(act.shape), self._min_act, self._max_act) * 0.1
        act = ms.ops.clip_by_value(act + noise, self._min_act, self._max_act)
        qa = self._expand_dims(self.target_criticA(outputs['state'], act), 1)
        qb = self._expand_dims(self.target_criticB(outputs['state'], act), 1)
        mim_q = self._minimum(qa, qb)
        return outputs, mim_q

    def Qaction(self, observation: Tensor, action: Tensor):
        outputs = self.representation(observation)
        qa = self._expand_dims(self.criticA(outputs['state'], action), 1)
        qb = self._expand_dims(self.criticB(outputs['state'], action), 1)
        return outputs, self._concat((qa, qb))

    def Qpolicy(self, observation: Tensor):
        outputs = self.representation(observation)
        act = self.actor(outputs['state'])
        qa = self._expand_dims(self.criticA(outputs['state'], act), 1)
        qb = self._expand_dims(self.criticB(outputs['state'], act), 1)
        return outputs, (qa + qb) / 2.0

    def soft_update(self, tau=0.005):
        for ep, tp in zip(self.actor.trainable_params(), self.target_actor.trainable_params()):
            tp.assign_value((tau * ep.data + (1 - tau) * tp.data))
        for ep, tp in zip(self.criticA.trainable_params(), self.target_criticA.trainable_params()):
            tp.assign_value((tau * ep.data + (1 - tau) * tp.data))
        for ep, tp in zip(self.criticB.trainable_params(), self.target_criticB.trainable_params()):
            tp.assign_value((tau * ep.data + (1 - tau) * tp.data))


class PDQNPolicy(Module):
    def __init__(self,
                 observation_space,
                 action_space,
                 representation: ModuleType,
                 conactor_hidden_size: Sequence[int],
                 qnetwork_hidden_size: Sequence[int],
                 normalize: Optional[ModuleType] = None,
                 initialize: Optional[Callable[..., Tensor]] = None,
                 activation: Optional[ModuleType] = None):
        super(PDQNPolicy, self).__init__()
        self.representation = representation
        self.observation_space = observation_space
        self.action_space = action_space
        self.num_disact = self.action_space.spaces[0].n
        self.conact_sizes = np.array([self.action_space.spaces[i].shape[0] for i in range(1, self.num_disact + 1)])
        self.conact_size = int(self.conact_sizes.sum())

        self.qnetwork = BasicQhead(self.observation_space.shape[0] + self.conact_size, self.num_disact,
                                   qnetwork_hidden_size, normalize,
                                   initialize, nn.ReLU)
        self.conactor = ActorNet(self.observation_space.shape[0], self.conact_size, conactor_hidden_size,
                                 initialize, nn.ReLU)
        self.target_conactor = deepcopy(self.conactor)
        self.target_qnetwork = deepcopy(self.qnetwork)
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
            tp.assign_value((tau * ep.data + (1 - tau) * tp.data))
        for ep, tp in zip(self.qnetwork.trainable_params(), self.target_qnetwork.trainable_params()):
            tp.assign_value((tau * ep.data + (1 - tau) * tp.data))


class MPDQNPolicy(Module):
    def __init__(self,
                 observation_space,
                 action_space,
                 representation: ModuleType,
                 conactor_hidden_size: Sequence[int],
                 qnetwork_hidden_size: Sequence[int],
                 normalize: Optional[ModuleType] = None,
                 initialize: Optional[Callable[..., Tensor]] = None,
                 activation: Optional[ModuleType] = None):
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
                                   initialize, nn.ReLU)
        self.conactor = ActorNet(self.observation_space.shape[0], self.conact_size, conactor_hidden_size,
                                 initialize, nn.ReLU)
        self.target_conactor = deepcopy(self.conactor)
        self.target_qnetwork = deepcopy(self.qnetwork)

        self.offsets = self.conact_sizes.cumsum()
        self.offsets = np.insert(self.offsets, 0, 0)
        self.offsets = Tensor(self.offsets)

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
            input_q[i * batch_size:(i + 1) * batch_size,
            self.obs_size + self.offsets[i]: self.obs_size + self.offsets[i + 1]] \
                = action[:, self.offsets[i]:self.offsets[i + 1]]
        input_q = Tensor(input_q, dtype=ms.float32)
        eval_qall = self.target_qnetwork(input_q)
        for i in range(self.num_disact):
            eval_q = eval_qall[i * batch_size:(i + 1) * batch_size, i]
            if len(eval_q.shape) == 1:
                eval_q = eval_q.expand_dims(1)
            Q.append(eval_q)
        Q = self._concat(Q)
        return Q

    def Qeval(self, state, action, input_q):
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
        # input_q = Tensor(input_q, dtype=ms.float32)
        eval_qall = self.qnetwork(input_q)
        for i in range(self.num_disact):
            eval_q = eval_qall[i * batch_size:(i + 1) * batch_size, i]
            if len(eval_q.shape) == 1:
                eval_q = eval_q.expand_dims(1)
            Q.append(eval_q)
        Q = self._concat(Q)
        return Q

    def Qpolicy(self, state, input_q):
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
            tp.assign_value((tau * ep.data + (1 - tau) * tp.data))
        for ep, tp in zip(self.qnetwork.trainable_params(), self.target_qnetwork.trainable_params()):
            tp.assign_value((tau * ep.data + (1 - tau) * tp.data))


class SPDQNPolicy(Module):
    def __init__(self,
                 observation_space,
                 action_space,
                 representation: ModuleType,
                 conactor_hidden_size: Sequence[int],
                 qnetwork_hidden_size: Sequence[int],
                 normalize: Optional[ModuleType] = None,
                 initialize: Optional[Callable[..., Tensor]] = None,
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
        self.target_conactor = deepcopy(self.conactor)
        self.target_qnetwork = deepcopy(self.qnetwork)

        self.offsets = self.conact_sizes.cumsum()
        self.offsets = np.insert(self.offsets, 0, 0)
        self.offsets = Tensor(self.offsets)

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
        input_q = Tensor(input_q, dtype=ms.float32)
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


class DRQNPolicy(Module):
    def __init__(self,
                 action_space: Discrete,
                 representation: Module,
                 **kwargs):
        super(DRQNPolicy, self).__init__()
        self.recurrent_layer_N = kwargs['recurrent_layer_N']
        self.rnn_hidden_dim = kwargs['recurrent_hidden_size']
        self.action_dim = action_space.n
        self.representation = representation
        self.target_representation = deepcopy(representation)
        self.representation_info_shape = self.representation.output_shapes
        kwargs["input_dim"] = self.representation.output_shapes['state'][0]
        kwargs["action_dim"] = self.action_dim
        self.lstm = True if kwargs["rnn"] == "LSTM" else False
        self.cnn = True if self.representation.cls_name == "Basic_CNN" else False
        self.eval_Qhead = BasicRecurrent(**kwargs)
        self.target_Qhead = deepcopy(self.eval_Qhead)
        self._zeroslike = ms.ops.ZerosLike()

    def construct(self, observation: Union[np.ndarray, dict], *rnn_hidden: Tensor):
        if self.cnn:
            obs_shape = observation.shape
            outputs = self.representation(observation.reshape((-1,) + obs_shape[-3:]))
            outputs['state'] = outputs['state'].reshape(obs_shape[0:-3] + (-1,))
        else:
            outputs = self.representation(observation)
        if self.lstm:
            hidden_states, cell_states, evalQ = self.eval_Qhead(outputs['state'], rnn_hidden[0], rnn_hidden[1])
        else:
            hidden_states, evalQ = self.eval_Qhead(outputs['state'], rnn_hidden[0])
            cell_states = None
        argmax_action = evalQ[:, -1].argmax(axis=-1)
        return outputs, argmax_action, evalQ, (hidden_states, cell_states)

    def target(self, observation: Union[np.ndarray, dict], *rnn_hidden: Tensor):
        if self.cnn:
            obs_shape = observation.shape
            outputs = self.representation(observation.reshape((-1,) + obs_shape[-3:]))
            outputs['state'] = outputs['state'].reshape(obs_shape[0:-3] + (-1,))
        else:
            outputs = self.representation(observation)
        if self.lstm:
            hidden_states, cell_states, targetQ = self.target_Qhead(outputs['state'], rnn_hidden[0], rnn_hidden[1])
        else:
            hidden_states, targetQ = self.target_Qhead(outputs['state'], rnn_hidden[0])
            cell_states = None
        argmax_action = targetQ.argmax(axis=-1)
        return outputs, argmax_action, targetQ, (hidden_states, cell_states)

    def init_hidden(self, batch):
        hidden_states = ms.ops.zeros(size=(self.recurrent_layer_N, batch, self.rnn_hidden_dim))
        cell_states = self._zeroslike(hidden_states) if self.lstm else None
        return hidden_states, cell_states

    def init_hidden_item(self, rnn_hidden, i):
        if self.lstm:
            rnn_hidden[0][:, i] = ms.ops.zeros(size=(self.recurrent_layer_N, self.rnn_hidden_dim))
            rnn_hidden[1][:, i] = ms.ops.zeros(size=(self.recurrent_layer_N, self.rnn_hidden_dim))
            return rnn_hidden
        else:
            rnn_hidden[:, i] = ms.ops.zeros(size=(self.recurrent_layer_N, self.rnn_hidden_dim))
            return rnn_hidden

    def copy_target(self):
        for ep, tp in zip(self.representation.trainable_params(), self.target_representation.trainable_params()):
            tp.assign_value(ep)
        for ep, tp in zip(self.eval_Qhead.trainable_params(), self.target_Qhead.trainable_params()):
            tp.assign_value(ep)
