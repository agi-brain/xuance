from xuance.tensorflow.policies import *
from xuance.tensorflow.utils import *
from xuance.tensorflow.representations import Basic_Identical


class BasicQhead(tk.Model):
    def __init__(self,
                 state_dim: int,
                 action_dim: int,
                 hidden_sizes: Sequence[int],
                 normalize: Optional[tk.layers.Layer] = None,
                 initializer: Optional[tk.initializers.Initializer] = None,
                 activation: Optional[tk.layers.Layer] = None,
                 device: str = "cpu:0"):
        super(BasicQhead, self).__init__()
        layers = []
        input_shape = (state_dim,)
        for h in hidden_sizes:
            mlp, input_shape = mlp_block(input_shape[0], h, normalize, activation, initializer, device)
            layers.extend(mlp)
        layers.extend(mlp_block(input_shape[0], action_dim, None, None, None, device)[0])
        self.model = tk.Sequential(layers)

    def call(self, inputs: tf.Tensor, **kwargs):
        return self.model(inputs)


class BasicRecurrent(tk.Model):
    def __init__(self, **kwargs):
        super(BasicRecurrent, self).__init__()
        self.lstm = False
        if kwargs["rnn"] == "GRU":
            output, _ = gru_block(kwargs["input_dim"],
                                  kwargs["recurrent_hidden_size"],
                                  kwargs["recurrent_layer_N"],
                                  kwargs["dropout"],
                                  kwargs["initialize"],
                                  kwargs["device"])
        elif kwargs["rnn"] == "LSTM":
            self.lstm = True
            output, _ = lstm_block(kwargs["input_dim"],
                                   kwargs["recurrent_hidden_size"],
                                   kwargs["recurrent_layer_N"],
                                   kwargs["dropout"],
                                   kwargs["initialize"],
                                   kwargs["device"])
        else:
            raise "Unknown recurrent module!"
        self.rnn_layer = output
        fc_layer = mlp_block(kwargs["recurrent_hidden_size"], kwargs["action_dim"], None, None, None, kwargs["device"])[0]
        self.model = tk.Sequential(*fc_layer)

    def call(self, x: tf.Tensor, **kwargs):
        if self.lstm:
            output, hn, cn = self.rnn_layer(x)
            return hn, cn, self.model(output)
        else:
            output, hn = self.rnn_layer(x)
            return hn, self.model(output)


class DuelQhead(tk.Model):
    def __init__(self,
                 state_dim: int,
                 action_dim: int,
                 hidden_sizes: Sequence[int],
                 normalize: Optional[tk.layers.Layer] = None,
                 initializer: Optional[tk.initializers.Initializer] = None,
                 activation: Optional[tk.layers.Layer] = None,
                 device: str = "cpu:0"):
        super(DuelQhead, self).__init__()
        v_layers = []
        input_shape = (state_dim,)
        for h in hidden_sizes:
            v_mlp, input_shape = mlp_block(input_shape[0], h // 2, normalize, activation, initializer, device)
            v_layers.extend(v_mlp)
        v_layers.extend(mlp_block(input_shape[0], 1, None, None, None, device)[0])
        a_layers = []
        input_shape = (state_dim,)
        for h in hidden_sizes:
            a_mlp, input_shape = mlp_block(input_shape[0], h // 2, normalize, activation, initializer, device)
            a_layers.extend(a_mlp)
        a_layers.extend(mlp_block(input_shape[0], action_dim, None, None, None, device)[0])
        self.a_model = tk.Sequential(a_layers)
        self.v_model = tk.Sequential(v_layers)

    def call(self, x: tf.Tensor, **kwargs):
        v = self.v_model(x)
        a = self.a_model(x)
        q = v + (a - tf.expand_dims(tf.reduce_mean(a, axis=-1), axis=-1))
        return q


class C51Qhead(tk.Model):
    def __init__(self,
                 state_dim: int,
                 action_dim: int,
                 atom_num: int,
                 hidden_sizes: Sequence[int],
                 normalize: Optional[tk.layers.Layer] = None,
                 initializer: Optional[tk.initializers.Initializer] = None,
                 activation: Optional[tk.layers.Layer] = None,
                 device: str = "cpu:0"):
        super(C51Qhead, self).__init__()
        self.action_dim = action_dim
        self.atom_num = atom_num
        layers = []
        input_shape = (state_dim,)
        for h in hidden_sizes:
            mlp, input_shape = mlp_block(input_shape[0], h, normalize, activation, initializer, device)
            layers.extend(mlp)
        layers.extend(mlp_block(input_shape[0], action_dim * atom_num, None, None, None, device)[0])
        self.model = tk.Sequential(layers)

    def call(self, x: tf.Tensor, **kwargs):
        dist_logits = tf.reshape(self.model(x), [-1, self.action_dim, self.atom_num])
        dist_probs = tf.nn.softmax(dist_logits, axis=-1)
        return dist_probs


class QRDQNhead(tk.Model):
    def __init__(self,
                 state_dim: int,
                 action_dim: int,
                 atom_num: int,
                 hidden_sizes: Sequence[int],
                 normalize: Optional[tk.layers.Layer] = None,
                 initializer: Optional[tk.initializers.Initializer] = None,
                 activation: Optional[tk.layers.Layer] = None,
                 device: str = "cpu:0"):
        super(QRDQNhead, self).__init__()
        self.action_dim = action_dim
        self.atom_num = atom_num
        layers = []
        input_shape = (state_dim,)
        for h in hidden_sizes:
            mlp, input_shape = mlp_block(input_shape[0], h, normalize, activation, initializer, device)
            layers.extend(mlp)
        layers.extend(mlp_block(input_shape[0], action_dim * atom_num, None, None, None, device)[0])
        self.model = tk.Sequential(layers)

    def call(self, x: tf.Tensor, **kwargs):
        quantiles = tf.reshape(self.model(x), [-1, self.action_dim, self.atom_num])
        return quantiles


class BasicQnetwork(tk.Model):
    def __init__(self,
                 action_space: Discrete,
                 representation: tk.Model,
                 hidden_size: Sequence[int] = None,
                 normalize: Optional[tk.layers.Layer] = None,
                 initializer: Optional[tk.initializers.Initializer] = None,
                 activation: Optional[tk.layers.Layer] = None,
                 device: str = "cpu:0"):
        super(BasicQnetwork, self).__init__()
        self.action_dim = action_space.n
        self.representation = representation
        self.target_representation = copy.deepcopy(representation)
        self.representation_info_shape = self.representation.output_shapes
        self.eval_Qhead = BasicQhead(self.representation.output_shapes['state'][0], self.action_dim, hidden_size,
                                     normalize, initializer, activation, device)
        self.target_Qhead = BasicQhead(self.representation.output_shapes['state'][0], self.action_dim, hidden_size,
                                       normalize, initializer, activation, device)
        self.target_Qhead.set_weights(self.eval_Qhead.get_weights())

    def call(self, observation: tf.Tensor, **kwargs):
        outputs = self.representation(observation)
        evalQ = self.eval_Qhead(outputs['state'])
        argmax_action = tf.math.argmax(evalQ, axis=-1)
        return outputs, argmax_action, evalQ

    def target(self, observation: Union[np.ndarray, dict]):
        outputs_target = self.target_representation(observation)
        targetQ = self.target_Qhead(outputs_target['state'])
        argmax_action = tf.math.argmax(targetQ, axis=-1)
        return outputs_target, tf.stop_gradient(argmax_action), tf.stop_gradient(targetQ)

    def copy_target(self):
        self.target_representation.set_weights(self.representation.get_weights())
        self.target_Qhead.set_weights(self.eval_Qhead.get_weights())


class DuelQnetwork(tk.Model):
    def __init__(self,
                 action_space: Space,
                 representation: Basic_Identical,
                 hidden_size: Sequence[int] = None,
                 normalize: Optional[tk.layers.Layer] = None,
                 initialize: Optional[Callable[..., tf.Tensor]] = None,
                 activation: Optional[tk.layers.Layer] = None,
                 device: str = "cpu:0"):
        super(DuelQnetwork, self).__init__()
        self.action_dim = action_space.n
        self.representation = representation
        self.target_representation = copy.deepcopy(representation)
        self.representation_info_shape = self.representation.output_shapes
        self.eval_Qhead = DuelQhead(self.representation.output_shapes['state'][0], self.action_dim, hidden_size,
                                    normalize, initialize, activation, device)
        self.target_Qhead = DuelQhead(self.representation.output_shapes['state'][0], self.action_dim, hidden_size,
                                      normalize, initialize, activation, device)
        self.target_Qhead.set_weights(self.eval_Qhead.get_weights())

    def call(self, observation: Union[np.ndarray, dict], **kwargs):
        outputs = self.representation(observation)
        evalQ = self.eval_Qhead(outputs['state'])
        argmax_action = tf.math.argmax(evalQ, axis=-1)
        return outputs, argmax_action, evalQ

    def target(self, observation: Union[np.ndarray, dict]):
        outputs = self.target_representation(observation)
        targetQ = self.target_Qhead(outputs['state'])
        argmax_action = tf.math.argmax(targetQ, axis=-1)
        return outputs, argmax_action, targetQ

    def copy_target(self):
        self.target_representation.set_weights(self.representation.get_weights())
        self.target_Qhead.set_weights(self.eval_Qhead.get_weights())


class NoisyQnetwork(tk.Model):
    def __init__(self,
                 action_space: Discrete,
                 representation: Basic_Identical,
                 hidden_size: Sequence[int] = None,
                 normalize: Optional[tk.layers.Layer] = None,
                 initialize: Optional[Callable[..., tf.Tensor]] = None,
                 activation: Optional[tk.layers.Layer] = None,
                 device: str = "cpu:0"):
        super(NoisyQnetwork, self).__init__()
        self.action_dim = action_space.n
        self.representation = representation
        self.target_representation = copy.deepcopy(representation)
        self.representation_info_shape = self.representation.output_shapes
        self.eval_Qhead = BasicQhead(self.representation.output_shapes['state'][0], self.action_dim, hidden_size,
                                     normalize, initialize, activation, device)
        self.target_Qhead = BasicQhead(self.representation.output_shapes['state'][0], self.action_dim, hidden_size,
                                       normalize, initialize, activation, device)
        self.target_Qhead.set_weights(self.eval_Qhead.get_weights())
        self.noise_scale = 0.0

    def update_noise(self, noisy_bound: float = 0.0):
        self.eval_noise_parameter = []
        self.target_noise_parameter = []
        for parameter in self.eval_Qhead.variables:
            self.eval_noise_parameter.append(
                tf.random.uniform(shape=parameter.shape, minval=-1.0, maxval=1.0) * noisy_bound)
            self.target_noise_parameter.append(
                tf.random.uniform(shape=parameter.shape, minval=-1.0, maxval=1.0) * noisy_bound)

    def call(self, observation: Union[np.ndarray, dict], **kwargs):
        outputs = self.representation(observation)
        self.update_noise(self.noise_scale)
        for parameter, noise_param in zip(self.eval_Qhead.variables, self.eval_noise_parameter):
            parameter.assign_add(noise_param)
        evalQ = self.eval_Qhead(outputs['state'])
        argmax_action = tf.math.argmax(evalQ, axis=-1)
        return outputs, argmax_action, evalQ

    def target(self, observation: Union[np.ndarray, dict]):
        outputs = self.target_representation(observation)
        self.update_noise(self.noise_scale)
        for parameter, noise_param in zip(self.target_Qhead.variables, self.target_noise_parameter):
            parameter.assign_add(noise_param)
        targetQ = self.target_Qhead(outputs['state'])
        argmax_action = tf.math.argmax(targetQ, axis=-1)
        return outputs, argmax_action, tf.stop_gradient(targetQ)

    def copy_target(self):
        self.target_representation.set_weights(self.representation.get_weights())
        self.target_Qhead.set_weights(self.eval_Qhead.get_weights())


class C51Qnetwork(tk.Model):
    def __init__(self,
                 action_space: Discrete,
                 atom_num: int,
                 vmin: float,
                 vmax: float,
                 representation: Basic_Identical,
                 hidden_size: Sequence[int] = None,
                 normalize: Optional[tk.layers.Layer] = None,
                 initialize: Optional[Callable[..., tf.Tensor]] = None,
                 activation: Optional[tk.layers.Layer] = None,
                 device: str = "cpu:0"):
        assert isinstance(action_space, Discrete)
        super(C51Qnetwork, self).__init__()
        self.action_dim = action_space.n
        self.atom_num = atom_num
        self.vmin = vmin
        self.vmax = vmax
        self.representation = representation
        self.target_representation = copy.deepcopy(representation)
        self.representation_info_shape = self.representation.output_shapes
        self.eval_Zhead = C51Qhead(self.representation.output_shapes['state'][0], self.action_dim, self.atom_num,
                                   hidden_size, normalize, initialize, activation, device)
        self.target_Zhead = C51Qhead(self.representation.output_shapes['state'][0], self.action_dim, self.atom_num,
                                     hidden_size, normalize, initialize, activation, device)
        self.target_Zhead.set_weights(self.eval_Zhead.get_weights())
        self.supports = tf.cast(tf.linspace(self.vmin, self.vmax, self.atom_num), dtype=tf.float32)
        self.deltaz = (vmax - vmin) / (atom_num - 1)

    def call(self, observation: Union[np.ndarray, dict], **kwargs):
        outputs = self.representation(observation)
        eval_Z = self.eval_Zhead(outputs['state'])
        eval_Q = tf.reduce_sum(self.supports * eval_Z, axis=-1)
        argmax_action = tf.math.argmax(eval_Q, axis=-1)
        return outputs, argmax_action, eval_Z

    def target(self, observation: Union[np.ndarray, dict]):
        outputs = self.target_representation(observation)
        target_Z = self.target_Zhead(outputs['state'])
        target_Q = tf.reduce_sum(self.supports * target_Z, axis=-1)
        argmax_action = tf.math.argmax(target_Q, axis=-1)
        return outputs, argmax_action, target_Z

    def copy_target(self):
        self.target_representation.set_weights(self.representation.get_weights())
        self.target_Zhead.set_weights(self.eval_Zhead.get_weights())


class QRDQN_Network(tk.Model):
    def __init__(self,
                 action_space: Discrete,
                 quantile_num: int,
                 representation: Basic_Identical,
                 hidden_size: Sequence[int] = None,
                 normalize: Optional[tk.layers.Layer] = None,
                 initialize: Optional[Callable[..., tf.Tensor]] = None,
                 activation: Optional[tk.layers.Layer] = None,
                 device: str = "cpu:0"):
        super(QRDQN_Network, self).__init__()
        self.action_dim = action_space.n
        self.quantile_num = quantile_num
        self.representation = representation
        self.target_representation = copy.deepcopy(representation)
        self.representation_info_shape = self.representation.output_shapes
        self.eval_Zhead = QRDQNhead(self.representation.output_shapes['state'][0], self.action_dim, self.quantile_num,
                                    hidden_size,
                                    normalize, initialize, activation, device)
        self.target_Zhead = QRDQNhead(self.representation.output_shapes['state'][0], self.action_dim, self.quantile_num,
                                      hidden_size,
                                      normalize, initialize, activation, device)
        self.target_Zhead.set_weights(self.eval_Zhead.get_weights())

    def call(self, observation: Union[np.ndarray, dict], **kwargs):
        outputs = self.representation(observation)
        eval_Z = self.eval_Zhead(outputs['state'])
        eval_Q = tf.reduce_mean(eval_Z, axis=-1)
        argmax_action = tf.math.argmax(eval_Q, axis=-1)
        return outputs, argmax_action, eval_Z

    def target(self, observation: Union[np.ndarray, dict]):
        outputs = self.target_representation(observation)
        target_Z = self.target_Zhead(outputs['state'])
        target_Q = tf.reduce_mean(target_Z, axis=-1)
        argmax_action = tf.math.argmax(target_Q, axis=-1)
        return outputs, argmax_action, target_Z

    def copy_target(self):
        self.target_representation.set_weights(self.representation.get_weights())
        self.target_Zhead.set_weights(self.eval_Zhead.get_weights())


class ActorNet(tk.Model):
    def __init__(self,
                 state_dim: int,
                 action_dim: int,
                 hidden_sizes: Sequence[int],
                 initialize: Optional[Callable[..., tf.Tensor]] = None,
                 activation: Optional[tk.layers.Layer] = None,
                 device: str = "cpu:0"):
        super(ActorNet, self).__init__()
        layers = []
        input_shape = (state_dim,)
        for h in hidden_sizes:
            mlp, input_shape = mlp_block(input_shape[0], h, None, activation, initialize, device)
            layers.extend(mlp)
        layers.extend(mlp_block(input_shape[0], action_dim, None, tk.layers.Activation('tanh'), initialize, device)[0])
        self.model = tk.Sequential(layers)

    def call(self, x: tf.Tensor, **kwargs):
        return self.model(x)


class CriticNet(tk.Model):
    def __init__(self,
                 state_dim: int,
                 action_dim: int,
                 hidden_sizes: Sequence[int],
                 initialize: Optional[Callable[..., tf.Tensor]] = None,
                 activation: Optional[tk.layers.Layer] = None,
                 device: str = "cpu:0"):
        super(CriticNet, self).__init__()
        layers = []
        input_shape = (state_dim + action_dim,)
        for h in hidden_sizes:
            mlp, input_shape = mlp_block(input_shape[0], h, None, activation, initialize, device)
            layers.extend(mlp)
        layers.extend(mlp_block(input_shape[0], 1, None, None, initialize, device)[0])
        self.model = tk.Sequential(layers)

    def call(self, inputs: Dict, **kwargs):
        x = inputs['x']
        a = inputs['a']
        return self.model(tf.concat((x, a), axis=-1))[:, 0]


class DDPGPolicy(tk.Model):
    def __init__(self,
                 action_space: Space,
                 representation: Basic_Identical,
                 actor_hidden_size: Sequence[int],
                 critic_hidden_size: Sequence[int],
                 initialize: Optional[Callable[..., tf.Tensor]] = None,
                 activation: Optional[tk.layers.Layer] = None,
                 device: str = "cpu:0"):
        super(DDPGPolicy, self).__init__()
        self.action_dim = action_space.shape[0]
        self.representation = representation
        self.representation_info_shape = self.representation.output_shapes

        self.actor = ActorNet(representation.output_shapes['state'][0], self.action_dim, actor_hidden_size, initialize,
                              activation, device)
        self.critic = CriticNet(representation.output_shapes['state'][0], self.action_dim, critic_hidden_size,
                                initialize, activation, device)
        self.target_actor = ActorNet(representation.output_shapes['state'][0], self.action_dim, actor_hidden_size,
                                     initialize,
                                     activation, device)
        self.target_critic = CriticNet(representation.output_shapes['state'][0], self.action_dim, critic_hidden_size,
                                       initialize, activation, device)
        self.soft_update(tau=1.0)

    def call(self, observation: Union[np.ndarray, dict], **kwargs):
        outputs = self.representation(observation)
        act = self.actor(outputs['state'])
        return outputs, act

    def Qtarget(self, observation: Union[np.ndarray, dict]):
        outputs = self.representation(observation)
        act = self.target_actor(outputs['state'])
        inputs_critic = {'x': outputs['state'], 'a': act}
        return self.target_critic(inputs_critic)

    def Qaction(self, observation: Union[np.ndarray, dict], action: tf.Tensor):
        outputs = self.representation(observation)
        inputs_critic = {'x': outputs['state'], 'a': action}
        return self.critic(inputs_critic)

    def Qpolicy(self, observation: Union[np.ndarray, dict]):
        outputs = self.representation(observation)
        action = self.actor(outputs['state'])
        inputs_critic = {'x': outputs['state'], 'a': action}
        return self.critic(inputs_critic)

    def soft_update(self, tau=0.005):
        for ep, tp in zip(self.actor.variables, self.target_actor.variables):
            tp.assign((1 - tau) * tp + tau * ep)
        for ep, tp in zip(self.critic.variables, self.target_critic.variables):
            tp.assign((1 - tau) * tp + tau * ep)


class TD3Policy(tk.Model):
    def __init__(self,
                 action_space: Space,
                 representation: Basic_Identical,
                 actor_hidden_size: Sequence[int],
                 critic_hidden_size: Sequence[int],
                 normalize: Optional[tk.layers.Layer] = None,
                 initialize: Optional[Callable[..., tf.Tensor]] = None,
                 activation: Optional[tk.layers.Layer] = None,
                 device: str = "cpu:0"):
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
        self.target_actor = ActorNet(representation.output_shapes['state'][0], self.action_dim, actor_hidden_size,
                                     initialize, activation, device)
        self.target_criticA = CriticNet(representation.output_shapes['state'][0], self.action_dim, critic_hidden_size,
                                        initialize, activation, device)
        self.target_criticB = CriticNet(representation.output_shapes['state'][0], self.action_dim, critic_hidden_size,
                                        initialize, activation, device)
        self.target_criticA.set_weights(self.criticA.get_weights())
        self.target_criticB.set_weights(self.criticB.get_weights())

    def call(self, observation: Union[np.ndarray, dict], **kwargs):
        outputs = self.representation(observation)
        act = self.actor(outputs['state'])
        return outputs, act

    def Qtarget(self, observation: Union[np.ndarray, dict]):
        outputs = self.representation(observation)
        act = self.target_actor(outputs['state'])
        noise = tf.random.uniform(act.shape, -1, 1) * 0.1
        act = tf.clip_by_value(act + noise, -1, 1)
        inputs_critic = {'x': outputs['state'], 'a': act}
        qa = tf.expand_dims(self.target_criticA(inputs_critic), axis=1)
        qb = tf.expand_dims(self.target_criticB(inputs_critic), axis=1)
        mim_q = tf.minimum(qa, qb)
        return outputs, mim_q

    def Qaction(self, observation: Union[np.ndarray, dict], action: tf.Tensor):
        outputs = self.representation(observation)
        inputs_critic = {'x': outputs['state'], 'a': action}
        qa = tf.expand_dims(self.criticA(inputs_critic), axis=1)
        qb = tf.expand_dims(self.criticB(inputs_critic), axis=1)
        return outputs, tf.concat((qa, qb), axis=-1)

    def Qpolicy(self, observation: Union[np.ndarray, dict]):
        outputs = self.representation(observation)
        act = self.actor(outputs['state'])
        inputs_critic = {'x': outputs['state'], 'a': act}
        qa = tf.expand_dims(self.criticA(inputs_critic), axis=1)
        qb = tf.expand_dims(self.criticB(inputs_critic), axis=1)
        return outputs, (qa + qb) / 2.0

    def soft_update(self, tau=0.005):
        for ep, tp in zip(self.actor.variables, self.target_actor.variables):
            tp.assign((1 - tau) * tp + tau * ep)
        for ep, tp in zip(self.criticA.variables, self.target_criticA.variables):
            tp.assign((1 - tau) * tp + tau * ep)
        for ep, tp in zip(self.criticB.variables, self.target_criticB.variables):
            tp.assign((1 - tau) * tp + tau * ep)


class PDQNPolicy(tk.Model):
    def __init__(self,
                 observation_space,
                 action_space,
                 representation: Basic_Identical,
                 conactor_hidden_size: Sequence[int],
                 qnetwork_hidden_size: Sequence[int],
                 normalize: Optional[tk.layers.Layer] = None,
                 initialize: Optional[Callable[..., tf.Tensor]] = None,
                 activation: Optional[tk.layers.Layer] = None,
                 device: str = "cpu:0"):
        super(PDQNPolicy, self).__init__()
        self.representation = representation
        self.target_representation = copy.deepcopy(representation)
        self.observation_space = observation_space
        self.action_space = action_space
        self.num_disact = self.action_space.spaces[0].n
        self.conact_sizes = np.array([self.action_space.spaces[i].shape[0] for i in range(1, self.num_disact + 1)])
        self.conact_size = int(self.conact_sizes.sum())

        self.qnetwork = BasicQhead(self.observation_space.shape[0] + self.conact_size, self.num_disact,
                                   qnetwork_hidden_size, normalize,
                                   initialize, activation, device)
        self.dim_input = self.observation_space.shape[0] + self.conact_size
        self.qnetwork._set_inputs(tf.TensorSpec([None, self.dim_input], tf.float32, name='inputs'))
        self.conactor = ActorNet(self.observation_space.shape[0], self.conact_size, conactor_hidden_size,
                                 initialize, activation, device)
        self.target_qnetwork = BasicQhead(self.observation_space.shape[0] + self.conact_size, self.num_disact,
                                          qnetwork_hidden_size, normalize, initialize, activation, device)
        self.target_conactor = ActorNet(self.observation_space.shape[0], self.conact_size, conactor_hidden_size,
                                        initialize, activation, device)
        self.target_conactor.set_weights(self.conactor.get_weights())
        self.target_qnetwork.set_weights(self.qnetwork.get_weights())

    def Atarget(self, state):
        target_conact = self.target_conactor(state)
        return target_conact

    def con_action(self, state):
        conaction = self.conactor(state)
        return conaction

    def Qtarget(self, state, action):
        input_q = tf.concat((state, action), axis=1)
        target_q = self.target_qnetwork(input_q)
        return target_q

    def Qeval(self, state, action):
        input_q = tf.concat((state, action), axis=1)
        eval_q = self.qnetwork(input_q)
        return eval_q

    def Qpolicy(self, state):
        conact = self.conactor(state)
        input_q = tf.concat((state, conact), axis=1)
        policy_q = tf.reduce_sum(self.qnetwork(input_q))
        return policy_q

    def soft_update(self, tau=0.005):
        # for ep, tp in zip(self.representation.variables, self.target_representation.variables):
        #     tp.assign((1 - tau) * tp + tau * ep)
        for ep, tp in zip(self.conactor.variables, self.target_conactor.variables):
            tp.assign((1 - tau) * tp + tau * ep)
        for ep, tp in zip(self.qnetwork.variables, self.target_qnetwork.variables):
            tp.assign((1 - tau) * tp + tau * ep)


class MPDQNPolicy(tk.Model):
    def __init__(self,
                 observation_space,
                 action_space,
                 representation: Basic_Identical,
                 conactor_hidden_size: Sequence[int],
                 qnetwork_hidden_size: Sequence[int],
                 normalize: Optional[tk.layers.Layer] = None,
                 initialize: Optional[Callable[..., tf.Tensor]] = None,
                 activation: Optional[tk.layers.Layer] = None,
                 device: str = "cpu:0"):
        super(MPDQNPolicy, self).__init__()
        self.representation = representation
        self.target_representation = copy.deepcopy(representation)
        self.observation_space = observation_space
        self.obs_size = self.observation_space.shape[0]
        self.action_space = action_space
        self.num_disact = self.action_space.spaces[0].n
        self.conact_sizes = np.array([self.action_space.spaces[i].shape[0] for i in range(1, self.num_disact + 1)])
        self.conact_size = int(self.conact_sizes.sum())

        self.qnetwork = BasicQhead(self.observation_space.shape[0] + self.conact_size, self.num_disact,
                                   qnetwork_hidden_size, normalize,
                                   initialize, activation, device)
        self.dim_input = self.observation_space.shape[0] + self.conact_size
        self.qnetwork._set_inputs(tf.TensorSpec([None, self.dim_input], tf.float32, name='inputs'))
        self.conactor = ActorNet(self.observation_space.shape[0], self.conact_size, conactor_hidden_size,
                                 initialize, activation, device)

        self.target_qnetwork = BasicQhead(self.observation_space.shape[0] + self.conact_size, self.num_disact,
                                          qnetwork_hidden_size, normalize,
                                          initialize, activation, device)
        self.target_conactor = ActorNet(self.observation_space.shape[0], self.conact_size, conactor_hidden_size,
                                        initialize, activation, device)
        self.offsets = self.conact_sizes.cumsum()
        self.offsets = np.insert(self.offsets, 0, 0)
        self.soft_update(tau=1.0)

    def Atarget(self, state):
        target_conact = self.target_conactor(state)
        return target_conact

    def con_action(self, state):
        conaction = self.conactor(state)
        return conaction

    def Qtarget(self, state, action):
        batch_size = state.shape[0]
        Q = []
        input_q = tf.concat((state, tf.zeros_like(action)), axis=1)
        input_q = tf.tile(input_q, (self.num_disact, 1)).numpy()
        for i in range(self.num_disact):
            input_q[i * batch_size:(i + 1) * batch_size,
            self.obs_size + self.offsets[i]: self.obs_size + self.offsets[i + 1]] \
                = action[:, self.offsets[i]:self.offsets[i + 1]]
        eval_qall = self.target_qnetwork(input_q)
        for i in range(self.num_disact):
            eval_q = eval_qall[i * batch_size:(i + 1) * batch_size, i]
            if len(eval_q.shape) == 1:
                eval_q = tf.expand_dims(eval_q, 1)
            Q.append(eval_q)
        Q = tf.concat(Q, axis=1)
        return Q

    def Qeval(self, state, action):
        batch_size = state.shape[0]
        Q = []
        input_q = tf.concat((state, tf.zeros_like(action)), axis=1)
        input_q = tf.tile(input_q, (self.num_disact, 1)).numpy()
        for i in range(self.num_disact):
            input_q[i * batch_size:(i + 1) * batch_size,
            self.obs_size + self.offsets[i]: self.obs_size + self.offsets[i + 1]] \
                = action[:, self.offsets[i]:self.offsets[i + 1]]
        eval_qall = self.qnetwork(input_q)
        for i in range(self.num_disact):
            eval_q = eval_qall[i * batch_size:(i + 1) * batch_size, i]
            if len(eval_q.shape) == 1:
                eval_q = tf.expand_dims(eval_q, axis=1)
            Q.append(eval_q)
        Q = tf.concat(Q, axis=1)
        return Q

    def Qpolicy(self, state):
        conact = self.conactor(state)
        batch_size = state.shape[0]
        Q = []
        input_q = tf.concat((state, tf.zeros_like(conact)), axis=1)
        input_q = tf.tile(input_q, (self.num_disact, 1)).numpy()
        for i in range(self.num_disact):
            input_q[i * batch_size:(i + 1) * batch_size,
            self.obs_size + self.offsets[i]: self.obs_size + self.offsets[i + 1]] \
                = conact[:, self.offsets[i]:self.offsets[i + 1]]
        eval_qall = self.qnetwork(input_q)
        for i in range(self.num_disact):
            eval_q = eval_qall[i * batch_size:(i + 1) * batch_size, i]
            if len(eval_q.shape) == 1:
                eval_q = tf.expand_dims(eval_q, axis=1)
            Q.append(eval_q)
        Q = tf.concat(Q, axis=1)
        return Q

    def soft_update(self, tau=0.005):
        # for ep, tp in zip(self.representation.variables, self.target_representation.variables):
        #     tp.assign((1 - tau) * tp + tau * ep)
        for ep, tp in zip(self.conactor.variables, self.target_conactor.variables):
            tp.assign((1 - tau) * tp + tau * ep)
        for ep, tp in zip(self.qnetwork.variables, self.target_qnetwork.variables):
            tp.assign((1 - tau) * tp + tau * ep)


class SPDQNPolicy(tk.Model):
    def __init__(self,
                 observation_space,
                 action_space,
                 representation: Basic_Identical,
                 conactor_hidden_size: Sequence[int],
                 qnetwork_hidden_size: Sequence[int],
                 normalize: Optional[tk.layers.Layer] = None,
                 initialize: Optional[Callable[..., tf.Tensor]] = None,
                 activation: Optional[tk.layers.Layer] = None,
                 device: str = "cpu:0"):
        super(SPDQNPolicy, self).__init__()
        self.representation = representation
        self.target_representation = copy.deepcopy(representation)
        self.observation_space = observation_space
        self.action_space = action_space
        self.num_disact = self.action_space.spaces[0].n
        self.conact_sizes = np.array([self.action_space.spaces[i].shape[0] for i in range(1, self.num_disact + 1)])
        self.conact_size = int(self.conact_sizes.sum())
        self.qnetwork, self.target_qnetwork = [], []
        for k in range(self.num_disact):
            self.qnetwork.append(
                BasicQhead(self.observation_space.shape[0] + self.conact_sizes[k], 1, qnetwork_hidden_size, normalize,
                           initialize, activation, device))
            dim_input = self.observation_space.shape[0] + self.conact_sizes[k]
            self.qnetwork[k]._set_inputs(tf.TensorSpec([None, dim_input], tf.float32, name='inputs_%d'%(k)))

        self.conactor = ActorNet(self.observation_space.shape[0], self.conact_size, conactor_hidden_size,
                                 initialize, activation, device)
        for k in range(self.num_disact):
            self.target_qnetwork.append(
                BasicQhead(self.observation_space.shape[0] + self.conact_sizes[k], 1, qnetwork_hidden_size, normalize,
                           initialize, activation, device))
        self.target_conactor = ActorNet(self.observation_space.shape[0], self.conact_size, conactor_hidden_size,
                                        initialize, activation, device)

        self.offsets = self.conact_sizes.cumsum()
        self.offsets = np.insert(self.offsets, 0, 0)
        self.soft_update(tau=1.0)

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
            input_q = tf.concat((state, conact), axis=1)
            eval_q = self.target_qnetwork[i](input_q)
            target_Q.append(eval_q)
        target_Q = tf.concat(target_Q, axis=1)
        return target_Q

    def Qeval(self, state, action):
        Q = []
        for i in range(self.num_disact):
            conact = action[:, self.offsets[i]:self.offsets[i + 1]]
            input_q = tf.concat((state, conact), axis=1)
            eval_q = self.qnetwork[i](input_q)
            Q.append(eval_q)
        Q = tf.concat(Q, axis=1)
        return Q

    def Qpolicy(self, state):
        conacts = self.conactor(state)
        Q = []
        for i in range(self.num_disact):
            conact = conacts[:, self.offsets[i]:self.offsets[i + 1]]
            input_q = tf.concat((state, conact), axis=1)
            eval_q = self.qnetwork[i](input_q)
            Q.append(eval_q)
        Q = tf.concat(Q, axis=1)
        return Q

    def soft_update(self, tau=0.005):
        for ep, tp in zip(self.conactor.variables, self.target_conactor.variables):
            tp.assign((1 - tau) * tp + tau * ep)
        for qnet, target_qnet in zip(self.qnetwork, self.target_qnetwork):
            for ep, tp in zip(qnet.variables, target_qnet.variables):
                tp.assign((1 - tau) * tp + tau * ep)


class DRQNPolicy(tk.Model):
    def __init__(self,
                 action_space: Discrete,
                 representation: tk.Model,
                 **kwargs):
        super(DRQNPolicy, self).__init__()
        self.device = kwargs['device']
        self.recurrent_layer_N = kwargs['recurrent_layer_N']
        self.rnn_hidden_dim = kwargs['recurrent_hidden_size']
        self.action_dim = action_space.n
        self.representation = representation
        self.target_representation = copy.deepcopy(representation)
        self.representation_info_shape = self.representation.output_shapes
        kwargs["input_dim"] = self.representation.output_shapes['state'][0]
        kwargs["action_dim"] = self.action_dim
        self.lstm = True if kwargs["rnn"] == "LSTM" else False
        self.cnn = True if self.representation.name == "basic_cnn" else False
        self.eval_Qhead = BasicRecurrent(**kwargs)
        self.target_Qhead = BasicRecurrent(**kwargs)
        self.target_Qhead.set_weights(self.eval_Qhead.get_weights())

    def call(self, observation: Union[np.ndarray, dict], *rnn_hidden: tf.Tensor, **kwargs):
        if self.cnn:
            obs_shape = observation.shape
            outputs = self.representation(observation.reshape((-1,) + obs_shape[-3:]))
            output_states = outputs['state'].reshape(obs_shape[0:-3] + (-1,))
        else:
            obs_shape = observation.shape
            observations_in = tf.reshape(observation, [-1, obs_shape[-1]])
            outputs = self.representation(observations_in)
            output_states = tf.reshape(outputs['state'], obs_shape[:-1] + self.representation.output_shapes['state'])
        if self.lstm:
            hidden_states, cell_states, evalQ = self.eval_Qhead(output_states)
        else:
            hidden_states, evalQ = self.eval_Qhead(output_states)
            cell_states = None
        argmax_action = tf.math.argmax(evalQ[:, -1], axis=-1)
        return outputs, argmax_action, evalQ, (hidden_states, cell_states)

    def target(self, observation: Union[np.ndarray, dict], *rnn_hidden: tf.Tensor):
        if self.cnn:
            obs_shape = observation.shape
            outputs = self.target_representation(observation.reshape((-1,) + obs_shape[-3:]))
            output_states = outputs['state'].reshape(obs_shape[0:-3] + (-1,))
        else:
            obs_shape = observation.shape
            observations_in = tf.reshape(observation, [-1, obs_shape[-1]])
            outputs = self.target_representation(observations_in)
            output_states = tf.reshape(outputs['state'], obs_shape[:-1] + self.representation.output_shapes['state'])
        if self.lstm:
            hidden_states, cell_states, targetQ = self.target_Qhead(output_states)
        else:
            hidden_states, targetQ = self.target_Qhead(output_states)
            cell_states = None
        argmax_action = tf.math.argmax(targetQ, axis=-1)
        return outputs, argmax_action, targetQ, (hidden_states, cell_states)

    def init_hidden(self, batch):
        with tf.device(self.device):
            hidden_states = tf.zeros(shape=(self.recurrent_layer_N, batch, self.rnn_hidden_dim))
            cell_states = tf.zeros_like(hidden_states) if self.lstm else None
            return hidden_states, cell_states

    def init_hidden_item(self, rnn_hidden, i):
        with tf.device(self.device):
            if self.lstm:
                rnn_hidden_0, rnn_hidden_1 = rnn_hidden[0].numpy(), rnn_hidden[1].numpy()
                rnn_hidden_0[i:i+1] = np.zeros((self.recurrent_layer_N, self.rnn_hidden_dim))
                rnn_hidden_1[i:i+1] = np.zeros((self.recurrent_layer_N, self.rnn_hidden_dim))
                return (tf.convert_to_tensor(rnn_hidden_0), tf.convert_to_tensor(rnn_hidden_1))
            else:
                rnn_hidden_np = rnn_hidden.numpy()
                rnn_hidden_np[i:i+1] = np.zeros((self.recurrent_layer_N, self.rnn_hidden_dim))
                return tf.convert_to_tensor(rnn_hidden_np)

    def copy_target(self):
        self.target_representation.set_weights(self.representation.get_weights())
        self.target_Qhead.set_weights(self.eval_Qhead.get_weights())

