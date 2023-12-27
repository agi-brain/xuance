from xuance.tensorflow.policies import *
from xuance.tensorflow.utils import *
from xuance.tensorflow.representations import Basic_Identical


class BasicQhead(tk.Model):
    def __init__(self,
                 state_dim: int,
                 action_dim: int,
                 n_agents: int,
                 hidden_sizes: Sequence[int],
                 normalize: Optional[tk.layers.Layer] = None,
                 initializer: Optional[tk.initializers.Initializer] = None,
                 activation: Optional[tk.layers.Layer] = None,
                 device: str = "cpu:0"):
        super(BasicQhead, self).__init__()
        layers_ = []
        input_shape = (state_dim + n_agents,)
        for h in hidden_sizes:
            mlp, input_shape = mlp_block(input_shape[0], h, normalize, activation, initializer, device)
            layers_.extend(mlp)
        layers_.extend(mlp_block(input_shape[0], action_dim, None, None, None, device)[0])
        self.model = tk.Sequential(layers_)

    def call(self, x: tf.Tensor, **kwargs):
        return self.model(x)


class BasicQnetwork(tk.Model):
    def __init__(self,
                 action_space: Discrete,
                 n_agents: int,
                 representation: Optional[Basic_Identical],
                 hidden_size: Sequence[int] = None,
                 normalize: Optional[tk.layers.Layer] = None,
                 initializer: Optional[tk.initializers.Initializer] = None,
                 activation: Optional[tk.layers.Layer] = None,
                 device: str = "cpu:0",
                 **kwargs):
        super(BasicQnetwork, self).__init__()
        self.action_dim = action_space.n
        self.representation = representation
        self.target_representation = copy.deepcopy(self.representation)
        self.representation_info_shape = self.representation.output_shapes
        self.obs_dim = self.representation.input_shapes[0]
        self.n_agents = n_agents
        self.lstm = True if kwargs["rnn"] == "LSTM" else False
        self.use_rnn = True if kwargs["use_recurrent"] else False
        self.eval_Qhead = BasicQhead(self.representation.output_shapes['state'][0], self.action_dim, n_agents,
                                     hidden_size, normalize, initializer, activation, device)
        self.target_Qhead = BasicQhead(self.representation.output_shapes['state'][0], self.action_dim, n_agents,
                                       hidden_size, normalize, initializer, activation, device)
        self.target_Qhead.set_weights(self.eval_Qhead.get_weights())

    def call(self, inputs: Union[np.ndarray, dict], *rnn_hidden, **kwargs):
        observations = tf.reshape(inputs['obs'], [-1, self.obs_dim])
        IDs = tf.reshape(inputs['ids'], [-1, self.n_agents])
        if self.use_rnn:
            outputs = self.representation(observations, *rnn_hidden)
            rnn_hidden = (outputs['rnn_hidden'], outputs['rnn_cell'])
        else:
            outputs = self.representation(observations)
            rnn_hidden = None
        q_inputs = tf.concat([outputs['state'], IDs], axis=-1)
        evalQ = tf.reshape(self.eval_Qhead(q_inputs), [-1, self.n_agents, self.action_dim])
        if ('avail_actions' in kwargs.keys()) and (kwargs['avail_actions'] is not None):
            evalQ_detach = evalQ.clone().detach()
            avail_actions = kwargs['avail_actions']
            evalQ_detach[avail_actions == 0] = -9999999
            argmax_action = evalQ_detach.argmax(dim=-1, keepdim=False)
        else:
            argmax_action = tf.argmax(evalQ, axis=-1)
        return rnn_hidden, argmax_action, evalQ

    def target_Q(self, inputs: Union[np.ndarray, dict]):
        shape_obs = inputs["obs"].shape
        shape_ids = inputs["ids"].shape
        observations = tf.reshape(inputs['obs'], [-1, shape_obs[-1]])
        IDs = tf.reshape(inputs['ids'], [-1, shape_ids[-1]])
        outputs = self.target_representation(observations)
        q_inputs = tf.concat([outputs['state'], IDs], axis=-1)
        return tf.reshape(self.target_Qhead(q_inputs), shape_obs[0:-1] + (self.action_dim,))

    def trainable_param(self):
        return self.representation.trainable_variables + self.eval_Qhead.trainable_variables

    def copy_target(self):
        self.target_representation.set_weights(self.representation.get_weights())
        self.target_Qhead.set_weights(self.eval_Qhead.get_weights())


class MFQnetwork(tk.Model):
    def __init__(self,
                 action_space: Discrete,
                 n_agents: int,
                 representation: Optional[Basic_Identical],
                 hidden_size: Sequence[int] = None,
                 normalize: Optional[tk.layers.Layer] = None,
                 initializer: Optional[tk.initializers.Initializer] = None,
                 activation: Optional[tk.layers.Layer] = None,
                 device: str = "cpu:0"):
        super(MFQnetwork, self).__init__()
        self.action_dim = action_space.n
        self.representation = representation
        self.target_representation = copy.deepcopy(self.representation)
        self.representation_info_shape = self.representation.output_shapes

        self.eval_Qhead = BasicQhead(self.representation.output_shapes['state'][0] + self.action_dim, self.action_dim,
                                     n_agents, hidden_size, normalize, initializer, activation, device)
        self.target_Qhead = BasicQhead(self.representation.output_shapes['state'][0] + self.action_dim, self.action_dim,
                                       n_agents, hidden_size, normalize, initializer, activation, device)
        self.target_Qhead.set_weights(self.eval_Qhead.get_weights())

    def call(self, inputs: Union[np.ndarray, dict], **kwargs):
        observation = inputs["obs"]
        actions_mean = inputs["act_mean"]
        agent_ids = inputs["ids"]
        outputs = self.representation(observation)
        q_inputs = tf.concat([outputs['state'], actions_mean, agent_ids], axis=-1)
        evalQ = self.eval_Qhead(q_inputs)
        argmax_action = tf.argmax(evalQ, axis=-1)
        return outputs, argmax_action, evalQ

    def sample_actions(self, logits: tf.Tensor):
        dist = tfp.distributions.Categorical(logits=logits)
        return dist.sample()

    def target_Q(self, observation: tf.Tensor, actions_mean: tf.Tensor, agent_ids: tf.Tensor):
        outputs = self.target_representation(observation)
        q_inputs = tf.concat([outputs['state'], actions_mean, agent_ids], axis=-1)
        return self.target_Qhead(q_inputs)

    def copy_target(self):
        self.target_representation.set_weights(self.representation.get_weights())
        self.target_Qhead.set_weights(self.eval_Qhead.get_weights())


class MixingQnetwork(tk.Model):
    def __init__(self,
                 action_space: Discrete,
                 n_agents: int,
                 representation: Optional[Basic_Identical],
                 mixer: Optional[VDN_mixer] = None,
                 hidden_size: Sequence[int] = None,
                 normalize: Optional[tk.layers.Layer] = None,
                 initializer: Optional[tk.initializers.Initializer] = None,
                 activation: Optional[tk.layers.Layer] = None,
                 device: str = "cpu:0",
                 **kwargs):
        super(MixingQnetwork, self).__init__()
        self.action_dim = action_space.n
        self.representation = representation
        self.target_representation = copy.deepcopy(self.representation)
        self.representation_info_shape = self.representation.output_shapes
        self.obs_dim = self.representation.input_shapes[0]
        self.n_agents = n_agents
        self.lstm = True if kwargs["rnn"] == "LSTM" else False
        self.use_rnn = True if kwargs["use_recurrent"] else False
        self.eval_Qhead = BasicQhead(self.representation.output_shapes['state'][0], self.action_dim, n_agents,
                                     hidden_size, normalize, initializer, activation, device)
        self.target_Qhead = BasicQhead(self.representation.output_shapes['state'][0], self.action_dim, n_agents,
                                       hidden_size, normalize, initializer, activation, device)
        self.eval_Qtot = mixer
        self.target_Qtot = mixer
        # self.copy_target()
        self.target_Qhead.set_weights(self.eval_Qhead.get_weights())
        self.target_Qtot.set_weights(self.eval_Qtot.get_weights())

    def call(self, inputs: Union[np.ndarray, dict], *rnn_hidden, **kwargs):
        observations = tf.reshape(inputs['obs'], [-1, self.obs_dim])
        IDs = tf.reshape(inputs['ids'], [-1, self.n_agents])
        if self.use_rnn:
            outputs = self.representation(observations, *rnn_hidden)
            rnn_hidden = (outputs['rnn_hidden'], outputs['rnn_cell'])
        else:
            outputs = self.representation(observations)
            rnn_hidden = None
        q_inputs = tf.concat([outputs['state'], IDs], axis=-1)
        evalQ = tf.reshape(self.eval_Qhead(q_inputs), [-1, self.n_agents, self.action_dim])
        if ('avail_actions' in kwargs.keys()) and (kwargs['avail_actions'] is not None):
            evalQ_detach = evalQ.clone().detach()
            avail_actions = kwargs['avail_actions']
            evalQ_detach[avail_actions == 0] = -9999999
            argmax_action = evalQ_detach.argmax(dim=-1, keepdim=False)
        else:
            argmax_action = tf.argmax(evalQ, axis=-1)
        return rnn_hidden, argmax_action, evalQ

    def target_Q(self, inputs: Union[np.ndarray, dict]):
        shape_obs = inputs["obs"].shape
        shape_ids = inputs["ids"].shape
        observations = tf.reshape(inputs['obs'], [-1, shape_obs[-1]])
        IDs = tf.reshape(inputs['ids'], [-1, shape_ids[-1]])
        outputs = self.target_representation(observations)
        q_inputs = tf.concat([outputs['state'], IDs], axis=-1)
        return tf.reshape(self.target_Qhead(q_inputs), shape_obs[0:-1] + (self.action_dim,))

    def Q_tot(self, q, states=None):
        return self.eval_Qtot(q, states)

    def target_Q_tot(self, q, states=None):
        return self.target_Qtot(q, states)

    def copy_target(self):
        self.target_representation.set_weights(self.representation.get_weights())
        self.target_Qhead.set_weights(self.eval_Qhead.get_weights())
        self.target_Qtot.set_weights(self.eval_Qtot.get_weights())


class Weighted_MixingQnetwork(MixingQnetwork):
    def __init__(self,
                 action_space: Discrete,
                 n_agents: int,
                 representation: Optional[Basic_Identical],
                 mixer: Optional[VDN_mixer] = None,
                 ff_mixer: Optional[QMIX_FF_mixer] = None,
                 hidden_size: Sequence[int] = None,
                 normalize: Optional[tk.layers.Layer] = None,
                 initializer: Optional[tk.initializers.Initializer] = None,
                 activation: Optional[tk.layers.Layer] = None,
                 device: str = "cpu:0",
                 **kwargs):
        super(Weighted_MixingQnetwork, self).__init__(action_space, n_agents, representation, mixer, hidden_size,
                                                      normalize, initializer, activation, device, **kwargs)
        self.eval_Qhead_centralized = BasicQhead(self.representation.output_shapes['state'][0], self.action_dim,
                                                 n_agents, hidden_size, normalize, initializer, activation, device)
        self.target_Qhead_centralized = BasicQhead(self.representation.output_shapes['state'][0], self.action_dim,
                                                   n_agents, hidden_size, normalize, initializer, activation, device)
        self.q_feedforward = ff_mixer
        self.target_q_feedforward = ff_mixer
        self.target_Qhead.set_weights(self.eval_Qhead.get_weights())
        self.target_Qtot.set_weights(self.eval_Qtot.get_weights())
        self.target_Qhead_centralized.set_weights(self.eval_Qhead_centralized.get_weights())
        self.target_q_feedforward.set_weights(self.q_feedforward.get_weights())

    def q_centralized(self, inputs: Union[np.ndarray, dict]):
        observations = tf.reshape(inputs['obs'], [-1, self.obs_dim])
        IDs = tf.reshape(inputs['ids'], [-1, self.n_agents])
        outputs = self.representation(observations)
        q_inputs = tf.concat([outputs['state'], IDs], axis=-1)
        return tf.reshape(self.eval_Qhead_centralized(q_inputs), [-1, self.n_agents, self.action_dim])

    def target_q_centralized(self, inputs: Union[np.ndarray, dict]):
        observations = tf.reshape(inputs['obs'], [-1, self.obs_dim])
        IDs = tf.reshape(inputs['ids'], [-1, self.n_agents])
        outputs = self.target_representation(observations)
        q_inputs = tf.concat([outputs['state'], IDs], axis=-1)
        return tf.reshape(self.target_Qhead_centralized(q_inputs), [-1, self.n_agents, self.action_dim])

    def copy_target(self):
        self.target_representation.set_weights(self.representation.get_weights())
        self.target_Qhead.set_weights(self.eval_Qhead.get_weights())
        self.target_Qtot.set_weights(self.eval_Qtot.get_weights())
        self.target_Qhead_centralized.set_weights(self.eval_Qhead_centralized.get_weights())
        self.target_q_feedforward.set_weights(self.q_feedforward.get_weights())


class Qtran_MixingQnetwork(tk.Model):
    def __init__(self,
                 action_space: Discrete,
                 n_agents: int,
                 representation: Optional[Basic_Identical],
                 mixer: Optional[VDN_mixer] = None,
                 qtran_mixer: Optional[QTRAN_base] = None,
                 hidden_size: Sequence[int] = None,
                 normalize: Optional[tk.layers.Layer] = None,
                 initializer: Optional[tk.initializers.Initializer] = None,
                 activation: Optional[tk.layers.Layer] = None,
                 device: str = "cpu:0",
                 **kwargs):
        super(Qtran_MixingQnetwork, self).__init__()
        self.action_dim = action_space.n
        self.representation = representation
        self.target_representation = copy.deepcopy(self.representation)
        self.representation_info_shape = self.representation.output_shapes
        self.obs_dim = self.representation.input_shapes[0]
        self.hidden_state_dim = self.representation.output_shapes['state'][0]
        self.n_agents = n_agents
        self.lstm = True if kwargs["rnn"] == "LSTM" else False
        self.use_rnn = True if kwargs["use_recurrent"] else False
        self.eval_Qhead = BasicQhead(self.representation.output_shapes['state'][0], self.action_dim, n_agents,
                                     hidden_size, normalize, initializer, activation, device)
        self.target_Qhead = BasicQhead(self.representation.output_shapes['state'][0], self.action_dim, n_agents,
                                       hidden_size, normalize, initializer, activation, device)
        self.qtran_net = qtran_mixer
        self.target_qtran_net = qtran_mixer
        self.q_tot = mixer
        self.target_Qhead.set_weights(self.eval_Qhead.get_weights())
        self.target_qtran_net.set_weights(self.qtran_net.get_weights())

    def call(self, inputs: Union[np.ndarray, dict], *rnn_hidden, **kwargs):
        observations = tf.reshape(inputs['obs'], [-1, self.obs_dim])
        IDs = tf.reshape(inputs['ids'], [-1, self.n_agents])
        outputs = self.representation(observations)
        q_inputs = tf.concat([outputs['state'], IDs], axis=-1)
        evalQ = tf.reshape(self.eval_Qhead(q_inputs), [-1, self.n_agents, self.action_dim])
        argmax_action = tf.argmax(evalQ, axis=-1)
        return tf.reshape(outputs['state'], [-1, self.n_agents, self.hidden_state_dim]), argmax_action, evalQ

    def target_Q(self, inputs: Union[np.ndarray, dict]):
        observations = tf.reshape(inputs['obs'], [-1, self.obs_dim])
        IDs = tf.reshape(inputs['ids'], [-1, self.n_agents])
        outputs = self.target_representation(observations)
        q_inputs = tf.concat([outputs['state'], IDs], axis=-1)
        return tf.reshape(outputs['state'], [-1, self.n_agents, self.hidden_state_dim]), self.target_Qhead(q_inputs)

    def copy_target(self):
        self.target_representation.set_weights(self.representation.get_weights())
        self.target_Qhead.set_weights(self.eval_Qhead.get_weights())
        self.target_qtran_net.set_weights(self.qtran_net.get_weights())


class DCG_policy(tk.Model):
    def __init__(self,
                 action_space: Discrete,
                 global_state_dim: int,
                 representation: Optional[Basic_Identical],
                 utility: Optional[tk.Model] = None,
                 payoffs: Optional[tk.Model] = None,
                 dcgraph: Optional[tk.Model] = None,
                 hidden_size_bias: Sequence[int] = None,
                 normalize: Optional[tk.layers.Layer] = None,
                 initializer: Optional[tk.initializers.Initializer] = None,
                 activation: Optional[tk.layers.Layer] = None,
                 device: str = "cpu:0",
                 **kwargs):
        super(DCG_policy, self).__init__()
        self.action_dim = action_space.n
        self.representation = representation
        self.target_representation = representation
        self.lstm = True if kwargs["rnn"] == "LSTM" else False
        self.use_rnn = True if kwargs["use_recurrent"] else False
        self.utility = utility
        self.target_utility = utility
        self.payoffs = payoffs
        self.target_payoffs = payoffs
        self.graph = dcgraph
        self.dcg_s = False
        if hidden_size_bias is not None:
            self.dcg_s = True
            self.bias = BasicQhead(global_state_dim, 1, 0, hidden_size_bias,
                                   normalize, initializer, activation, device)
            self.target_bias = BasicQhead(global_state_dim, 1, 0, hidden_size_bias,
                                          normalize, initializer, activation, device)

    def call(self, inputs: Union[np.ndarray, dict], *rnn_hidden: tf.Tensor, **kwargs):
        observations = tf.reshape(inputs['obs'], [-1, self.obs_dim])
        IDs = tf.reshape(inputs['ids'], [-1, self.n_agents])
        outputs = self.representation(observations)
        q_inputs = tf.concat([outputs['state'], IDs], axis=-1)
        evalQ = self.eval_Qhead(q_inputs)
        evalQ = tf.reshape(evalQ, [-1, self.n_agents, self.action_dim])
        argmax_action = tf.argmax(evalQ, axis=-1)
        return outputs, argmax_action, evalQ

    def copy_target(self):
        self.target_representation.set_weights(self.representation.get_weights())
        self.target_utility.set_weights(self.utility.get_weights())
        self.target_payoffs.set_weights(self.payoffs.get_weights())
        if self.dcg_s:
            self.target_bias.set_weights(self.bias.get_weights())


class ActorNet(tk.Model):
    def __init__(self,
                 state_dim: int,
                 n_agents: int,
                 action_dim: int,
                 hidden_sizes: Sequence[int],
                 normalize: Optional[tk.layers.Layer] = None,
                 initializer: Optional[tk.initializers.Initializer] = None,
                 activation: Optional[tk.layers.Layer] = None,
                 device: str = "cpu:0"):
        super(ActorNet, self).__init__()
        layers = []
        input_shape = (state_dim + n_agents,)
        for h in hidden_sizes:
            mlp, input_shape = mlp_block(input_shape[0], h, normalize, activation, initializer, device)
            layers.extend(mlp)
        layers.extend(mlp_block(input_shape[0], action_dim, None, tk.layers.Activation("tanh"), initializer, device)[0])
        self.model = tk.Sequential(layers)

    def call(self, x: tf.Tensor, **kwargs):
        return self.model(x)


class CriticNet(tk.Model):
    def __init__(self,
                 independent: bool,
                 state_dim: int,
                 n_agents: int,
                 action_dim: int,
                 hidden_sizes: Sequence[int],
                 normalize: Optional[tk.layers.Layer] = None,
                 initializer: Optional[tk.initializers.Initializer] = None,
                 activation: Optional[tk.layers.Layer] = None,
                 device: str = "cpu:0"
                 ):
        super(CriticNet, self).__init__()
        layers = []
        if independent:
            input_shape = (state_dim + action_dim + n_agents,)
        else:
            input_shape = (state_dim * n_agents + action_dim * n_agents + n_agents,)
        for h in hidden_sizes:
            mlp, input_shape = mlp_block(input_shape[0], h, normalize, activation, initializer, device)
            layers.extend(mlp)
        layers.extend(mlp_block(input_shape[0], 1, None, None, initializer, device)[0])
        self.model = tk.Sequential(layers)

    def call(self, x: tf.Tensor, **kwargs):
        return self.model(x)


class Basic_DDPG_policy(tk.Model):
    def __init__(self,
                 action_space: Space,
                 n_agents: int,
                 representation: Optional[Basic_Identical],
                 actor_hidden_size: Sequence[int],
                 critic_hidden_size: Sequence[int],
                 normalize: Optional[tk.layers.Layer] = None,
                 initializer: Optional[tk.initializers.Initializer] = None,
                 activation: Optional[tk.layers.Layer] = None,
                 device: str = "cpu:0"
                 ):
        super(Basic_DDPG_policy, self).__init__()
        self.action_dim = action_space.shape[0]
        self.n_agents = n_agents
        self.representation = representation
        self.obs_dim = self.representation.input_shapes[0]
        self.representation_info_shape = self.representation.output_shapes

        self.actor_net = ActorNet(representation.output_shapes['state'][0], n_agents, self.action_dim,
                                  actor_hidden_size, normalize, initializer, activation, device)
        self.target_actor_net = ActorNet(representation.output_shapes['state'][0], n_agents, self.action_dim,
                                         actor_hidden_size, normalize, initializer, activation, device)
        self.critic_net = CriticNet(True, representation.output_shapes['state'][0], n_agents, self.action_dim,
                                    critic_hidden_size, normalize, initializer, activation, device)
        self.target_critic_net = CriticNet(True, representation.output_shapes['state'][0], n_agents, self.action_dim,
                                           critic_hidden_size, normalize, initializer, activation, device)
        if isinstance(self.representation, Basic_Identical):
            self.parameters_actor = self.actor_net.trainable_variables
        else:
            self.parameters_actor = self.representation.trainable_variables + self.actor_net.trainable_variables
        self.parameters_critic = self.critic_net.trainable_variables
        self.soft_update(1.0)

    def call(self, inputs: Union[np.ndarray, dict], **kwargs):
        observations = tf.reshape(inputs['obs'], [-1, self.obs_dim])
        IDs = tf.reshape(inputs['ids'], [-1, self.n_agents])
        outputs = self.representation(observations)
        actor_in = tf.concat([outputs['state'], IDs], axis=-1)
        act = self.actor_net(actor_in)
        return outputs, tf.reshape(act, [-1, self.n_agents, self.action_dim])

    def critic(self, observation: tf.Tensor, actions: tf.Tensor, agent_ids: tf.Tensor):
        observation = tf.reshape(observation, [-1, self.obs_dim])
        actions = tf.reshape(actions, [-1, self.action_dim])
        agent_ids = tf.reshape(agent_ids, [-1, self.n_agents])
        outputs = self.representation(observation)
        critic_in = tf.concat([outputs['state'], actions, agent_ids], axis=-1)
        return tf.reshape(self.critic_net(critic_in), [-1, self.n_agents, 1])

    def target_critic(self, observation: tf.Tensor, actions: tf.Tensor, agent_ids: tf.Tensor):
        observation = tf.reshape(observation, [-1, self.obs_dim])
        actions = tf.reshape(actions, [-1, self.action_dim])
        agent_ids = tf.reshape(agent_ids, [-1, self.n_agents])
        outputs = self.representation(observation)
        critic_in = tf.concat([outputs['state'], actions, agent_ids], axis=-1)
        return tf.reshape(self.target_critic_net(critic_in), [-1, self.n_agents, 1])

    def target_actor(self, inputs: Union[np.ndarray, dict]):
        observations = tf.reshape(inputs['obs'], [-1, self.obs_dim])
        IDs = tf.reshape(inputs['ids'], [-1, self.n_agents])
        outputs = self.representation(observations)
        actor_in = tf.concat([outputs['state'], IDs], axis=-1)
        act = self.target_actor_net(actor_in)
        return tf.reshape(act, [-1, self.n_agents, self.action_dim])

    def soft_update(self, tau=0.005):
        for ep, tp in zip(self.actor_net.variables, self.target_actor_net.variables):
            tp.assign((1 - tau) * tp + tau * ep)
        for ep, tp in zip(self.critic_net.variables, self.target_critic_net.variables):
            tp.assign((1 - tau) * tp + tau * ep)


class MADDPG_policy(Basic_DDPG_policy):
    def __init__(self,
                 action_space: Space,
                 n_agents: int,
                 representation: Optional[Basic_Identical],
                 actor_hidden_size: Sequence[int],
                 critic_hidden_size: Sequence[int],
                 normalize: Optional[tk.layers.Layer] = None,
                 initializer: Optional[tk.initializers.Initializer] = None,
                 activation: Optional[tk.layers.Layer] = None,
                 device: str = "cpu:0"
                 ):
        super(MADDPG_policy, self).__init__(action_space, n_agents, representation,
                                            actor_hidden_size, critic_hidden_size,
                                            normalize, initializer, activation, device)
        self.critic_net = CriticNet(False, representation.output_shapes['state'][0], n_agents, self.action_dim,
                                    critic_hidden_size, normalize, initializer, activation, device)
        self.target_critic_net = CriticNet(False, representation.output_shapes['state'][0], n_agents, self.action_dim,
                                           critic_hidden_size, normalize, initializer, activation, device)
        self.parameters_critic = self.critic_net.trainable_variables
        self.soft_update(1.0)

    def critic(self, observation: tf.Tensor, actions: tf.Tensor, agent_ids: tf.Tensor):
        bs = observation.shape[0]
        outputs_n = tf.reshape(self.representation(observation)['state'], (bs, 1, -1))
        outputs_n = tf.tile(outputs_n, (1, self.n_agents, 1))
        actions_n = tf.tile(tf.reshape(actions, (bs, 1, -1)), (1, self.n_agents, 1))
        critic_in = tf.concat([outputs_n, actions_n, agent_ids], axis=-1)
        return self.critic_net(critic_in)

    def target_critic(self, observation: tf.Tensor, actions: tf.Tensor, agent_ids: tf.Tensor):
        bs = observation.shape[0]
        outputs_n = tf.reshape(self.representation(observation)['state'], (bs, 1, -1))
        outputs_n = tf.tile(outputs_n, (1, self.n_agents, 1))
        actions_n = tf.tile(tf.reshape(actions, (bs, 1, -1)), (1, self.n_agents, 1))
        critic_in = tf.concat([outputs_n, actions_n, agent_ids], axis=-1)
        return self.target_critic_net(critic_in)


class MATD3_policy(tk.Model):
    def __init__(self,
                 action_space: Space,
                 n_agents: int,
                 representation: Optional[Basic_Identical],
                 actor_hidden_size: Sequence[int],
                 critic_hidden_size: Sequence[int],
                 normalize: Optional[tk.layers.Layer] = None,
                 initializer: Optional[tk.initializers.Initializer] = None,
                 activation: Optional[tk.layers.Layer] = None,
                 device: str = "cpu:0"
                 ):
        super(MATD3_policy, self).__init__()
        self.action_dim = action_space.shape[0]
        self.n_agents = n_agents
        self.representation = representation
        self.obs_dim = self.representation.input_shapes[0]
        self.representation_info_shape = self.representation.output_shapes

        self.actor_net = ActorNet(representation.output_shapes['state'][0], n_agents, self.action_dim,
                                  actor_hidden_size, normalize, initializer, activation, device)
        self.target_actor_net = ActorNet(representation.output_shapes['state'][0], n_agents, self.action_dim,
                                         actor_hidden_size, normalize, initializer, activation, device)
        self.critic_net_A = CriticNet(False, representation.output_shapes['state'][0], n_agents, self.action_dim,
                                      critic_hidden_size, normalize, initializer, activation, device)
        self.critic_net_B = CriticNet(False, representation.output_shapes['state'][0], n_agents, self.action_dim,
                                      critic_hidden_size, normalize, initializer, activation, device)
        self.target_critic_net_A = CriticNet(False, representation.output_shapes['state'][0], n_agents, self.action_dim,
                                             critic_hidden_size, normalize, initializer, activation, device)
        self.target_critic_net_B = CriticNet(False, representation.output_shapes['state'][0], n_agents, self.action_dim,
                                             critic_hidden_size, normalize, initializer, activation, device)
        self.soft_update(tau=1.0)
        self.critic_parameters = self.critic_net_A.trainable_variables + self.critic_net_B.trainable_variables

    def call(self, inputs: Union[np.ndarray, dict], **kwargs):
        observations = tf.reshape(inputs['obs'], [-1, self.obs_dim])
        IDs = tf.reshape(inputs['ids'], [-1, self.n_agents])
        outputs = self.representation(observations)
        actor_in = tf.concat([outputs['state'], IDs], axis=-1)
        act = self.actor_net(actor_in)
        return outputs, tf.reshape(act, [-1, self.n_agents, self.action_dim])

    def critic(self, observation: tf.Tensor, actions: tf.Tensor, agent_ids: tf.Tensor):
        bs = observation.shape[0]
        outputs_n = tf.reshape(self.representation(observation)['state'], (bs, 1, -1))
        outputs_n = tf.tile(outputs_n, (1, self.n_agents, 1))
        actions_n = tf.tile(tf.reshape(actions, (bs, 1, -1)), (1, self.n_agents, 1))
        critic_in = tf.concat([outputs_n, actions_n, agent_ids], axis=-1)
        qa = self.critic_net_A(critic_in)
        qb = self.critic_net_B(critic_in)
        return outputs_n, (qa + qb) / 2.0

    def target_critic(self, observation: tf.Tensor, actions: tf.Tensor, agent_ids: tf.Tensor):
        bs = observation.shape[0]
        outputs_n = tf.reshape(self.representation(observation)['state'], (bs, 1, -1))
        outputs_n = tf.tile(outputs_n, (1, self.n_agents, 1))
        actions_n = tf.tile(tf.reshape(actions, (bs, 1, -1)), (1, self.n_agents, 1))
        critic_in = tf.concat([outputs_n, actions_n, agent_ids], axis=-1)
        qa = self.target_critic_net_A(critic_in)
        qb = self.target_critic_net_B(critic_in)
        min_q = tf.math.minimum(qa, qb)
        return outputs_n, min_q

    def Qaction(self, observation: tf.Tensor, actions: tf.Tensor, agent_ids: tf.Tensor):
        bs = observation.shape[0]
        outputs_n = tf.reshape(self.representation(observation)['state'], (bs, 1, -1))
        outputs_n = tf.tile(outputs_n, (1, self.n_agents, 1))
        actions_n = tf.tile(tf.reshape(actions, (bs, 1, -1)), (1, self.n_agents, 1))
        critic_in = tf.concat([outputs_n, actions_n, agent_ids], axis=-1)
        qa = self.critic_net_A(critic_in)
        qb = self.critic_net_B(critic_in)
        return outputs_n, tf.concat((qa, qb), axis=-1)

    def target_actor(self, inputs: Union[np.ndarray, dict]):
        observations = tf.reshape(inputs['obs'], [-1, self.obs_dim])
        IDs = tf.reshape(inputs['ids'], [-1, self.n_agents])
        outputs = self.representation(observations)
        actor_in = tf.concat([outputs['state'], IDs], axis=-1)
        act = self.target_actor_net(actor_in)
        return tf.reshape(act, [-1, self.n_agents, self.action_dim])

    def soft_update(self, tau=0.005):
        for ep, tp in zip(self.actor_net.variables, self.target_actor_net.variables):
            tp.assign((1 - tau) * tp + tau * ep)
        for ep, tp in zip(self.critic_net_A.variables, self.target_critic_net_A.variables):
            tp.assign((1 - tau) * tp + tau * ep)
        for ep, tp in zip(self.critic_net_B.variables, self.target_critic_net_B.variables):
            tp.assign((1 - tau) * tp + tau * ep)
