from xuance.tensorflow.policies import *
from xuance.tensorflow.utils import *
from xuance.tensorflow.representations import Basic_Identical
import tensorflow_probability as tfp

tfd = tfp.distributions


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
                 device: str = "cpu:0"):
        super(BasicQnetwork, self).__init__()
        self.action_dim = action_space.n
        self.representation = representation
        self.representation_info_shape = self.representation.output_shapes

        self.eval_Qhead = BasicQhead(self.representation.output_shapes['state'][0], self.action_dim, n_agents,
                                     hidden_size, normalize, initializer, activation, device)
        self.target_Qhead = BasicQhead(self.representation.output_shapes['state'][0], self.action_dim, n_agents,
                                       hidden_size, normalize, initializer, activation, device)
        self.copy_target()

    def call(self, inputs: Union[np.ndarray, dict], **kwargs):
        observations = tf.reshape(inputs['obs'], [-1, self.obs_dim])
        IDs = tf.reshape(inputs['ids'], [-1, self.n_agents])
        outputs = self.representation(observations)
        q_inputs = tf.concat([outputs['state'], IDs], axis=-1)
        evalQ = tf.reshape(self.eval_Qhead(q_inputs), [-1, self.n_agents, self.action_dim])
        argmax_action = tf.argmax(evalQ, axis=-1)
        return outputs, argmax_action, evalQ

    def target_Q(self, inputs: Union[np.ndarray, dict]):
        shape_obs = inputs["obs"].shape
        shape_ids = inputs["ids"].shape
        observations = tf.reshape(inputs['obs'], [-1, shape_obs[-1]])
        IDs = tf.reshape(inputs['ids'], [-1, shape_ids[-1]])
        outputs = self.representation(observations)
        q_inputs = tf.concat([outputs['state'], IDs], axis=-1)
        return tf.reshape(self.target_Qhead(q_inputs), shape_obs[0:-1] + (self.action_dim,))

    def copy_target(self):
        self.target_Qhead.set_weights(self.eval_Qhead.get_weights())


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
        self.device = device
        layers = []
        input_shape = (state_dim + n_agents,)
        for h in hidden_sizes:
            mlp, input_shape = mlp_block(input_shape[0], h, normalize, activation, initializer, device)
            layers.extend(mlp)
        # layers.extend(mlp_block(input_shape[0], action_dim, None, nn.ReLU, initialize, device)[0])
        # self.mu = tk.Sequential(*layers)
        # self.logstd = tk.Sequential(*layers)
        self.outputs = tk.Sequential(layers)
        self.out_mu = tk.layers.Dense(units=action_dim, input_shape=(hidden_sizes[0],))
        self.out_std = tk.layers.Dense(units=action_dim, input_shape=(hidden_sizes[0],))

    def call(self, x: tf.Tensor, **kwargs):
        output = self.outputs(x)
        mu = tf.sigmoid(self.out_mu(output))
        std = tf.clip_by_value(self.out_std(output), -20, 1)
        std = tf.exp(std)
        return mu, std


class CriticNet(tk.Model):
    def __init__(self,
                 state_dim: int,
                 n_agents: int,
                 hidden_sizes: Sequence[int],
                 normalize: Optional[tk.layers.Layer] = None,
                 initializer: Optional[tk.initializers.Initializer] = None,
                 activation: Optional[tk.layers.Layer] = None,
                 device: str = "cpu:0"
                 ):
        super(CriticNet, self).__init__()
        layers = []
        input_shape = (state_dim + n_agents,)
        for h in hidden_sizes:
            mlp, input_shape = mlp_block(input_shape[0], h, normalize, activation, initializer, device)
            layers.extend(mlp)
        layers.extend(mlp_block(input_shape[0], 1, None, None, initializer, device)[0])
        self.model = tk.Sequential(layers)

    def call(self, x: tf.Tensor, **kwargs):
        return self.model(x)


class MAAC_Policy(tk.Model):
    """
    MAAC_Policy: Multi-Agent Actor-Critic Policy with Gaussian policies
    """

    def __init__(self,
                 action_space: Discrete,
                 n_agents: int,
                 representation: tk.Model,
                 mixer: Optional[VDN_mixer] = None,
                 actor_hidden_size: Sequence[int] = None,
                 critic_hidden_size: Sequence[int] = None,
                 normalize: Optional[tk.layers.Layer] = None,
                 initialize: Optional[tk.initializers.Initializer] = None,
                 activation: Optional[tk.layers.Layer] = None,
                 device: Optional[Union[str, int]] = None,
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
                              actor_hidden_size, normalize, initialize, activation, device)
        dim_input_critic = self.representation_critic.output_shapes['state'][0]
        self.critic = CriticNet(dim_input_critic, n_agents,  critic_hidden_size,
                                normalize, initialize, activation, device)
        self.mixer = mixer
        self.identical_rep = True if isinstance(self.representation, Basic_Identical) else False
        self.pi_dist = None

    def call(self, inputs: Union[np.ndarray, dict], *rnn_hidden, **kwargs):
        observation = inputs['obs']
        agent_ids = inputs['ids']
        obs_shape = observation.shape
        if self.use_rnn:
            outputs = self.representation(observation, *rnn_hidden)
            outputs_state = outputs['state']  # need to be improved
            rnn_hidden = (outputs['rnn_hidden'], outputs['rnn_cell'])
        else:
            observation_reshape = tf.reshape(observation, [-1, obs_shape[-1]])
            outputs = self.representation(observation_reshape)
            outputs_state = tf.reshape(outputs['state'], obs_shape[:-1] + self.representation_info_shape['state'])
            rnn_hidden = None
        actor_input = tf.concat([outputs_state, agent_ids], axis=-1)
        mu, std = self.actor(actor_input)
        mu = tf.reshape(mu, [-1, self.n_agents, self.action_dim])
        std = tf.reshape(std, [-1, self.n_agents, self.action_dim])
        cov_mat = tf.linalg.diag(std)
        dist = tfd.MultivariateNormalTriL(loc=mu, scale_tril=cov_mat)
        return rnn_hidden, dist

    def get_values(self, critic_in: tf.Tensor, agent_ids: tf.Tensor, *rnn_hidden: tf.Tensor, **kwargs):
        shape_obs = critic_in.shape
        # get representation features
        if self.use_rnn:
            batch_size, n_agent, episode_length, dim_obs = tuple(shape_obs)
            outputs = self.representation_critic(critic_in.reshape(-1, episode_length, dim_obs), *rnn_hidden)
            outputs['state'] = outputs['state'].view(batch_size, n_agent, episode_length, -1)
            rnn_hidden = (outputs['rnn_hidden'], outputs['rnn_cell'])
        else:
            batch_size, n_agent, dim_obs = tuple(shape_obs)
            outputs = self.representation_critic(tf.reshape(critic_in, [-1, dim_obs]))
            outputs['state'] = tf.reshape(outputs['state'], [batch_size, n_agent, -1])
            rnn_hidden = None
        # get critic values
        critic_in = tf.concat([outputs['state'], agent_ids], axis=-1)
        v = self.critic(critic_in)
        return rnn_hidden, v

    def value_tot(self, values_n: tf.Tensor, global_state=None):
        if global_state is not None:
            global_state = torch.as_tensor(global_state).to(self.device)
        return values_n if self.mixer is None else self.mixer(values_n, global_state)

    def trainable_param(self):
        params = self.actor.trainable_variables + self.critic.trainable_variables
        if self.mixer is not None:
            params += self.mixer.trainable_variables
        if self.identical_rep:
            return params
        else:
            return params + self.representation.trainable_variables


class Basic_ISAC_policy(tk.Model):
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
        super(Basic_ISAC_policy, self).__init__()
        self.action_dim = action_space.shape[0]
        self.n_agents = n_agents
        self.representation = representation
        self.obs_dim = self.representation.input_shapes[0]
        self.representation_info_shape = self.representation.output_shapes

        self.actor_net = ActorNet(representation.output_shapes['state'][0], n_agents, self.action_dim,
                                  actor_hidden_size, normalize, initializer, activation, device)
        dim_input_critic = representation.output_shapes['state'][0] + self.action_dim
        self.critic_net = CriticNet(dim_input_critic, n_agents, critic_hidden_size,
                                    normalize, initializer, activation, device)
        self.target_actor_net = ActorNet(representation.output_shapes['state'][0], n_agents, self.action_dim,
                                         actor_hidden_size, normalize, initializer, activation, device)
        self.target_critic_net = CriticNet(dim_input_critic, n_agents, critic_hidden_size,
                                           normalize, initializer, activation, device)
        if isinstance(self.representation, Basic_Identical):
            self.parameters_actor = self.actor_net.trainable_variables
        else:
            self.parameters_actor = self.representation.trainable_variables + self.actor_net.trainable_variables
        self.parameters_critic = self.critic_net.trainable_variables
        self.soft_update(tau=1.0)

    def call(self, inputs: Union[np.ndarray, dict], **kwargs):
        observations = tf.reshape(inputs['obs'], [-1, self.obs_dim])
        IDs = tf.reshape(inputs['ids'], [-1, self.n_agents])
        outputs = self.representation(observations)
        actor_in = tf.concat([outputs['state'], IDs], axis=-1)
        mu, std = self.actor_net(actor_in)
        mu = tf.reshape(mu, [-1, self.n_agents, self.action_dim])
        std = tf.reshape(std, [-1, self.n_agents, self.action_dim])
        cov_mat = tf.linalg.diag(std)
        dist = tfd.MultivariateNormalTriL(loc=mu, scale_tril=cov_mat)
        return outputs, dist

    def critic(self, observation: tf.Tensor, actions: tf.Tensor, agent_ids: tf.Tensor):
        outputs = self.representation(observation)
        critic_in = tf.concat([outputs['state'], actions, agent_ids], axis=-1)
        return self.critic_net(critic_in)

    def target_critic(self, observation: tf.Tensor, actions: tf.Tensor, agent_ids: tf.Tensor):
        outputs = self.representation(observation)
        critic_in = tf.concat([outputs['state'], actions, agent_ids], axis=-1)
        return self.target_critic_net(critic_in)

    def target_actor(self, observation: tf.Tensor, agent_ids: tf.Tensor):
        outputs = self.representation(observation)
        actor_in = tf.concat([outputs['state'], agent_ids], axis=-1)
        mu, std = self.target_actor_net(actor_in)
        mu = tf.reshape(mu, [-1, self.n_agents, self.action_dim])
        std = tf.reshape(std, [-1, self.n_agents, self.action_dim])
        cov_mat = tf.linalg.diag(std)
        dist = tfd.MultivariateNormalTriL(loc=mu, scale_tril=cov_mat)
        return dist

    def soft_update(self, tau=0.005):
        for ep, tp in zip(self.actor_net.variables, self.target_actor_net.variables):
            tp.assign((1 - tau) * tp + tau * ep)
        for ep, tp in zip(self.critic_net.variables, self.target_critic_net.variables):
            tp.assign((1 - tau) * tp + tau * ep)


class MASAC_policy(Basic_ISAC_policy):
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
        super(MASAC_policy, self).__init__(action_space, n_agents, representation,
                                           actor_hidden_size, critic_hidden_size,
                                           normalize, initializer, activation, device)
        dim_input_critic = (representation.output_shapes['state'][0] + self.action_dim) * self.n_agents
        self.critic_net = CriticNet(dim_input_critic, n_agents, critic_hidden_size,
                                    normalize, initializer, activation, device)
        self.target_critic_net = CriticNet(dim_input_critic, n_agents, critic_hidden_size,
                                           normalize, initializer, activation, device)
        self.parameters_critic = self.critic_net.trainable_variables
        self.soft_update(tau=1.0)

    def critic(self, observation: tf.Tensor, actions: tf.Tensor, agent_ids: tf.Tensor):
        bs = observation.shape[0]
        outputs_n = self.representation(observation)['state']
        outputs_n = tf.tile(tf.reshape(outputs_n, [bs, 1, -1]), (1, self.n_agents, 1))
        actions_n = tf.tile(tf.reshape(actions, [bs, 1, -1]), (1, self.n_agents, 1))
        critic_in = tf.concat([outputs_n, actions_n, agent_ids], axis=-1)
        return self.critic_net(critic_in)

    def target_critic(self, observation: tf.Tensor, actions: tf.Tensor, agent_ids: tf.Tensor):
        bs = observation.shape[0]
        outputs_n = self.representation(observation)['state']
        outputs_n = tf.tile(tf.reshape(outputs_n, [bs, 1, -1]), (1, self.n_agents, 1))
        actions_n = tf.tile(tf.reshape(actions, [bs, 1, -1]), (1, self.n_agents, 1))
        critic_in = tf.concat([outputs_n, actions_n, agent_ids], axis=-1)
        return self.target_critic_net(critic_in)
