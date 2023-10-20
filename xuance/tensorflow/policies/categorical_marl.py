from xuance.tensorflow.policies import *
from xuance.tensorflow.utils import *
from xuance.tensorflow.representations import Basic_Identical
from .deterministic_marl import BasicQhead


class ActorNet(tk.Model):
    def __init__(self,
                 state_dim: int,
                 action_dim: int,
                 n_agents: int,
                 hidden_sizes: Sequence[int],
                 normalize: Optional[tk.layers.Layer] = None,
                 initializer: Optional[tk.initializers.Initializer] = None,
                 activation: Optional[tk.layers.Layer] = None,
                 device: Optional[Union[str, int, torch.device]] = None):
        super(ActorNet, self).__init__()
        layers = []
        input_shape = (state_dim + n_agents,)
        for h in hidden_sizes:
            mlp, input_shape = mlp_block(input_shape[0], h, normalize, activation, initializer, device)
            layers.extend(mlp)
        layers.extend(mlp_block(input_shape[0], action_dim, None, None, None, device)[0])
        self.model = tk.Sequential(layers)
        self.dist = CategoricalDistribution(action_dim)

    def call(self, x: tf.Tensor, training=None, masks=None):
        self.dist.set_param(self.model(x))
        return self.dist


class CriticNet(tk.Model):
    def __init__(self,
                 state_dim: int,
                 n_agents: int,
                 hidden_sizes: Sequence[int],
                 normalize: Optional[tk.layers.Layer] = None,
                 initializer: Optional[tk.initializers.Initializer] = None,
                 activation: Optional[tk.layers.Layer] = None,
                 device: Optional[Union[str, int, torch.device]] = None):
        super(CriticNet, self).__init__()
        layers = []
        input_shape = (state_dim + n_agents,)
        for h in hidden_sizes:
            mlp, input_shape = mlp_block(input_shape[0], h, normalize, activation, initializer, device)
            layers.extend(mlp)
        layers.extend(mlp_block(input_shape[0], 1, None, None, None, device)[0])
        self.model = tk.Sequential(layers)

    def call(self, x: tf.Tensor, training=None, masks=None):
        return self.model(x)[:, :, 0]


class COMA_CriticNet(tk.Model):
    def __init__(self,
                 state_dim: int,
                 obs_dim: int,
                 act_dim: int,
                 n_agents: int,
                 hidden_sizes: Sequence[int],
                 normalize: Optional[tk.layers.Layer] = None,
                 initializer: Optional[tk.initializers.Initializer] = None,
                 activation: Optional[tk.layers.Layer] = None,
                 device: Optional[Union[str, int, torch.device]] = None):
        super(COMA_CriticNet, self).__init__()
        layers = []
        input_shape = (state_dim + obs_dim + act_dim * n_agents + n_agents,)
        for h in hidden_sizes:
            mlp, input_shape = mlp_block(input_shape[0], h, normalize, activation, initializer, device)
            layers.extend(mlp)
        layers.extend(mlp_block(input_shape[0], act_dim, None, None, None, device)[0])
        self.model = tk.Sequential(layers)

    def call(self, x: tf.Tensor, training=None, masks=None):
        return self.model(x)


class MultiAgentActorCriticPolicy(tk.Model):
    def __init__(self,
                 action_space: Discrete,
                 n_agents: int,
                 representation: Optional[Basic_Identical],
                 mixer: Optional[VDN_mixer] = None,
                 actor_hidden_size: Sequence[int] = None,
                 critic_hidden_size: Sequence[int] = None,
                 normalize: Optional[tk.layers.Layer] = None,
                 initializer: Optional[tk.initializers.Initializer] = None,
                 activation: Optional[tk.layers.Layer] = None,
                 device: Optional[Union[str, int, torch.device]] = None):
        assert isinstance(action_space, Discrete)
        super(MultiAgentActorCriticPolicy, self).__init__()
        self.device = device
        self.action_dim = action_space.n
        self.n_agents = n_agents
        self.representation = representation
        self.representation_info_shape = self.representation.output_shapes
        self.obs_dim = self.representation.input_shapes[0]
        self.actor = ActorNet(representation.output_shapes['state'][0], self.action_dim, n_agents,
                              actor_hidden_size, normalize, initializer, activation, device)
        self.critic = CriticNet(representation.output_shapes['state'][0], n_agents, critic_hidden_size,
                                normalize, initializer, activation, device)
        self.mixer = mixer
        if mixer is None:
            self.parameters_train = self.actor.trainable_variables + self.critic.trainable_variables
        else:
            self.parameters_train = self.actor.trainable_variables + self.critic.trainable_variables + self.mixer.trainable_variables
        if not isinstance(self.representation, Basic_Identical):
            self.parameters_train += self.representation.trainable_variables

    def call(self, inputs: Union[np.ndarray, dict], training=None, masks=None):
        observation = inputs['obs']
        agent_ids = inputs['ids']
        outputs = self.representation(observation)
        input_with_id = tf.concat([outputs['state'], agent_ids], axis=-1)
        act_dist = self.actor(input_with_id)
        v = tf.expand_dims(self.critic(input_with_id), axis=-1)
        return outputs, act_dist, v

    def value_tot(self, values_n: tf.Tensor, global_state=None):
        if global_state is not None:
            with tf.device(self.device):
                global_state = tf.convert_to_tensor(global_state)
        return values_n if self.mixer is None else self.mixer(values_n, global_state)


class MAPPO_ActorCriticPolicy(MultiAgentActorCriticPolicy):
    def __init__(self,
                 dim_state: int,
                 action_space: Discrete,
                 n_agents: int,
                 representation: Optional[Basic_Identical],
                 actor_hidden_size: Sequence[int] = None,
                 critic_hidden_size: Sequence[int] = None,
                 normalize: Optional[tk.layers.Layer] = None,
                 initializer: Optional[tk.initializers.Initializer] = None,
                 activation: Optional[tk.layers.Layer] = None,
                 device: Optional[Union[str, int, torch.device]] = None):
        assert isinstance(action_space, Discrete)
        super(MAPPO_ActorCriticPolicy, self).__init__(action_space, n_agents, representation, None,
                                                      actor_hidden_size, critic_hidden_size,
                                                      normalize, initializer, activation, device)
        self.critic = CriticNet(dim_state, n_agents, critic_hidden_size, normalize, initializer, activation, device)
        if isinstance(self.representation, Basic_Identical):
            self.parameters_train = self.actor.trainable_variables + self.critic.trainable_variables
        else:
            self.parameters_train = self.representation.trainable_variables + self.actor.trainable_variables + self.critic.trainable_variables

    def call(self, inputs: Union[np.ndarray, dict], training=None, masks=None):
        observation = inputs['obs']
        agent_ids = inputs['ids']
        outputs = self.representation(observation)
        input_with_id = tf.concat([outputs['state'], agent_ids], axis=-1)
        act_dist = self.actor(input_with_id)
        return outputs, act_dist

    def values(self, state: tf.Tensor, agent_ids: tf.Tensor):
        input_with_id = tf.concat([state, agent_ids], axis=-1)
        return tf.expand_dims(self.critic(input_with_id), axis=-1)


class MeanFieldActorCriticPolicy(tk.Model):
    def __init__(self,
                 action_space: Discrete,
                 n_agents: int,
                 representation: Optional[Basic_Identical],
                 actor_hidden_size: Sequence[int] = None,
                 critic_hidden_size: Sequence[int] = None,
                 normalize: Optional[tk.layers.Layer] = None,
                 initializer: Optional[tk.initializers.Initializer] = None,
                 activation: Optional[tk.layers.Layer] = None,
                 device: Optional[Union[str, int, torch.device]] = None
                 ):
        assert isinstance(action_space, Discrete)
        super(MeanFieldActorCriticPolicy, self).__init__()
        self.action_dim = action_space.n
        self.n_agents = n_agents
        self.representation = representation
        self.representation_info_shape = self.representation.output_shapes
        self.obs_dim = self.representation.input_shapes[0]
        self.actor_net = ActorNet(representation.output_shapes['state'][0], self.action_dim, n_agents,
                                  actor_hidden_size, normalize, initializer, activation, device)
        self.critic_net = BasicQhead(representation.output_shapes['state'][0] + self.action_dim, self.action_dim,
                                     n_agents, critic_hidden_size, normalize, initializer, activation, device)
        self.target_actor_net = ActorNet(representation.output_shapes['state'][0], self.action_dim, n_agents,
                                         actor_hidden_size, normalize, initializer, activation, device)
        self.target_critic_net = BasicQhead(representation.output_shapes['state'][0] + self.action_dim, self.action_dim,
                                            n_agents, critic_hidden_size, normalize, initializer, activation, device)
        if isinstance(self.representation, Basic_Identical):
            self.parameters_actor = self.actor_net.trainable_variables
        else:
            self.parameters_actor = self.actor_net.trainable_variables + self.representation.trainable_variables
        self.parameters_critic = self.critic_net.trainable_variables
        self.soft_update(tau=1.0)

    def call(self, inputs: Union[np.ndarray, dict], training=None, masks=None):
        observations = inputs['obs']
        IDs = inputs['ids']
        outputs = self.representation(observations)
        input_actor = tf.concat([outputs['state'], IDs], axis=-1)
        act_dist = self.actor_net(input_actor)
        return outputs, act_dist

    def target_actor(self, observation: tf.Tensor, agent_ids: tf.Tensor):
        outputs = self.representation(observation)
        input_actor = tf.concat([outputs['state'], agent_ids], axis=-1)
        act_dist = self.target_actor_net(input_actor)
        return act_dist

    def critic(self, observation: tf.Tensor, actions_mean: tf.Tensor, agent_ids: tf.Tensor):
        outputs = self.representation(observation)
        critic_in = tf.concat([outputs['state'], actions_mean, agent_ids], axis=-1)
        return self.critic_net(critic_in)

    def target_critic(self, observation: tf.Tensor, actions_mean: tf.Tensor, agent_ids: tf.Tensor):
        outputs = self.representation(observation)
        critic_in = tf.concat([outputs['state'], actions_mean, agent_ids], axis=-1)
        return self.target_critic_net(critic_in)

    def soft_update(self, tau=0.005):
        for ep, tp in zip(self.actor_net.variables, self.target_actor_net.variables):
            tp.assign((1 - tau) * tp + tau * ep)
        for ep, tp in zip(self.critic_net.variables, self.target_critic_net.variables):
            tp.assign((1 - tau) * tp + tau * ep)


class COMAPolicy(tk.Model):
    def __init__(self,
                 state_dim: int,
                 action_space: Discrete,
                 n_agents: int,
                 representation: Optional[Basic_Identical],
                 actor_hidden_size: Sequence[int] = None,
                 critic_hidden_size: Sequence[int] = None,
                 normalize: Optional[tk.layers.Layer] = None,
                 initializer: Optional[tk.initializers.Initializer] = None,
                 activation: Optional[tk.layers.Layer] = None,
                 device: Optional[Union[str, int, torch.device]] = None):
        assert isinstance(action_space, Discrete)
        super(COMAPolicy, self).__init__()
        self.device = device
        self.action_dim = action_space.n
        self.n_agents = n_agents
        self.representation = representation
        self.representation_info_shape = self.representation.output_shapes
        self.obs_dim = self.representation.input_shapes[0]
        self.actor = ActorNet(representation.output_shapes['state'][0], self.action_dim, n_agents,
                              actor_hidden_size, normalize, initializer, activation, device)
        self.critic = COMA_CriticNet(state_dim, representation.output_shapes['state'][0], self.action_dim, n_agents,
                                     critic_hidden_size, normalize, initializer, activation, device)
        self.target_critic = COMA_CriticNet(state_dim, representation.output_shapes['state'][0], self.action_dim,
                                            n_agents, critic_hidden_size, normalize, initializer, activation, device)
        if isinstance(self.representation, Basic_Identical):
            self.parameters_actor = self.actor.trainable_variables
        else:
            self.parameters_actor = self.representation.trainable_variables + self.actor.trainable_variables
        self.parameters_critic = self.critic.trainable_variables
        self.copy_target()

    def build_critic_in(self, state, observations, actions_onehot, agent_ids, t=None):
        bs, act_dim = state.shape[0], actions_onehot.shape[-1]
        step_len = state.shape[1] if t is None else 1
        ts = slice(None) if t is None else slice(t, t + 1)
        obs_encode = self.representation(observations)['state']
        inputs = [state[:, ts], obs_encode[:, ts]]
        # counterfactual actions inputs
        actions_joint = tf.tile(tf.reshape(actions_onehot[:, ts], (bs, step_len, 1, -1)), (1, 1, self.n_agents, 1))
        agent_mask = tf.tile(tf.reshape(1 - torch.eye(self.n_agents), [-1, 1]), (1, act_dim))
        agent_mask = tf.reshape(agent_mask, [self.n_agents, -1])
        agent_mask = tf.expand_dims(tf.expand_dims(agent_mask, axis=0), axis=0)
        inputs.append(actions_joint * agent_mask)
        inputs.append(agent_ids[:, ts])
        return tf.concat(inputs, axis=-1)

    def call(self, inputs: Union[np.ndarray, dict], training=None, masks=None):
        observations = inputs['obs']
        IDs = inputs['ids']
        outputs = self.representation(observations)
        input_with_id = tf.concat([outputs['state'], IDs], axis=-1)
        act_dist = self.actor(input_with_id)
        return outputs, act_dist

    def copy_target(self):
        self.target_critic.set_weights(self.critic.get_weights())
