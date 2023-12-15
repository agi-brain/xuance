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
                 gain: float = 1.0,
                 activation: Optional[tk.layers.Layer] = None,
                 device: str = "cpu:0"):
        super(ActorNet, self).__init__()
        layers = []
        input_shape = (state_dim + n_agents,)
        for h in hidden_sizes:
            mlp, input_shape = mlp_block(input_shape[0], h, normalize, activation, initializer, device)
            layers.extend(mlp)
        layers.extend(mlp_block(input_shape[0], action_dim, None, None, initializer, device=device)[0])
        self.pi_logits = tk.Sequential(layers)
        self.dist = CategoricalDistribution(action_dim)

    def call(self, x: tf.Tensor, **kwargs):
        self.dist.set_param(self.pi_logits(x))
        return self.pi_logits(x)


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

    def call(self, x: tf.Tensor, **kwargs):
        return self.model(x)[:, :, 0]


class COMA_CriticNet(tk.Model):
    def __init__(self,
                 state_dim: int,
                 act_dim: int,
                 hidden_sizes: Sequence[int],
                 normalize: Optional[tk.layers.Layer] = None,
                 initializer: Optional[tk.initializers.Initializer] = None,
                 activation: Optional[tk.layers.Layer] = None,
                 device: Optional[Union[str, int, torch.device]] = None):
        super(COMA_CriticNet, self).__init__()
        layers = []
        input_shape = (state_dim,)
        for h in hidden_sizes:
            mlp, input_shape = mlp_block(input_shape[0], h, normalize, activation, initializer, device)
            layers.extend(mlp)
        layers.extend(mlp_block(input_shape[0], act_dim, None, None, None, device)[0])
        self.model = tk.Sequential(layers)

    def call(self, x: tf.Tensor, **kwargs):
        return self.model(x)


class MAAC_Policy(tk.Model):
    """
    MAAC_Policy: Multi-Agent Actor-Critic Policy
    """
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
                 device: Optional[Union[str, int, torch.device]] = None,
                 **kwargs):
        super(MAAC_Policy, self).__init__()
        self.device = device
        self.action_dim = action_space.n
        self.n_agents = n_agents
        self.representation = representation[0]
        self.representation_critic = representation[1]
        self.representation_info_shape = self.representation.output_shapes
        self.lstm = True if kwargs["rnn"] == "LSTM" else False
        self.use_rnn = True if kwargs["use_recurrent"] else False
        self.actor = ActorNet(representation.output_shapes['state'][0], self.action_dim, n_agents,
                              actor_hidden_size, normalize, initializer, kwargs['gain'], activation, device)
        self.critic = CriticNet(representation.output_shapes['state'][0], n_agents, critic_hidden_size,
                                normalize, initializer, activation, device)
        self.mixer = mixer
        if mixer is None:
            self.parameters_train = self.actor.trainable_variables + self.critic.trainable_variables
        else:
            self.parameters_train = self.actor.trainable_variables + self.critic.trainable_variables + self.mixer.trainable_variables
        if not isinstance(self.representation, Basic_Identical):
            self.parameters_train += self.representation.trainable_variables
        self.dist = CategoricalDistribution(self.action_dim)

    def call(self, inputs: Union[np.ndarray, dict], *rnn_hidden, **kwargs):
        observation = inputs['obs']
        agent_ids = inputs['ids']
        if self.use_rnn:
            outputs = self.representation(observation, *rnn_hidden)
            rnn_hidden = (outputs['rnn_hidden'], outputs['rnn_cell'])
        else:
            outputs = self.representation(observation)
            rnn_hidden = (outputs['rnn_hidden'], outputs['rnn_cell'])
        actor_input = tf.concat([outputs['state'], agent_ids], axis=-1)
        act_logits = self.actor(actor_input)
        if ('avail_actions' in kwargs.keys()) and (kwargs['avail_actions'] is not None):
            avail_actions = tf.convert_to_tensor(kwargs['avail_actions'])
            act_logits[avail_actions == 0] = -1e10
            self.pi_dist.set_param(logits=act_logits)
        else:
            self.pi_dist.set_param(logits=act_logits)
        return rnn_hidden, self.pi_dist

    def get_values(self, critic_in: tf.Tensor, agent_ids: tf.Tensor, *rnn_hidden: tf.Tensor):
        shape_obs = critic_in.shape
        # get representation features
        if self.use_rnn:
            batch_size, n_agent, episode_length, dim_obs = tuple(shape_obs)
            outputs = self.representation_critic(critic_in.reshape(-1, episode_length, dim_obs), *rnn_hidden)
            outputs['state'] = outputs['state'].view(batch_size, n_agent, episode_length, -1)
            rnn_hidden = (outputs['rnn_hidden'], outputs['rnn_cell'])
        else:
            batch_size, n_agent, dim_obs = tuple(shape_obs)
            outputs = self.representation_critic(critic_in.reshape(-1, dim_obs))
            outputs['state'] = outputs['state'].view(batch_size, n_agent, -1)
            rnn_hidden = None
        # get critic values
        critic_in = tf.concat([outputs['state'], agent_ids], axis=-1)
        v = self.critic(critic_in)
        return rnn_hidden, v

    def value_tot(self, values_n: tf.Tensor, global_state=None):
        if global_state is not None:
            with tf.device(self.device):
                global_state = tf.convert_to_tensor(global_state)
        return values_n if self.mixer is None else self.mixer(values_n, global_state)


class MAAC_Policy_Share(MAAC_Policy):
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
                 device: Optional[Union[str, int, torch.device]] = None,
                 **kwargs):
        super(MAAC_Policy, self).__init__()
        self.device = device
        self.action_dim = action_space.n
        self.n_agents = n_agents
        self.lstm = True if kwargs["rnn"] == "LSTM" else False
        self.use_rnn = True if kwargs["use_recurrent"] else False
        self.representation = representation
        self.representation_info_shape = self.representation.output_shapes
        self.actor = ActorNet(self.representation.output_shapes['state'][0], self.action_dim, n_agents,
                              actor_hidden_size, normalize, initialize, kwargs['gain'], activation, device)
        self.critic = CriticNet(self.representation.output_shapes['state'][0], n_agents, critic_hidden_size,
                                normalize, initialize, activation, device)
        self.mixer = mixer
        if mixer is None:
            self.parameters_train = self.actor.trainable_variables + self.critic.trainable_variables
        else:
            self.parameters_train = self.actor.trainable_variables + self.critic.trainable_variables + self.mixer.trainable_variables
        if not isinstance(self.representation, Basic_Identical):
            self.parameters_train += self.representation.trainable_variables
        self.dist = CategoricalDistribution(self.action_dim)
        self.pi_dist = CategoricalDistribution(self.action_dim)

    def call(self, inputs: Union[np.ndarray, dict], *rnn_hidden, **kwargs):
        observation = inputs['obs']
        agent_ids = inputs['ids']
        batch_size = len(observation)
        if self.use_rnn:
            sequence_length = observation.shape[1]
            outputs = self.representation(observation, *rnn_hidden)
            rnn_hidden = (outputs['rnn_hidden'], outputs['rnn_cell'])
            representated_state = outputs['state'].view(batch_size, self.n_agents, sequence_length, -1)
            actor_critic_input = torch.concat([representated_state, agent_ids], dim=-1)
        else:
            outputs = self.representation(observation)
            rnn_hidden = None
            actor_critic_input = torch.concat([outputs['state'], agent_ids], dim=-1)
        act_logits = self.actor(actor_critic_input)
        if ('avail_actions' in kwargs.keys()) and (kwargs['avail_actions'] is not None):
            avail_actions = torch.Tensor(kwargs['avail_actions'])
            act_logits[avail_actions == 0] = -1e10
            self.pi_dist.set_param(logits=act_logits)
        else:
            self.pi_dist.set_param(logits=act_logits)

        values_independent = self.critic(actor_critic_input)
        if self.use_rnn:
            if self.mixer is None:
                values_tot = values_independent
            else:
                sequence_length = observation.shape[1]
                values_independent = values_independent.transpose(1, 2).reshape(batch_size * sequence_length,
                                                                                self.n_agents)
                values_tot = self.value_tot(values_independent, global_state=kwargs['state'])
                values_tot = values_tot.reshape([batch_size, sequence_length, 1])
                values_tot = values_tot.unsqueeze(1).expand(-1, self.n_agents, -1, -1)
        else:
            values_tot = values_independent if self.mixer is None else self.value_tot(values_independent,
                                                                                      global_state=kwargs['state'])
            values_tot = values_tot.unsqueeze(1).expand(-1, self.n_agents, -1)

        return rnn_hidden, self.pi_dist, values_tot

    def values(self, values_n: tf.Tensor, global_state=None):
        if global_state is not None:
            global_state = torch.as_tensor(global_state).to(self.device)
        return values_n if self.mixer is None else self.mixer(values_n, global_state)


class COMAPolicy(tk.Model):
    def __init__(self,
                 action_space: Discrete,
                 n_agents: int,
                 representation: Optional[Basic_Identical],
                 actor_hidden_size: Sequence[int] = None,
                 critic_hidden_size: Sequence[int] = None,
                 normalize: Optional[tk.layers.Layer] = None,
                 initializer: Optional[tk.initializers.Initializer] = None,
                 activation: Optional[tk.layers.Layer] = None,
                 device: Optional[Union[str, int, torch.device]] = None,
                 **kwargs):
        super(COMAPolicy, self).__init__()
        self.device = device
        self.action_dim = action_space.n
        self.n_agents = n_agents
        self.representation = representation
        self.representation_info_shape = self.representation.output_shapes
        self.lstm = True if kwargs["rnn"] == "LSTM" else False
        self.use_rnn = True if kwargs["use_recurrent"] else False
        self.actor = ActorNet(representation.output_shapes['state'][0], self.action_dim, n_agents,
                              actor_hidden_size, normalize, initializer, kwargs['gain'], activation, device)
        critic_input_dim = kwargs['dim_obs'] + self.action_dim * self.n_agents
        if kwargs["use_global_state"]:
            critic_input_dim += kwargs["dim_state"]
        self.critic = COMA_CriticNet(critic_input_dim, self.action_dim, critic_hidden_size,
                                     normalize, initializer, activation, device)
        self.target_critic = COMA_CriticNet(critic_input_dim, self.action_dim, critic_hidden_size,
                                            normalize, initializer, activation, device)
        self.parameters_critic = self.critic.trainable_variables
        self.pi_dist = CategoricalDistribution(self.action_dim)

    def call(self, inputs: Union[np.ndarray, dict], *rnn_hidden, **kwargs):
        observation = inputs['obs']
        agent_ids = inputs['ids']
        obs_shape = observation.shape
        if self.use_rnn:
            outputs = self.representation(observation, *rnn_hidden)
            rnn_hidden = (outputs['rnn_hidden'], outputs['rnn_cell'])
        else:
            observation_reshape = tf.reshape(observation, [-1, obs_shape[-1]])
            outputs = self.representation(observation_reshape)
            outputs_state = tf.reshape(outputs['state'], obs_shape[:-1] + self.representation_info_shape['state'])
            rnn_hidden = None
        actor_input = tf.concat([outputs_state, agent_ids], axis=-1)
        act_logits = self.actor(actor_input)
        act_probs = tf.nn.softmax(act_logits, axis=-1)
        act_probs = (1 - kwargs['epsilon']) * act_probs + kwargs['epsilon'] * 1 / self.action_dim
        if ('avail_actions' in kwargs.keys()) and (kwargs['avail_actions'] is not None):
            avail_actions = tf.Tensor(kwargs['avail_actions'])
            act_probs[avail_actions == 0] = 0.0
        return rnn_hidden, act_probs

    def get_values(self, critic_in: tf.Tensor, *rnn_hidden: tf.Tensor, target=False):
        # get critic values
        v = self.target_critic(critic_in) if target else self.critic(critic_in)
        return [None, None], v

    def param_actor(self):
        if isinstance(self.representation, Basic_Identical):
            return self.actor.trainable_variables
        else:
            return self.representation.trainable_variables + self.actor.trainable_variables

    def copy_target(self):
        self.target_critic.set_weights(self.critic.get_weights())


class MeanFieldActorCriticPolicy(tk.Model):
    def __init__(self,
                 action_space: Discrete,
                 n_agents: int,
                 representation: tk.Model,
                 actor_hidden_size: Sequence[int] = None,
                 critic_hidden_size: Sequence[int] = None,
                 normalize: Optional[tk.layers.Layer] = None,
                 initializer: Optional[tk.initializers.Initializer] = None,
                 activation: Optional[tk.layers.Layer] = None,
                 device: Optional[Union[str, int, torch.device]] = None,
                 **kwargs):
        super(MeanFieldActorCriticPolicy, self).__init__()
        self.action_dim = action_space.n
        self.representation = representation
        self.representation_info_shape = self.representation.output_shapes
        self.actor_net = ActorNet(representation.output_shapes['state'][0], self.action_dim, n_agents,
                                  actor_hidden_size, normalize, initializer, kwargs['gain'], activation, device)
        self.critic_net = CriticNet(representation.output_shapes['state'][0] + self.action_dim, n_agents,
                                    critic_hidden_size, normalize, initializer, activation, device)
        self.target_actor_net = ActorNet(representation.output_shapes['state'][0], self.action_dim, n_agents,
                                         actor_hidden_size, normalize, initializer, kwargs['gain'], activation, device)
        self.target_critic_net = CriticNet(representation.output_shapes['state'][0] + self.action_dim, n_agents,
                                           critic_hidden_size, normalize, initializer, activation, device)
        if isinstance(self.representation, Basic_Identical):
            self.parameters_actor = self.actor_net.trainable_variables
        else:
            self.parameters_actor = self.actor_net.trainable_variables + self.representation.trainable_variables
        self.parameters_critic = self.critic_net.trainable_variables
        self.pi_dist = CategoricalDistribution(self.action_dim)

    def call(self, inputs: Union[np.ndarray, dict], **kwargs):
        observations = inputs['obs']
        IDs = inputs['ids']
        outputs = self.representation(observations)
        input_actor = tf.concat([outputs['state'], IDs], axis=-1)
        act_logits = self.actor_net(input_actor)
        self.pi_dist.set_param(logits=act_logits)
        return outputs, self.pi_dist

    def critic(self, observation: tf.Tensor, actions_mean: tf.Tensor, agent_ids: tf.Tensor):
        outputs = self.representation(observation)
        critic_in = tf.concat([outputs['state'], actions_mean, agent_ids], axis=-1)
        return self.critic_net(critic_in)
