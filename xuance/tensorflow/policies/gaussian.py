from xuance.tensorflow.policies import *
from xuance.tensorflow.utils import *
from xuance.tensorflow.representations import Basic_Identical
import tensorflow_probability as tfp

tfd = tfp.distributions


class ActorNet(tk.Model):
    def __init__(self,
                 state_dim: int,
                 action_dim: int,
                 hidden_sizes: Sequence[int],
                 normalize: Optional[tk.layers.Layer] = None,
                 initializer: Optional[tk.initializers.Initializer] = None,
                 activation: Optional[tk.layers.Layer] = None,
                 device: str = "cpu:0"):
        super(ActorNet, self).__init__()
        layers = []
        input_shapes = (state_dim,)
        for h in hidden_sizes:
            mlp, input_shapes = mlp_block(input_shapes[0], h, normalize, activation, initializer, device)
            layers.extend(mlp)
        layers.extend(mlp_block(input_shapes[0], action_dim, device=device)[0])
        self.mu_model = tk.Sequential(layers)
        self.logstd = tf.Variable(tf.zeros((action_dim,)) - 1, trainable=True)
        self.dist = DiagGaussianDistribution(action_dim)

    def call(self, x: tf.Tensor, **kwargs):
        self.dist.set_param(self.mu_model(x), tf.math.exp(self.logstd))
        return self.mu_model(x)


class CriticNet(tk.Model):
    def __init__(self,
                 state_dim: int,
                 hidden_sizes: Sequence[int],
                 normalize: Optional[tk.layers.Layer] = None,
                 initializer: Optional[tk.initializers.Initializer] = None,
                 activation: Optional[tk.layers.Layer] = None,
                 device: str = "cpu:0"):
        super(CriticNet, self).__init__()
        layers = []
        input_shapes = (state_dim,)
        for h in hidden_sizes:
            mlp, input_shapes = mlp_block(input_shapes[0], h, normalize, activation, initializer, device)
            layers.extend(mlp)
        layers.extend(mlp_block(input_shapes[0], 1, device=device)[0])
        self.model = tk.Sequential(layers)

    def call(self, x: tf.Tensor, **kwargs):
        return self.model(x)[:, 0]


class ActorCriticPolicy(tk.Model):
    def __init__(self,
                 action_space: Space,
                 representation: tk.Model,
                 actor_hidden_size: Sequence[int] = None,
                 critic_hidden_size: Sequence[int] = None,
                 normalize: Optional[tk.layers.Layer] = None,
                 initializer: Optional[tk.initializers.Initializer] = None,
                 activation: Optional[tk.layers.Layer] = None,
                 device: str = "cpu:0"):
        super(ActorCriticPolicy, self).__init__()
        self.action_dim = action_space.shape[0]
        self.representation = representation
        self.representation_info_shape = self.representation.output_shapes
        self.actor = ActorNet(representation.output_shapes['state'][0], self.action_dim, actor_hidden_size,
                              normalize, initializer, activation, device)
        self.critic = CriticNet(representation.output_shapes['state'][0], critic_hidden_size,
                                normalize, initializer, activation, device)

    def call(self, observations: Union[np.ndarray, dict], **kwargs):
        outputs = self.representation(observations)
        a = self.actor(outputs['state'])
        v = self.critic(outputs['state'])
        return outputs, a, v


class ActorPolicy(tk.Model):
    def __init__(self,
                 action_space: Space,
                 representation: tk.Model,
                 actor_hidden_size: Sequence[int] = None,
                 normalize: Optional[tk.layers.Layer] = None,
                 initializer: Optional[tk.initializers.Initializer] = None,
                 activation: Optional[tk.layers.Layer] = None,
                 device: str = "cpu:0",
                 fixed_std: bool = True):
        super(ActorPolicy, self).__init__()
        self.action_dim = action_space.shape[0]
        self.representation = representation
        self.representation_info_shape = self.representation.output_shapes
        self.actor = ActorNet(representation.output_shapes['state'][0], self.action_dim, actor_hidden_size,
                              normalize, initializer, activation, device)

    def call(self, observation: Union[np.ndarray, dict], **kwargs):
        outputs = self.representation(observation)
        a = self.actor(outputs['state'])
        return outputs, a


class PPGActorCritic(tk.Model):
    def __init__(self,
                 action_space: Space,
                 representation: tk.Model,
                 actor_hidden_size: Sequence[int] = None,
                 critic_hidden_size: Sequence[int] = None,
                 normalize: Optional[tk.layers.Layer] = None,
                 initializer: Optional[tk.initializers.Initializer] = None,
                 activation: Optional[tk.layers.Layer] = None,
                 device: str = "cpu:0"):
        super(PPGActorCritic, self).__init__()
        self.action_dim = action_space.shape[0]
        self.actor_representation = representation
        self.critic_representation = copy.deepcopy(representation)
        self.representation_info_shape = self.actor_representation.output_shapes
        self.actor = ActorNet(representation.output_shapes['state'][0], self.action_dim, actor_hidden_size,
                              normalize, initializer, activation, device)
        self.critic = CriticNet(representation.output_shapes['state'][0], critic_hidden_size,
                                normalize, initializer, activation, device)
        self.aux_critic = CriticNet(representation.output_shapes['state'][0], critic_hidden_size,
                                    normalize, initializer, activation, device)

    def call(self, observation: Union[np.ndarray, dict], **kwargs):
        policy_outputs = self.actor_representation(observation)
        critic_outputs = self.critic_representation(observation)
        a = self.actor(policy_outputs['state'])
        v = self.critic(critic_outputs['state'])
        aux_v = self.aux_critic(policy_outputs)
        return policy_outputs, a, v, aux_v


class ActorNet_SAC(tk.Model):
    def __init__(self,
                 state_dim: int,
                 action_dim: int,
                 hidden_sizes: Sequence[int],
                 initializer: Optional[tk.initializers.Initializer] = None,
                 activation: Optional[tk.layers.Layer] = None,
                 device: str = "cpu:0"):
        super(ActorNet_SAC, self).__init__()
        layers = []
        input_shape = (state_dim,)
        for h in hidden_sizes:
            mlp, input_shape = mlp_block(input_shape[0], h, None, activation, initializer, device)
            layers.extend(mlp)
        self.device = device
        self.outputs = tk.Sequential(layers)
        self.out_mu = tk.layers.Dense(units=action_dim,
                                      input_shape=(hidden_sizes[0],))
        self.out_std = tk.layers.Dense(units=action_dim,
                                       input_shape=(hidden_sizes[0],))

    def call(self, x: tf.Tensor, **kwargs):
        output = self.outputs(x)
        mu = tf.tanh(self.out_mu(output))
        std = tf.clip_by_value(self.out_std(output), -20, 2)
        std = tf.exp(std)
        return tfd.Normal(mu, std)
        # self.dist = tfd.Normal(mu, std)
        # return mu, std


class CriticNet_SAC(tk.Model):
    def __init__(self,
                 state_dim: int,
                 action_dim: int,
                 hidden_sizes: Sequence[int],
                 initializer: Optional[tk.initializers.Initializer] = None,
                 activation: Optional[tk.layers.Layer] = None,
                 device: str = "cpu:0"):
        super(CriticNet_SAC, self).__init__()
        layers = []
        input_shape = (state_dim + action_dim,)
        for h in hidden_sizes:
            mlp, input_shape = mlp_block(input_shape[0], h, None, activation, initializer, device)
            layers.extend(mlp)
        layers.extend(mlp_block(input_shape[0], 1, None, None, initializer, device)[0])
        self.model = tk.Sequential(layers)

    def call(self, inputs: Union[np.ndarray, dict], **kwargs):
        obs = inputs['obs']
        act = inputs['act']
        return self.model(tf.concat((obs, act), axis=-1))


class SACPolicy(tk.Model):
    def __init__(self,
                 action_space: Space,
                 representation: Basic_Identical,
                 actor_hidden_size: Sequence[int],
                 critic_hidden_size: Sequence[int],
                 initializer: Optional[tk.initializers.Initializer] = None,
                 activation: Optional[tk.layers.Layer] = None,
                 device: str = "cpu:0"):
        assert isinstance(action_space, Box)
        super(SACPolicy, self).__init__()
        self.action_dim = action_space.shape[0]
        self.representation = representation
        self.representation_info_shape = self.representation.output_shapes

        self.actor = ActorNet_SAC(representation.output_shapes['state'][0], self.action_dim, actor_hidden_size,
                                  initializer, activation, device)
        self.critic = CriticNet_SAC(representation.output_shapes['state'][0], self.action_dim, critic_hidden_size,
                                    initializer, activation, device)
        self.target_actor = ActorNet_SAC(representation.output_shapes['state'][0], self.action_dim, actor_hidden_size,
                                         initializer, activation, device)
        self.target_critic = CriticNet_SAC(representation.output_shapes['state'][0], self.action_dim,
                                           critic_hidden_size,
                                           initializer, activation, device)
        self.soft_update(tau=1.0)

    def action(self, observation: Union[np.ndarray, dict]):
        outputs = self.representation(observation)
        dist = self.actor(outputs['state'])

        return outputs, dist

    def Qtarget(self, observation: Union[np.ndarray, dict]):
        outputs = self.representation(observation)
        act_dist = self.target_actor(outputs['state'])
        act = act_dist.sample()
        act_log = act_dist.log_prob(act)
        inputs = {'obs': outputs['state'], 'act': act}
        return outputs, act_log, self.target_critic(inputs)

    def Qaction(self, observation: Union[np.ndarray, dict], action: tf.Tensor):
        outputs = self.representation(observation)
        inputs = {'obs': outputs['state'], 'act': action}
        return outputs, self.critic(inputs)

    def Qpolicy(self, observation: Union[np.ndarray, dict]):
        outputs = self.representation(observation)
        act_dist = self.actor(outputs['state'])
        act = act_dist.sample()
        act_log = act_dist.log_prob(act)
        inputs = {'obs': outputs['state'], 'act': act}
        return outputs, act_log, self.critic(inputs)

    def soft_update(self, tau=0.005):
        for ep, tp in zip(self.actor.variables, self.target_actor.variables):
            tp.assign((1 - tau) * tp + tau * ep)
        for ep, tp in zip(self.critic.variables, self.target_critic.variables):
            tp.assign((1 - tau) * tp + tau * ep)
