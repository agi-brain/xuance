from xuanpolicy.mindspore.policies import *
from xuanpolicy.mindspore.utils import *
from xuanpolicy.mindspore.representations import Basic_Identical
from .deterministic_marl import BasicQhead
from mindspore.nn.probability.distribution import Categorical
import copy


class ActorNet(nn.Cell):
    class Sample(nn.Cell):
        def __init__(self):
            super(ActorNet.Sample, self).__init__()
            self._dist = Categorical(dtype=ms.float32)

        def construct(self, probs: ms.tensor):
            return self._dist.sample(probs=probs).astype("int32")

    class LogProb(nn.Cell):
        def __init__(self):
            super(ActorNet.LogProb, self).__init__()
            self._dist = Categorical(dtype=ms.float32)

        def construct(self, value, probs):
            return self._dist.log_prob(value=value, probs=probs)

    class Entropy(nn.Cell):
        def __init__(self):
            super(ActorNet.Entropy, self).__init__()
            self._dist = Categorical(dtype=ms.float32)

        def construct(self, probs):
            return self._dist.entropy(probs=probs)

    class KL_Div(nn.Cell):
        def __init__(self):
            super(ActorNet.KL_Div, self).__init__()
            self._dist = Categorical(dtype=ms.float32)

        def construct(self, probs_p, probs_q):
            return self._dist.kl_loss('Categorical', probs_p, probs_q)

    def __init__(self,
                 state_dim: int,
                 action_dim: int,
                 n_agents: int,
                 hidden_sizes: Sequence[int],
                 normalize: Optional[ModuleType] = None,
                 initialize: Optional[Callable[..., ms.Tensor]] = None,
                 activation: Optional[ModuleType] = None):
        super(ActorNet, self).__init__()
        layers = []
        input_shape = (state_dim + n_agents,)
        for h in hidden_sizes:
            mlp, input_shape = mlp_block(input_shape[0], h, normalize, activation, initialize)
            layers.extend(mlp)
        layers.extend(mlp_block(input_shape[0], action_dim, None, nn.Softmax, None)[0])
        self.model = nn.SequentialCell(*layers)
        self.sample = self.Sample()
        self.log_prob = self.LogProb()
        self.entropy = self.Entropy()
        self.kl_div = self.KL_Div()

    def construct(self, x: ms.Tensor):
        return self.model(x)


class CriticNet(nn.Cell):
    def __init__(self,
                 state_dim: int,
                 n_agents: int,
                 hidden_sizes: Sequence[int],
                 normalize: Optional[ModuleType] = None,
                 initialize: Optional[Callable[..., ms.Tensor]] = None,
                 activation: Optional[ModuleType] = None):
        super(CriticNet, self).__init__()
        layers = []
        input_shape = (state_dim + n_agents,)
        for h in hidden_sizes:
            mlp, input_shape = mlp_block(input_shape[0], h, normalize, activation, initialize)
            layers.extend(mlp)
        layers.extend(mlp_block(input_shape[0], 1, None, None, None)[0])
        self.model = nn.SequentialCell(*layers)

    def construct(self, x: ms.Tensor):
        return self.model(x)[:, :, 0]


class COMA_CriticNet(nn.Cell):
    def __init__(self,
                 state_dim: int,
                 obs_dim: int,
                 act_dim: int,
                 n_agents: int,
                 hidden_sizes: Sequence[int],
                 normalize: Optional[ModuleType] = None,
                 initialize: Optional[Callable[..., ms.Tensor]] = None,
                 activation: Optional[ModuleType] = None):
        super(COMA_CriticNet, self).__init__()
        layers = []
        input_shape = (state_dim + obs_dim + act_dim * n_agents + n_agents,)
        for h in hidden_sizes:
            mlp, input_shape = mlp_block(input_shape[0], h, normalize, activation, initialize)
            layers.extend(mlp)
        layers.extend(mlp_block(input_shape[0], act_dim, None, None, None)[0])
        self.model = nn.SequentialCell(*layers)

    def construct(self, x: ms.Tensor):
        return self.model(x)


class MultiAgentActorCriticPolicy(nn.Cell):
    def __init__(self,
                 action_space: Discrete,
                 n_agents: int,
                 representation: Optional[Basic_Identical],
                 mixer: Optional[VDN_mixer] = None,
                 actor_hidden_size: Sequence[int] = None,
                 critic_hidden_size: Sequence[int] = None,
                 normalize: Optional[ModuleType] = None,
                 initialize: Optional[Callable[..., ms.Tensor]] = None,
                 activation: Optional[ModuleType] = None):
        assert isinstance(action_space, Discrete)
        super(MultiAgentActorCriticPolicy, self).__init__()
        self.action_dim = action_space.n
        self.representation = representation
        self.representation_info_shape = self.representation.output_shapes
        self.actor = ActorNet(representation.output_shapes['state'][0], self.action_dim, n_agents,
                              actor_hidden_size, normalize, initialize, activation)
        self.critic = CriticNet(representation.output_shapes['state'][0], n_agents, critic_hidden_size,
                                normalize, initialize, activation)
        self.mixer = mixer
        self._concat = ms.ops.Concat(axis=-1)
        self.expand_dims = ms.ops.ExpandDims()

    def construct(self, observation: ms.Tensor, agent_ids: ms.Tensor):
        outputs = self.representation(observation)
        input_with_id = self._concat([outputs['state'], agent_ids])
        act_dist = self.actor(input_with_id)
        v = self.expand_dims(self.critic(input_with_id), -1)
        return outputs, act_dist, v

    def value_tot(self, values_n: ms.Tensor, global_state=None):
        if global_state is not None:
            global_state = global_state
        return values_n if self.mixer is None else self.mixer(values_n, global_state)


class MAPPO_ActorCriticPolicy(MultiAgentActorCriticPolicy):
    def __init__(self,
                 dim_state: int,
                 action_space: Discrete,
                 n_agents: int,
                 representation: Optional[Basic_Identical],
                 actor_hidden_size: Sequence[int] = None,
                 critic_hidden_size: Sequence[int] = None,
                 normalize: Optional[ModuleType] = None,
                 initialize: Optional[Callable[..., ms.Tensor]] = None,
                 activation: Optional[ModuleType] = None):
        assert isinstance(action_space, Discrete)
        super(MAPPO_ActorCriticPolicy, self).__init__(action_space, n_agents, representation, None,
                                                      actor_hidden_size, critic_hidden_size,
                                                      normalize, initialize, activation)
        self.critic = CriticNet(dim_state, n_agents, critic_hidden_size, normalize, initialize, activation)
        self._concat = ms.ops.Concat(axis=-1)
        self.expand_dims = ms.ops.ExpandDims()

    def construct(self, observation: ms.Tensor, agent_ids: ms.Tensor):
        outputs = self.representation(observation)
        input_with_id = self._concat([outputs['state'], agent_ids])
        act_dist = self.actor(input_with_id)
        return outputs, act_dist

    def values(self, state: ms.Tensor, agent_ids: ms.Tensor):
        input_with_id = self._concat([state, agent_ids])
        return self.expand_dims(self.critic(input_with_id), -1)


class MeanFieldActorCriticPolicy(nn.Cell):
    def __init__(self,
                 action_space: Discrete,
                 n_agents: int,
                 representation: Optional[Basic_Identical],
                 actor_hidden_size: Sequence[int] = None,
                 critic_hidden_size: Sequence[int] = None,
                 normalize: Optional[ModuleType] = None,
                 initialize: Optional[Callable[..., ms.Tensor]] = None,
                 activation: Optional[ModuleType] = None):
        assert isinstance(action_space, Discrete)
        super(MeanFieldActorCriticPolicy, self).__init__()
        self.action_dim = action_space.n
        self.representation = representation
        self.representation_info_shape = self.representation.output_shapes
        self.actor_net = ActorNet(representation.output_shapes['state'][0], self.action_dim, n_agents,
                                  actor_hidden_size, normalize, initialize, activation)
        self.critic_net = BasicQhead(representation.output_shapes['state'][0] + self.action_dim, self.action_dim,
                                     n_agents, critic_hidden_size, normalize, initialize, activation)
        self.target_actor_net = ActorNet(representation.output_shapes['state'][0], self.action_dim, n_agents,
                                         actor_hidden_size, normalize, initialize, activation)
        for ep, tp in zip(self.actor_net.trainable_params(), self.target_actor_net.trainable_params()):
            tp.assign_value(ep)
        self.target_critic_net = copy.deepcopy(self.critic_net)
        self.parameters_actor = self.actor_net.trainable_params() + self.representation.trainable_params()
        self.parameters_critic = self.critic_net.trainable_params()
        self._concat = ms.ops.Concat(axis=-1)

    def construct(self, observation: ms.Tensor, agent_ids: ms.Tensor):
        outputs = self.representation(observation)
        input_actor = self._concat([outputs['state'], agent_ids])
        act_dist = self.actor_net(input_actor)
        return outputs, act_dist

    def target_actor(self, observation: ms.Tensor, agent_ids: ms.Tensor):
        outputs = self.representation(observation)
        input_actor = self._concat([outputs[0], agent_ids])
        act_dist = self.target_actor_net(input_actor)
        return act_dist

    def critic(self, observation: ms.Tensor, actions_mean: ms.Tensor, agent_ids: ms.Tensor):
        outputs = self.representation(observation)
        critic_in = self._concat([outputs['state'], actions_mean, agent_ids])
        return self.critic_net(critic_in)

    def target_critic(self, observation: ms.Tensor, actions_mean: ms.Tensor, agent_ids: ms.Tensor):
        outputs = self.representation(observation)
        critic_in = self._concat([outputs[0], actions_mean, agent_ids])
        return self.target_critic_net(critic_in)

    def target_critic_for_train(self, observation: ms.Tensor, actions_mean: ms.Tensor, agent_ids: ms.Tensor):
        outputs = self.representation(observation)
        critic_in = self._concat([outputs['state'], actions_mean, agent_ids])
        return self.target_critic_net(critic_in)

    def soft_update(self, tau=0.005):
        for ep, tp in zip(self.actor_net.trainable_params(), self.target_actor_net.trainable_params()):
            tp.assign_value((tau*ep.data+(1-tau)*tp.data))
        for ep, tp in zip(self.critic_net.trainable_params(), self.target_critic_net.trainable_params()):
            tp.assign_value((tau*ep.data+(1-tau)*tp.data))


class COMAPolicy(nn.Cell):
    def __init__(self,
                 state_dim: int,
                 action_space: Discrete,
                 n_agents: int,
                 representation: Optional[Basic_Identical],
                 actor_hidden_size: Sequence[int] = None,
                 critic_hidden_size: Sequence[int] = None,
                 normalize: Optional[ModuleType] = None,
                 initialize: Optional[Callable[..., ms.Tensor]] = None,
                 activation: Optional[ModuleType] = None):
        assert isinstance(action_space, Discrete)
        super(COMAPolicy, self).__init__()
        self.action_dim = action_space.n
        self.n_agents = n_agents
        self.representation = representation
        self.representation_info_shape = self.representation.output_shapes
        self.actor = ActorNet(representation.output_shapes['state'][0], self.action_dim, n_agents,
                              actor_hidden_size, normalize, initialize, activation)
        self.critic = COMA_CriticNet(state_dim, representation.output_shapes['state'][0], self.action_dim, n_agents,
                                     critic_hidden_size, normalize, initialize, activation)
        self.target_critic = copy.deepcopy(self.critic)
        self.parameters_critic = self.critic.trainable_params()
        self.parameters_actor = self.representation.trainable_params() + self.actor.trainable_params()
        self.eye = ms.ops.Eye()
        self._concat = ms.ops.Concat(axis=-1)

    def build_critic_in(self, state, observations, actions_onehot, agent_ids, t=None):
        bs, act_dim = state.shape[0], actions_onehot.shape[-1]
        step_len = state.shape[1] if t is None else 1
        ts = slice(None) if t is None else slice(t, t + 1)
        obs_encode = self.representation(observations)[0]
        inputs = [state[:, ts], obs_encode[:, ts]]
        # counterfactual actions inputs
        actions_joint = ms.ops.broadcast_to(actions_onehot[:, ts].view(bs, step_len, 1, -1),
                                            (-1, -1, self.n_agents, -1))
        agent_mask = ms.ops.broadcast_to((1 - self.eye(self.n_agents, self.n_agents, ms.float32)).view(-1, 1),
                                         (-1, act_dim)).view(self.n_agents, -1)
        agent_mask = ms.ops.expand_dims(ms.ops.expand_dims(agent_mask, 0), 0)
        inputs.append(actions_joint * agent_mask)
        inputs.append(agent_ids[:, ts])
        return self._concat(inputs)

    def construct(self, observation: ms.Tensor, agent_ids: ms.Tensor):
        outputs = self.representation(observation)
        input_with_id = self._concat([outputs['state'], agent_ids])
        act_dist = self.actor(input_with_id)
        return outputs, act_dist

    def copy_target(self):
        for ep, tp in zip(self.critic.trainable_params(), self.target_critic.trainable_params()):
            tp.assign_value(ep)
