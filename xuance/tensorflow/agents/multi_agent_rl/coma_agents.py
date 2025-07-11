from argparse import Namespace
from xuance.common import List, Optional, Union
from xuance.environment import DummyVecMultiAgentEnv, SubprocVecMultiAgentEnv
from xuance.tensorflow import Module
from xuance.tensorflow.utils import NormalizeFunctions, ActivationFunctions, InitializeFunctions
from xuance.tensorflow.policies import REGISTRY_Policy
from xuance.tensorflow.agents import OnPolicyMARLAgents, BaseCallback


class COMA_Agents(OnPolicyMARLAgents):
    """The implementation of COMA agents.

    Args:
        config: the Namespace variable that provides hyperparameters and other settings.
        envs: the vectorized environments.
        callback: A user-defined callback function object to inject custom logic during training.
    """

    def __init__(self,
                 config: Namespace,
                 envs: Union[DummyVecMultiAgentEnv, SubprocVecMultiAgentEnv],
                 callback: Optional[BaseCallback] = None):
        super(COMA_Agents, self).__init__(config, envs, callback)
        self.start_greedy, self.end_greedy = config.start_greedy, config.end_greedy
        self.egreedy = self.start_greedy
        self.delta_egreedy = (self.start_greedy - self.end_greedy) / config.decay_step_greedy

        self.use_global_state = True
        self.continuous_control = False
        self.state_space = envs.state_space

        self.policy = self._build_policy()  # build policy
        self.memory = self._build_memory()  # build memory
        self.learner = self._build_learner(self.config, self.model_keys, self.agent_keys, self.policy, self.callback)
        self.learner.egreedy = self.egreedy

    def _build_policy(self) -> Module:
        """
        Build representation(s) and policy(ies) for agent(s)

        Returns:
            policy (torch.nn.Module): A dict of policies.
        """
        normalize_fn = NormalizeFunctions[self.config.normalize] if hasattr(self.config, "normalize") else None
        initializer = InitializeFunctions[self.config.initialize] if hasattr(self.config, "initialize") else None
        activation = ActivationFunctions[self.config.activation]

        # build representations
        A_representation = self._build_representation(self.config.representation, self.observation_space, self.config)
        C_representation = self._build_representation(self.config.representation, self.observation_space, self.config)

        # build policies
        if self.config.policy == "Categorical_COMA_Policy":
            policy = REGISTRY_Policy["Categorical_COMA_Policy"](
                action_space=self.action_space, n_agents=self.n_agents,
                representation_actor=A_representation, representation_critic=C_representation,
                actor_hidden_size=self.config.actor_hidden_size, critic_hidden_size=self.config.critic_hidden_size,
                normalize=normalize_fn, initialize=initializer, activation=activation,
                use_parameter_sharing=self.use_parameter_sharing, model_keys=self.model_keys,
                use_rnn=self.use_rnn, rnn=self.config.rnn if self.use_rnn else None,
                dim_global_state=self.state_space.shape[0])
        else:
            raise AttributeError(f"COMA currently does not support the policy named {self.config.policy}.")

        return policy

    def act(self, obs_n, *rnn_hidden, avail_actions=None, test_mode=False):
        batch_size = len(obs_n)
        with tf.device(self.device):
            # build critic input
            agents_id = tf.tile(tf.expand_dims(tf.eye(self.n_agents), axis=0), multiples=(batch_size, 1, 1))
            inputs_policy = {"obs": tf.convert_to_tensor(obs_n), "ids": agents_id}
            epsilon = 0.0 if test_mode else self.egreedy
            if self.use_rnn:
                batch_agents = batch_size * self.n_agents
                hidden_state, _ = self.policy(inputs_policy,
                                              *rnn_hidden,
                                              avail_actions=avail_actions.reshape(batch_agents, 1, -1),
                                              epsilon=epsilon)
            else:
                hidden_state, _ = self.policy(inputs_policy,
                                              avail_actions=avail_actions,
                                              epsilon=epsilon)
        dists = self.policy.actor.dist
        picked_actions = dists.stochastic_sample()
        onehot_actions = self.learner.onehot_action(picked_actions, self.dim_act)
        return hidden_state, picked_actions.numpy(), onehot_actions.numpy()

    def values(self, obs_n, *rnn_hidden, state=None, actions_n=None, actions_onehot=None):
        batch_size = len(obs_n)
        # build critic input
        obs_n = tf.convert_to_tensor(obs_n)
        actions_n = tf.expand_dims(tf.convert_to_tensor(actions_n), axis=-1)
        actions_in = tf.expand_dims(tf.convert_to_tensor(actions_onehot), 1)
        actions_in = tf.repeat(tf.reshape(actions_in, [batch_size, 1, -1]), self.n_agents, axis=1)
        agent_mask = 1 - tf.eye(self.n_agents)
        agent_mask = tf.reshape(tf.repeat(tf.reshape(agent_mask, [-1, 1]), self.dim_act, axis=1), [self.n_agents, -1])
        actions_in = actions_in * tf.expand_dims(agent_mask, 0)
        if self.use_global_state:
            state = tf.repeat(tf.expand_dims(tf.convert_to_tensor(state), 1), self.n_agents, axis=1)
            critic_in = tf.concat([state, obs_n, actions_in], axis=-1)
        else:
            critic_in = tf.concat([obs_n, actions_in], axis=-1)
        # get critic values
        hidden_state, values_n = self.policy.get_values(critic_in, target=True)

        target_values = tf.gather(values_n, actions_n, axis=-1, batch_dims=-1)
        return hidden_state, target_values.numpy()

    def train(self, i_step, **kwargs):
        if self.egreedy >= self.end_greedy:
            self.egreedy = self.start_greedy - self.delta_egreedy * i_step
        info_train = {}
        if self.memory.full:
            indexes = np.arange(self.buffer_size)
            for _ in range(self.n_epochs):
                np.random.shuffle(indexes)
                for start in range(0, self.buffer_size, self.batch_size):
                    end = start + self.batch_size
                    sample_idx = indexes[start:end]
                    sample = self.memory.sample(sample_idx)
                    if self.use_rnn:
                        info_train = self.learner.update_recurrent(sample, self.egreedy)
                    else:
                        info_train = self.learner.update(sample, self.egreedy)
            self.memory.clear()
        info_train["epsilon-greedy"] = self.egreedy
        return info_train
