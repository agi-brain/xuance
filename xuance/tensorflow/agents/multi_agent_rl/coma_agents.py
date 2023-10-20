from xuance.tensorflow.agents import *
from xuance.tensorflow.agents.agents_marl import linear_decay_or_increase


class COMA_Agents(MARLAgents):
    def __init__(self,
                 config: Namespace,
                 envs: DummyVecEnv_Pettingzoo,
                 device: str = "cpu:0"):
        config.batch_size = config.batch_size * envs.num_envs

        self.gamma = config.gamma

        if config.state_space is not None:
            config.dim_state, state_shape = config.state_space.shape, config.state_space.shape
        else:
            config.dim_state, state_shape = None, None

        input_representation = get_repre_in(config)
        representation = REGISTRY_Representation[config.representation](*input_representation)
        input_policy = get_policy_in_marl(config, representation, config.agent_keys, None)
        policy = REGISTRY_Policy[config.policy](*input_policy)
        lr_scheduler = [MyLinearLR(config.learning_rate_actor, start_factor=1.0, end_factor=0.5,
                                   total_iters=get_total_iters(config.agent_name, config)),
                        MyLinearLR(config.learning_rate_critic, start_factor=1.0, end_factor=0.5,
                                   total_iters=get_total_iters(config.agent_name, config))]
        optimizer = [tk.optimizers.Adam(lr_scheduler[0]),
                     tk.optimizers.Adam(lr_scheduler[1])]
        self.observation_space = envs.observation_space
        self.action_space = envs.action_space
        self.representation_info_shape = policy.representation.output_shapes
        self.auxiliary_info_shape = {}

        if config.state_space is not None:
            config.dim_state, state_shape = config.state_space.shape, config.state_space.shape
        else:
            config.dim_state, state_shape = None, None
        config.act_onehot_shape = config.act_shape + tuple([config.dim_act])
        memory = COMA_Buffer(state_shape, config.obs_shape, config.act_shape, config.act_onehot_shape,
                             config.rew_shape, config.done_shape, envs.num_envs,
                             config.buffer_size, config.batch_size, envs.envs[0].max_cycles)
        learner = COMA_Learner(config, policy, optimizer,
                               config.device, config.modeldir, config.gamma, config.sync_frequency)

        self.epsilon_decay = linear_decay_or_increase(config.start_greedy, config.end_greedy,
                                                      config.greedy_update_steps)
        super(COMA_Agents, self).__init__(config, envs, policy, memory, learner, device, config.logdir, config.modeldir)

    def act(self, obs_n, episode, test_mode, noise=False):
        batch_size = len(obs_n)
        with tf.device(self.device):
            agents_id = tf.tile(tf.expand_dims(tf.eye(self.n_agents), axis=0), multiples=(batch_size, 1, 1))
            inputs_policy = {"obs": tf.convert_to_tensor(obs_n), "ids": agents_id}
            states, dists = self.policy(inputs_policy)
            # acts = dists.stochastic_sample()  # stochastic policy
            epsilon = 1.0 if test_mode else self.epsilon_decay.epsilon
            greedy_actions = tf.argmax(dists.logits, axis=-1)
        if noise:
            random_variable = np.random.random(greedy_actions.shape)
            action_pick = np.int32((random_variable < epsilon))
            random_actions = np.array([[self.args.action_space[agent].sample() for agent in self.agent_keys]])
            actions_select = action_pick * greedy_actions.numpy() + (1 - action_pick) * random_actions
            actions_onehot = self.learner.onehot_action(actions_select, self.dim_act)
            return actions_select, actions_onehot.numpy()
        else:
            actions_onehot = self.learner.onehot_action(greedy_actions, self.dim_act)
            return greedy_actions.numpy(), actions_onehot.numpy()

    def train(self, i_episode):
        self.epsilon_decay.update()
        for i in range(self.nenvs):
            self.writer.add_scalars("epsilon", {"env-%d" % i: self.epsilon_decay.epsilon}, i_episode)
        if self.memory.full:
            sample = self.memory.sample()
            self.learner.update(sample)
        # self.memory.clear()
