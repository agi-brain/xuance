from xuance.tensorflow.agents import *


class MFAC_Agents(MARLAgents):
    def __init__(self,
                 config: Namespace,
                 envs: DummyVecEnv_Pettingzoo,
                 device: str = "cpu:0"):
        self.gamma = config.gamma
        self.n_envs = envs.num_envs
        self.n_size = config.buffer_size
        self.n_epoch = config.n_epoch
        self.n_minibatch = config.n_minibatch
        if config.state_space is not None:
            config.dim_state, state_shape = config.state_space.shape, config.state_space.shape
        else:
            config.dim_state, state_shape = None, None

        input_representation = get_repre_in(config)
        representation = REGISTRY_Representation[config.representation](*input_representation)
        input_policy = get_policy_in_marl(config, representation)
        policy = REGISTRY_Policy[config.policy](*input_policy, gain=config.gain)
        lr_scheduler = MyLinearLR(config.learning_rate, start_factor=1.0, end_factor=0.5,
                                  total_iters=get_total_iters(config.agent_name, config))
        optimizer = tk.optimizers.Adam(lr_scheduler)
        self.observation_space = envs.observation_space
        self.action_space = envs.action_space
        self.representation_info_shape = policy.representation.output_shapes
        self.auxiliary_info_shape = {}

        if config.state_space is not None:
            config.dim_state, state_shape = config.state_space.shape, config.state_space.shape
        else:
            config.dim_state, state_shape = None, None
        if config.state_space is not None:
            config.dim_state, state_shape = config.state_space.shape, config.state_space.shape
        else:
            config.dim_state, state_shape = None, None
        memory = MeanField_OnPolicyBuffer(config.n_agents,
                                          state_shape,
                                          config.obs_shape,
                                          config.act_shape,
                                          config.rew_shape,
                                          config.done_shape,
                                          envs.num_envs,
                                          config.buffer_size,
                                          config.use_gae, config.use_advnorm, config.gamma, config.gae_lambda,
                                          prob_space=config.act_prob_shape)
        self.buffer_size = memory.buffer_size
        self.batch_size = self.buffer_size // self.n_minibatch
        learner = MFAC_Learner(config, policy, optimizer, config.device, config.model_dir, config.gamma)
        super(MFAC_Agents, self).__init__(config, envs, policy, memory, learner, device,
                                          config.log_dir, config.model_dir)
        self.on_policy = True

    def act(self, obs_n, test_mode, act_mean=None, agent_mask=None):
        batch_size = len(obs_n)
        inputs = {"obs": obs_n,
                  "ids": np.tile(np.expand_dims(np.eye(self.n_agents), 0), (batch_size, 1, 1))}
        _, dists = self.policy(inputs)
        acts = dists.stochastic_sample()

        n_alive = np.expand_dims(np.sum(agent_mask, axis=-1), axis=-1).repeat(self.dim_act, axis=1)
        action_n_mask = np.expand_dims(agent_mask, axis=-1).repeat(self.dim_act, axis=-1)
        act_neighbor_onehot = self.learner.onehot_action(acts, self.dim_act).numpy() * action_n_mask
        act_mean_current = np.sum(act_neighbor_onehot, axis=1) / n_alive

        return acts.numpy(), act_mean_current

    def values(self, obs, actions_mean):
        batch_size = len(obs)
        agents_id = np.tile(np.expand_dims(np.eye(self.n_agents), 0), (batch_size, 1, 1))
        agents_id = tf.convert_to_tensor(agents_id, dtype=tf.float32)
        actions_mean = tf.repeat(tf.expand_dims(tf.convert_to_tensor(actions_mean, dtype=tf.float32), 1),
                                 repeats=self.n_agents, axis=1)
        values_n = self.policy.critic(obs, actions_mean, agents_id)
        hidden_states = None
        return hidden_states, values_n.numpy()

    def train(self, i_step, **kwargs):
        if self.memory.full:
            info_train = {}
            indexes = np.arange(self.buffer_size)
            for _ in range(self.n_epoch):
                np.random.shuffle(indexes)
                for start in range(0, self.buffer_size, self.batch_size):
                    end = start + self.batch_size
                    sample_idx = indexes[start:end]
                    sample = self.memory.sample(sample_idx)
                    info_train = self.learner.update(sample)
            self.learner.lr_decay(i_step)
            self.memory.clear()
            return info_train
        else:
            return {}
