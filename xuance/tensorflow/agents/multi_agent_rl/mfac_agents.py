from xuance.tensorflow.agents import *


class MFAC_Agents(MARLAgents):
    def __init__(self,
                 config: Namespace,
                 envs: DummyVecEnv_MAS,
                 device: str = "cpu:0"):
        self.comm = MPI.COMM_WORLD

        self.gamma = config.gamma
        self.use_obsnorm = config.use_obsnorm
        self.use_rewnorm = config.use_rewnorm
        self.obsnorm_range = config.obsnorm_range
        self.rewnorm_range = config.rewnorm_range

        if config.state_space is not None:
            config.dim_state, state_shape = config.state_space.shape, config.state_space.shape
        else:
            config.dim_state, state_shape = None, None

        input_representation = get_repre_in(config)
        representation = REGISTRY_Representation[config.representation](*input_representation)
        input_policy = get_policy_in_marl(config, representation, config.agent_keys)
        policy = REGISTRY_Policy[config.policy](*input_policy)
        lr_scheduler = [MyLinearLR(config.lr_a, start_factor=1.0, end_factor=0.5, total_iters=get_total_iters(config.agent_name, config)),
                        MyLinearLR(config.lr_c, start_factor=1.0, end_factor=0.5, total_iters=get_total_iters(config.agent_name, config))]
        optimizer = [tk.optimizers.Adam(lr_scheduler[0]), tk.optimizers.Adam(lr_scheduler[1])]
        self.observation_space = envs.observation_space
        self.action_space = envs.action_space
        self.representation_info_shape = policy.representation.output_shapes
        self.auxiliary_info_shape = {}

        writer = SummaryWriter(config.logdir)
        if config.state_space is not None:
            config.dim_state, state_shape = config.state_space.shape, config.state_space.shape
        else:
            config.dim_state, state_shape = None, None
        memory = MeanField_OnPolicyBuffer(state_shape,
                                          config.obs_shape,
                                          config.act_shape,
                                          config.act_prob_shape,
                                          config.rew_shape,
                                          config.done_shape,
                                          envs.num_envs,
                                          config.nsteps,
                                          config.nminibatch,
                                          config.use_gae, config.use_advnorm, config.gamma, config.lam)
        learner = MFAC_Learner(config, policy, optimizer, writer, config.device, config.modeldir, config.gamma)

        self.obs_rms = RunningMeanStd(shape=space2shape(self.observation_space[config.agent_keys[0]]),
                                      comm=self.comm, use_mpi=False)
        self.ret_rms = RunningMeanStd(shape=(), comm=self.comm, use_mpi=False)
        super(MFAC_Agents, self).__init__(config, envs, policy, memory, learner, writer, device,
                                          config.logdir, config.modeldir)

    def _process_observation(self, observations):
        if self.use_obsnorm:
            if isinstance(self.observation_space, Dict):
                for key in self.observation_space.spaces.keys():
                    observations[key] = np.clip(
                        (observations[key] - self.obs_rms.mean[key]) / (self.obs_rms.std[key] + EPS),
                        -self.obsnorm_range, self.obsnorm_range)
            else:
                observations = np.clip((observations - self.obs_rms.mean) / (self.obs_rms.std + EPS),
                                       -self.obsnorm_range, self.obsnorm_range)
            return observations
        return observations

    def _process_reward(self, rewards):
        if self.use_rewnorm:
            std = np.clip(self.ret_rms.std, 0.1, 100)
            return np.clip(rewards / std, -self.rewnorm_range, self.rewnorm_range)
        return rewards

    def act(self, obs_n, episode, test_mode, act_mean=None, agent_mask=None, noise=False):
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

    def value(self, obs, state):
        batch_size = len(state)
        agents_id = np.tile(np.expand_dims(np.eye(self.n_agents), 0), (batch_size, 1, 1))
        repre_out = self.policy.representation(obs)
        critic_input = tf.concat([repre_out['state'], agents_id], axis=-1)
        values_n = self.policy.critic(critic_input)

        values = tf.expand_dims(tf.tile(tf.reshape(self.policy.value_tot(values_n, global_state=state), [-1, 1]), (1, self.n_agents)), axis=-1)
        return values.numpy()

    def train(self, i_episode):
        if self.memory.full:
            for _ in range(self.args.nminibatch * self.args.nepoch):
                sample = self.memory.sample()
                self.learner.update(sample)
            self.memory.clear()
