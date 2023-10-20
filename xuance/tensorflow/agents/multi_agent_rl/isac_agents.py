from xuance.tensorflow.agents import *


class ISAC_Agents(MARLAgents):
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

        input_representation = get_repre_in(config)
        representation = REGISTRY_Representation[config.representation](*input_representation)
        input_policy = get_policy_in_marl(config, representation, config.agent_keys)
        policy = REGISTRY_Policy[config.policy](*input_policy)
        lr_scheduler = [MyLinearLR(config.lr_a, start_factor=1.0, end_factor=0.5,
                                   total_iters=get_total_iters(config.agent_name, config)),
                        MyLinearLR(config.lr_c, start_factor=1.0, end_factor=0.5,
                                   total_iters=get_total_iters(config.agent_name, config))]
        optimizer = [tk.optimizers.Adam(lr_scheduler[0]),
                     tk.optimizers.Adam(lr_scheduler[1])]
        self.observation_space = envs.observation_space
        self.action_space = envs.action_space
        self.actions_high, self.actions_low = [], []
        for k in config.agent_keys:
            self.actions_high.append(self.action_space[k].high)
            self.actions_low.append(self.action_space[k].low)
        self.actions_high, self.actions_low = np.array(self.actions_high), np.array(self.actions_low)
        self.representation_info_shape = policy.representation.output_shapes
        self.auxiliary_info_shape = {}

        writer = SummaryWriter(config.logdir)
        if config.state_space is not None:
            config.dim_state, state_shape = config.state_space.shape, config.state_space.shape
        else:
            config.dim_state, state_shape = None, None
        memory = MARL_OffPolicyBuffer(state_shape,
                                      config.obs_shape,
                                      config.act_shape,
                                      config.rew_shape,
                                      config.done_shape,
                                      envs.num_envs,
                                      config.buffer_size,
                                      config.batch_size)
        learner = ISAC_Learner(config, policy, optimizer, writer, config.device, config.modeldir, config.gamma)

        self.obs_rms = RunningMeanStd(shape=space2shape(self.observation_space[config.agent_keys[0]]),
                                      comm=self.comm, use_mpi=False)
        self.ret_rms = RunningMeanStd(shape=(), comm=self.comm, use_mpi=False)
        super(ISAC_Agents, self).__init__(config, envs, policy, memory, learner, writer, device,
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

    def act(self, obs_n, episode, test_mode, state=None, noise=False):
        batch_size = len(obs_n)
        with tf.device(self.device):
            agents_id = tf.tile(tf.expand_dims(tf.eye(self.n_agents), axis=0), multiples=(batch_size, 1, 1))
            inputs_policy = {"obs": tf.convert_to_tensor(obs_n), "ids": agents_id}
            _, dists = self.policy(inputs_policy)
            acts = dists.sample()
        actions = acts.numpy()
        actions = np.clip(actions, self.actions_low, self.actions_high)
        return actions

    def train(self, i_episode):
        if self.memory.can_sample(self.args.batch_size):
            sample = self.memory.sample()
            self.learner.update(sample)
