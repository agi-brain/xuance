from xuance.tensorflow.agents import *
from xuance.tensorflow.agents.agents_marl import linear_decay_or_increase


class QMIX_Agents(MARLAgents):
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

        self.start_greedy = config.start_greedy
        self.end_greedy = config.end_greedy
        self.egreedy = config.start_greedy
        if config.state_space is not None:
            config.dim_state, state_shape = config.state_space.shape, config.state_space.shape
        else:
            config.dim_state, state_shape = None, None

        input_representation = get_repre_in(config)
        representation = REGISTRY_Representation[config.representation](*input_representation)
        mixer = QMIX_mixer(config.dim_state[0], config.hidden_dim_mixing_net, config.hidden_dim_hyper_net,
                           config.n_agents)
        input_policy = get_policy_in_marl(config, representation, config.agent_keys, mixer)
        policy = REGISTRY_Policy[config.policy](*input_policy)
        # lr_scheduler = tk.optimizers.schedules.ExponentialDecay(config.learning_rate, decay_steps=1000,
        #                                                         decay_rate=0.9)
        lr_scheduler = MyLinearLR(config.learning_rate, start_factor=1.0, end_factor=0.5,
                                  total_iters=get_total_iters(config.agent_name, config))
        optimizer = tk.optimizers.Adam(lr_scheduler)
        self.observation_space = envs.observation_space
        self.action_space = envs.action_space
        self.representation_info_shape = policy.representation.output_shapes
        self.auxiliary_info_shape = {}

        writer = SummaryWriter(config.logdir)
        memory = MARL_OffPolicyBuffer(state_shape,
                                      config.obs_shape,
                                      config.act_shape,
                                      config.rew_shape,
                                      config.done_shape,
                                      envs.num_envs,
                                      config.buffer_size,
                                      config.batch_size)
        learner = QMIX_Learner(config, policy, optimizer, writer,
                               config.device, config.modeldir, config.gamma,
                               config.sync_frequency)

        self.obs_rms = RunningMeanStd(shape=space2shape(self.observation_space[config.agent_keys[0]]),
                                      comm=self.comm, use_mpi=False)
        self.ret_rms = RunningMeanStd(shape=(), comm=self.comm, use_mpi=False)
        self.epsilon_decay = linear_decay_or_increase(config.start_greedy, config.end_greedy,
                                                      config.greedy_update_steps)
        super(QMIX_Agents, self).__init__(config, envs, policy, memory, learner, writer, device,
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

    def train(self, i_episode):
        self.epsilon_decay.update()
        for i in range(self.nenvs):
            self.writer.add_scalars("epsilon", {"env-%d" % i: self.epsilon_decay.epsilon}, i_episode)
        if self.memory.can_sample(self.args.batch_size):
            sample = self.memory.sample()
            self.learner.update(sample)
