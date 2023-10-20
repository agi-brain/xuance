from xuance.tensorflow.agents import *
from xuance.tensorflow.agents.agents_marl import linear_decay_or_increase


class MFQ_Agents(MARLAgents):
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

        input_representation = get_repre_in(config)
        representation = REGISTRY_Representation[config.representation](*input_representation)
        input_policy = get_policy_in_marl(config, representation, config.agent_keys)
        policy = REGISTRY_Policy[config.policy](*input_policy)
        lr_scheduler = MyLinearLR(config.learning_rate, start_factor=1.0, end_factor=0.5,
                                  total_iters=get_total_iters(config.agent_name, config))
        optimizer = tk.optimizers.Adam(lr_scheduler)
        self.observation_space = envs.observation_space
        self.action_space = envs.action_space
        self.representation_info_shape = policy.representation.output_shapes
        self.auxiliary_info_shape = {}

        writer = SummaryWriter(config.logdir)
        if config.state_space is not None:
            config.dim_state, state_shape = config.state_space.shape, config.state_space.shape
        else:
            config.dim_state, state_shape = None, None
        memory = MeanField_OffPolicyBuffer(state_shape,
                                           config.obs_shape,
                                           config.act_shape,
                                           config.act_prob_shape,
                                           config.rew_shape,
                                           config.done_shape,
                                           envs.num_envs,
                                           config.buffer_size,
                                           config.batch_size)
        learner = MFQ_Learner(config, policy, optimizer, writer,
                              config.device, config.modeldir, config.gamma, config.sync_frequency)

        self.obs_rms = RunningMeanStd(shape=space2shape(self.observation_space[config.agent_keys[0]]),
                                      comm=self.comm, use_mpi=False)
        self.ret_rms = RunningMeanStd(shape=(), comm=self.comm, use_mpi=False)
        self.epsilon_decay = linear_decay_or_increase(config.start_greedy, config.end_greedy,
                                                      config.greedy_update_steps)
        super(MFQ_Agents, self).__init__(config, envs, policy, memory, learner, writer, device,
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
        if not test_mode:
            epsilon = self.epsilon_decay.epsilon
        else:
            epsilon = 1.0
        batch_size = obs_n.shape[0]
        act_mean = np.expand_dims(act_mean, axis=-2).repeat(self.n_agents, axis=1)
        inputs = {"obs": obs_n,
                  "act_mean": act_mean,
                  "ids": np.tile(np.expand_dims(np.eye(self.n_agents), 0), (batch_size, 1, 1))}
        _, greedy_actions, q_output = self.policy(inputs)
        n_alive = np.expand_dims(np.sum(agent_mask, axis=-1), axis=-1).repeat(self.dim_act, axis=1)
        action_n_mask = np.expand_dims(agent_mask, axis=-1).repeat(self.dim_act, axis=-1)
        act_neighbor_sample = self.policy.sample_actions(logits=q_output)
        act_neighbor_onehot = self.learner.onehot_action(act_neighbor_sample, self.dim_act).numpy() * action_n_mask
        act_mean_current = np.sum(act_neighbor_onehot, axis=1) / n_alive
        greedy_actions = greedy_actions.numpy()
        if noise:
            random_variable = np.random.rand(batch_size, self.n_agents).reshape(greedy_actions.shape)
            action_pick = np.int32((random_variable < epsilon))
            random_actions = np.array([[self.args.action_space[agent].sample() for agent in self.agent_keys]])
            return action_pick * greedy_actions + (1 - action_pick) * random_actions, act_mean_current
        else:
            return greedy_actions, act_mean_current

    def train(self, i_episode):
        self.epsilon_decay.update()
        for i in range(self.nenvs):
            self.writer.add_scalars("epsilon", {"env-%d" % i: self.epsilon_decay.epsilon}, i_episode)
        if self.memory.can_sample(self.args.batch_size):
            sample = self.memory.sample()
            self.learner.update(sample)
