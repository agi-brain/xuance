from xuance.mindspore.agents import *
from xuance.mindspore.agents.agents_marl import linear_decay_or_increase


class MFQ_Agents(MARLAgents):
    def __init__(self,
                 config: Namespace,
                 envs: DummyVecEnv_Pettingzoo):
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
        scheduler = lr_decay_model(learning_rate=config.learning_rate, decay_rate=0.5,
                                   decay_steps=get_total_iters(config.agent_name, config))
        optimizer = Adam(policy.trainable_params(), scheduler, eps=1e-5)
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
        learner = MFQ_Learner(config, policy, optimizer, scheduler, writer,
                              config.modeldir, config.gamma, config.sync_frequency)

        self.obs_rms = RunningMeanStd(shape=space2shape(self.observation_space[config.agent_keys[0]]),
                                      comm=self.comm, use_mpi=False)
        self.ret_rms = RunningMeanStd(shape=(), comm=self.comm, use_mpi=False)
        self.epsilon_decay = linear_decay_or_increase(config.start_greedy, config.end_greedy,
                                                      config.greedy_update_steps)
        super(MFQ_Agents, self).__init__(config, envs, policy, memory, learner, writer, config.logdir, config.modeldir)

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
        agents_id = ops.broadcast_to(self.expand_dims(self.eye(self.n_agents, self.n_agents, ms.float32), 0),
                                     (batch_size, -1, -1))
        obs_in = Tensor(obs_n)
        act_mean = ops.broadcast_to(self.expand_dims(Tensor(act_mean), -2), (-1, self.n_agents, -1))

        _, greedy_actions, q_output = self.policy(obs_in, act_mean.astype(ms.float32), agents_id)
        n_alive = ops.broadcast_to(self.expand_dims(Tensor(agent_mask).sum(axis=-1), -1), (-1, self.dim_act))
        action_n_mask = ops.broadcast_to(self.expand_dims(Tensor(agent_mask), -1), (-1, -1, self.dim_act))
        act_neighbor_sample = self.policy.sample_actions(logits=q_output)
        act_neighbor_onehot = self.learner.onehot_action(act_neighbor_sample, self.dim_act) * action_n_mask
        act_mean_current = act_neighbor_onehot.sum(axis=1) / n_alive
        act_mean_current = act_mean_current.asnumpy()
        greedy_actions = greedy_actions.asnumpy()
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
