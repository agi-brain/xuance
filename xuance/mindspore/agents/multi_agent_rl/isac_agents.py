from xuance.mindspore.agents import *


class ISAC_Agents(MARLAgents):
    def __init__(self,
                 config: Namespace,
                 envs: DummyVecEnv_Pettingzoo):
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
        scheduler = [lr_decay_model(learning_rate=config.lr_a, decay_rate=0.5,
                                    decay_steps=get_total_iters(config.agent_name, config)),
                     lr_decay_model(learning_rate=config.lr_c, decay_rate=0.5,
                                    decay_steps=get_total_iters(config.agent_name, config))]
        optimizer = [Adam(policy.parameters_actor, scheduler[0], eps=1e-5),
                     Adam(policy.parameters_critic, scheduler[1], eps=1e-5)]
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
        learner = ISAC_Learner(config, policy, optimizer, scheduler, writer, config.modeldir, config.gamma)

        self.obs_rms = RunningMeanStd(shape=space2shape(self.observation_space[config.agent_keys[0]]),
                                      comm=self.comm, use_mpi=False)
        self.ret_rms = RunningMeanStd(shape=(), comm=self.comm, use_mpi=False)
        super(ISAC_Agents, self).__init__(config, envs, policy, memory, learner, writer,
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
        agents_id = ops.broadcast_to(self.expand_dims(self.eye(self.n_agents, self.n_agents, ms.float32), 0),
                                     (batch_size, -1, -1))
        _, act_mu, act_std = self.policy(Tensor(obs_n), agents_id)
        acts = self.policy.actor_net.sample(act_mu, act_std)
        actions = acts.asnumpy()
        actions = np.clip(actions, self.actions_low, self.actions_high)
        return actions

    def train(self, i_episode):
        if self.memory.can_sample(self.args.batch_size):
            sample = self.memory.sample()
            self.learner.update(sample)
