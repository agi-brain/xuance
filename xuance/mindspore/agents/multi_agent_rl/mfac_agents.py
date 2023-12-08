from xuance.mindspore.agents import *


class MFAC_Agents(MARLAgents):
    def __init__(self,
                 config: Namespace,
                 envs: DummyVecEnv_Pettingzoo):
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
        scheduler = [lr_decay_model(learning_rate=config.lr_a, decay_rate=0.5,
                                    decay_steps=get_total_iters(config.agent_name, config)),
                     lr_decay_model(learning_rate=config.lr_c, decay_rate=0.5,
                                    decay_steps=get_total_iters(config.agent_name, config))]
        optimizer = [Adam(policy.parameters_actor, scheduler[0], eps=1e-5),
                     Adam(policy.parameters_critic, scheduler[1], eps=1e-5)]
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
        learner = MFAC_Learner(config, policy, optimizer, scheduler, writer, config.modeldir, config.gamma)

        self.obs_rms = RunningMeanStd(shape=space2shape(self.observation_space[config.agent_keys[0]]),
                                      comm=self.comm, use_mpi=False)
        self.ret_rms = RunningMeanStd(shape=(), comm=self.comm, use_mpi=False)
        super(MFAC_Agents, self).__init__(config, envs, policy, memory, learner, writer, config.logdir, config.modeldir)
        self._concat = ops.Concat(axis=-1)

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
        agents_id = ops.broadcast_to(self.expand_dims(self.eye(self.n_agents, self.n_agents, ms.float32), 0),
                                     (batch_size, -1, -1))
        obs_n = Tensor(obs_n)

        _, act_probs = self.policy(obs_n, agents_id)
        acts = self.policy.actor_net.sample(act_probs)

        n_alive = ops.broadcast_to(self.expand_dims(Tensor(agent_mask).sum(axis=-1), -1), (-1, self.dim_act))
        action_n_mask = ops.broadcast_to(self.expand_dims(Tensor(agent_mask), -1), (-1, -1, self.dim_act))
        act_neighbor_onehot = self.learner.onehot_action(acts, self.dim_act) * action_n_mask
        act_mean_current = act_neighbor_onehot.sum(axis=1) / n_alive
        act_mean_current = act_mean_current.asnumpy()

        return acts.asnumpy(), act_mean_current

    def value(self, obs, state):
        batch_size = len(state)
        agents_id = ops.broadcast_to(self.expand_dims(self.eye(self.n_agents, self.n_agents, ms.int32), 0), (batch_size, -1, -1))
        repre_out = self.policy.representation(obs)
        critic_input = self._concat([torch.Tensor(repre_out[0]), agents_id])
        values_n = self.policy.critic(critic_input)
        values = self.expand_dims(ops.broadcast_to(self.policy.value_tot(values_n, global_state=state).view(-1, 1), (-1, self.n_agents)), -1)
        return values.asnumpy()

    def train(self, i_episode):
        if self.memory.full:
            for _ in range(self.args.nminibatch * self.args.nepoch):
                sample = self.memory.sample()
                self.learner.update(sample)
            # self.memory.clear()
