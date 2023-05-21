from xuanpolicy.xuanpolicy_torch.agents import *


class MADQN_Agents(MARLAgents):
    def __init__(self,
                 config: Namespace,
                 envs: DummyVecEnv_MAS,
                 device: Optional[Union[int, str, torch.device]] = None):
        self.comm = MPI.COMM_WORLD

        self.gamma = config.gamma
        self.use_obsnorm = config.use_obsnorm
        self.use_rewnorm = config.use_rewnorm
        self.obsnorm_range = config.obsnorm_range
        self.rewnorm_range = config.rewnorm_range

        self.train_frequency = config.training_frequency
        self.start_training = config.start_training
        self.start_greedy = config.start_greedy
        self.end_greedy = config.end_greedy
        self.egreedy = config.start_greedy

        input_representation = get_repre_in(config)
        representation = REGISTRY_Representation[config.representation](*input_representation)
        input_policy = get_policy_in_marl(config, representation, config.agent_keys)
        policy = REGISTRY_Policy[config.policy](*input_policy)
        optimizer = torch.optim.Adam(policy.parameters(), config.learning_rate, eps=1e-5)
        scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.5,
                                                       total_iters=get_total_iters(config.agent_name, config))
        self.observation_space = envs.observation_space
        self.action_space = envs.action_space
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
                                      config.n_size,
                                      config.batch_size)
        learner = MADQN_Learner(config, policy, optimizer, scheduler, writer,
                                 config.device, config.modeldir, config.gamma)

        self.obs_rms = RunningMeanStd(shape=space2shape(self.observation_space[config.agent_keys[0]]),
                                      comm=self.comm, use_mpi=False)
        self.ret_rms = RunningMeanStd(shape=(), comm=self.comm, use_mpi=False)
        super(MADQN_Agents, self).__init__(config, envs, policy, memory, learner, writer, device,
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

    def _action(self, obs_n, egreedy, test_mode, noise):
        batch_size = len(obs_n)
        agents_id = torch.eye(self.n_agents).unsqueeze(0).expand(batch_size, -1, -1).to(self.device)
        states, argmax_action, _ = self.policy(torch.Tensor(obs_n), agents_id)
        random_action = np.random.choice(self.action_space['agent_0'].n, [self.nenvs, self.n_agents])
        if np.random.rand() < egreedy:
            actions = random_action
        else:
            actions = argmax_action.detach().cpu().numpy()
        for key in states.keys():
            states[key] = states[key].detach().cpu().numpy()
        return states, actions

    def train(self, train_steps=10000):
        episodes = np.zeros((self.nenvs, self.n_agents), np.int32)
        scores = np.zeros((self.nenvs, self.nenvs), np.float32)
        returns = np.zeros((self.nenvs, self.n_agents), np.float32)
        obs_n = self.envs.reset()
        for step in tqdm(range(train_steps)):
            self.obs_rms.update(obs_n)
            obs_n = self._process_observation(obs_n)
            states, acts = self._action(obs_n, self.egreedy, None, None)
            next_obs, rewards, dones, infos = self.envs.step(acts)
            if self.render: self.envs.render()

            self.memory.store(obs_n, acts, self._process_reward(rewards), dones, self._process_observation(next_obs),
                              states, {})
            if step > self.start_training and step % self.train_frequency == 0:
                # training
                obs_batch, act_batch, rew_batch, terminal_batch, next_batch, _, _ = self.memory.sample()
                self.learner.update(obs_batch, act_batch, rew_batch, next_batch, terminal_batch)

            scores += rewards
            returns = self.gamma * returns + rewards
            obs_n = next_obs
            self.egreedy = self.egreedy - (self.start_greedy - self.end_greedy) / train_steps

            for i in range(self.nenvs):
                if dones[i] == True:
                    self.ret_rms.update(returns[i:i + 1])
                    self.writer.add_scalars("returns-episode", {"env-%d" % i: scores[i]}, episodes[i])
                    self.writer.add_scalars("returns-step", {"env-%d" % i: scores[i]}, step)
                    scores[i] = 0
                    returns[i] = 0
                    episodes[i] += 1
            if step % 50000 == 0 or step == train_steps - 1:
                self.save_model()
                np.save(self.modeldir + "/obs_rms.npy",
                        {'mean': self.obs_rms.mean, 'std': self.obs_rms.std, 'count': self.obs_rms.count})

    def test(self, test_steps=10000, episode=0):
        self.load_model(self.modeldir)
        scores = np.zeros((self.nenvs, self.n_agents), np.float32)
        returns = np.zeros((self.nenvs, self.n_agents), np.float32)
        obs_n = self.envs.reset()
        for _ in tqdm(range(test_steps)):
            self.obs_rms.update(obs_n)
            obs_n = self._process_observation(obs_n)
            states, acts = self._action(obs_n, egreedy=0.0, test_mode=None, noise=None)
            next_obs, rewards, dones, infos = self.envs.step(acts)
            self.envs.render()

            scores += rewards
            returns = self.gamma * returns + rewards
            obs_n = next_obs

            for i in range(self.nenvs):
                if dones[i] == True:
                    scores[i], returns[i] = 0, 0

    def evaluate(self):
        pass