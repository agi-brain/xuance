from xuance.mindspore.agents import *
import gym
from gym import spaces

class SPDQN_Agent(Agent):
    def __init__(self,
                 config: Namespace,
                 envs: Gym_Env,
                 policy: nn.Cell,
                 optimizer: Sequence[nn.Optimizer],
                 scheduler):
        self.config = config
        self.envs = envs
        self.render = config.render
        self.comm = MPI.COMM_WORLD

        self.gamma = config.gamma
        self.use_obsnorm = config.use_obsnorm
        self.use_rewnorm = config.use_rewnorm
        self.obsnorm_range = config.obsnorm_range
        self.rewnorm_range = config.rewnorm_range

        self.train_frequency = config.training_frequency
        self.start_training = config.start_training
        self.start_noise = config.start_noise
        self.end_noise = config.end_noise
        self.noise_scale = config.start_noise

        self.observation_space = envs.observation_space.spaces[0]
        old_as = envs.action_space
        num_disact = old_as.spaces[0].n
        self.action_space = gym.spaces.Tuple((old_as.spaces[0], *(gym.spaces.Box(old_as.spaces[1].spaces[i].low,
                                                                                 old_as.spaces[1].spaces[i].high,
                                                                                 dtype=np.float32) for i in
                                                                  range(0, num_disact))))
        self.action_high = [self.action_space.spaces[i].high for i in range(1, num_disact + 1)]
        self.action_low = [self.action_space.spaces[i].low for i in range(1, num_disact + 1)]
        self.action_range = [self.action_space.spaces[i].high - self.action_space.spaces[i].low for i in
                             range(1, num_disact + 1)]
        self.representation_info_shape = {'state': (envs.observation_space.spaces[0].shape)}
        self.auxiliary_info_shape = {}
        self.nenvs = 1
        self.epsilon = 1.0
        self.epsilon_steps = 1000
        self.epsilon_initial = 1.0
        self.epsilon_final = 0.1
        self.buffer_action_space = spaces.Box(np.zeros(4), np.ones(4), dtype=np.float64)

        writer = SummaryWriter(config.logdir)
        memory = DummyOffPolicyBuffer(self.observation_space,
                                      self.buffer_action_space,
                                      self.representation_info_shape,
                                      self.auxiliary_info_shape,
                                      self.nenvs,
                                      config.nsize,
                                      config.batchsize)
        learner = MPDQN_Learner(policy,
                                optimizer,
                                scheduler,
                                writer,
                                config.modeldir,
                                config.gamma,
                                config.tau)

        self.num_disact = self.action_space.spaces[0].n
        self.conact_sizes = np.array([self.action_space.spaces[i].shape[0] for i in range(1, self.num_disact + 1)])
        self.conact_size = int(self.conact_sizes.sum())

        self.obs_rms = RunningMeanStd(shape=space2shape(self.observation_space), comm=self.comm, use_mpi=False)
        self.ret_rms = RunningMeanStd(shape=(), comm=self.comm, use_mpi=False)
        super(SPDQN_Agent, self).__init__(envs, policy, memory, learner, writer, config.logdir, config.modeldir)

    def _process_observation(self, observations):
        if self.use_obsnorm:
            if isinstance(self.observation_space, gym.spaces.Dict):
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

    def _action(self, obs):
        obs = ms.Tensor(obs)
        con_actions = self.policy.con_action(obs)
        rnd = np.random.rand()
        if rnd < self.epsilon:
            disaction = np.random.choice(self.num_disact)
        else:
            obs = obs.expand_dims(0)
            conact_batch = con_actions.expand_dims(0)
            obs = obs.astype(conact_batch.dtype)
            batch_size = obs.shape[0]
            input_q = self.policy._concat((obs, self.policy._zeroslike(conact_batch)))
            input_q = input_q.repeat(self.policy.num_disact, 0)
            input_q = input_q.asnumpy()
            conact_batch = conact_batch.asnumpy()
            for i in range(self.policy.num_disact):
                input_q[i * batch_size:(i + 1) * batch_size,
                self.policy.obs_size + self.policy.offsets[i]: self.policy.obs_size + self.policy.offsets[i + 1]] \
                    = conact_batch[:, self.policy.offsets[i]:self.policy.offsets[i + 1]]
            input_q = ms.Tensor(input_q, dtype=ms.float32)

            q = self.policy.Qeval(obs, conact_batch, input_q)
            q = q.asnumpy()
            disaction = np.argmax(q)
        con_actions = con_actions.asnumpy()
        offset = np.array([self.conact_sizes[i] for i in range(disaction)], dtype=int).sum()
        conaction = con_actions[offset:offset + self.conact_sizes[disaction]]

        return disaction, conaction, con_actions

    def pad_action(self, disaction, conaction):
        con_actions = [np.zeros((1,), dtype=np.float32), np.zeros((1,), dtype=np.float32),
                       np.zeros((1,), dtype=np.float32)]
        con_actions[disaction][:] = conaction
        return (disaction, con_actions)

    def train(self, train_steps=10000):
        episodes = np.zeros((self.nenvs,), np.int32)
        scores = np.zeros((self.nenvs,), np.float32)
        returns = np.zeros((self.nenvs,), np.float32)
        obs, _ = self.envs.reset()
        for step in tqdm(range(train_steps)):
            disaction, conaction, con_actions = self._action(obs)
            action = self.pad_action(disaction, conaction)
            action[1][disaction] = self.action_range[disaction] * (action[1][disaction] + 1) / 2. + self.action_low[
                disaction]
            (next_obs, steps), rewards, terminal, _ = self.envs.step(action)
            if self.render: self.envs.render("human")
            acts = np.concatenate(([disaction], con_actions), axis=0).ravel()
            state = {'state': obs}
            self.memory.store(obs, acts, rewards, terminal, next_obs, state, {})
            if step > self.start_training and step % self.train_frequency == 0:
                # training
                obs_batch, act_batch, rew_batch, terminal_batch, next_batch, _, _ = self.memory.sample()
                self.learner.update(obs_batch, act_batch, rew_batch, next_batch, terminal_batch)

            scores += rewards
            returns = self.gamma * returns + rewards
            obs = next_obs
            self.noise_scale = self.start_noise - (self.start_noise - self.end_noise) / train_steps
            if terminal == True:
                self.writer.add_scalar("returns-episode", scores, episodes)
                scores = 0
                returns = 0
                episodes += 1
                self.end_episode(episodes)
                obs, _ = self.envs.reset()

            if step % 50000 == 0 or step == train_steps - 1:
                self.save_model()
                np.save(self.modeldir + "/obs_rms.npy",
                        {'mean': self.obs_rms.mean, 'std': self.obs_rms.std, 'count': self.obs_rms.count})

    def end_episode(self, episode):
        if episode < self.epsilon_steps:
            self.epsilon = self.epsilon_initial - (self.epsilon_initial - self.epsilon_final) * (
                    episode / self.epsilon_steps)
        else:
            self.epsilon = self.epsilon_final

    def test(self, test_steps=10000):
        self.load_model(self.modeldir)
        scores = np.zeros((self.nenvs,), np.float32)
        returns = np.zeros((self.nenvs,), np.float32)
        obs = self.envs.reset()
        for _ in tqdm(range(test_steps)):
            disaction, conaction, con_actions = self._action(obs)
            action = self.pad_action(disaction, conaction)
            action[1][disaction] = self.action_range[disaction] * (action[1][disaction] + 1) / 2. + self.action_low[
                disaction]
            (next_obs, steps), rewards, terminal, _ = self.envs.step(action)
            self.envs.render("human")
            scores += rewards
            returns = self.gamma * returns + rewards
            obs = next_obs
            if terminal == True:
                scores, returns = 0, 0
                obs, _ = self.envs.reset()

    def evaluate(self):
        pass
