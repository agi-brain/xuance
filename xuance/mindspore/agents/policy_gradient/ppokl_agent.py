from xuance.mindspore.agents import *


class PPOKL_Agent(Agent):
    def __init__(self,
                 config: Namespace,
                 envs: VecEnv,
                 policy: nn.Cell,
                 optimizer: nn.Optimizer,
                 scheduler: Optional[nn.exponential_decay_lr] = None):
        self.config = config
        self.render = config.render
        self.comm = MPI.COMM_WORLD
        self.nenvs = envs.num_envs
        self.nsteps = config.nsteps
        self.nminibatch = config.nminibatch
        self.nepoch = config.nepoch

        self.gamma = config.gamma
        self.lam = config.lam
        self.use_obsnorm = config.use_obsnorm
        self.use_rewnorm = config.use_rewnorm
        self.obsnorm_range = config.obsnorm_range
        self.rewnorm_range = config.rewnorm_range

        self.observation_space = envs.observation_space
        self.action_space = envs.action_space
        self.representation_info_shape = policy.representation.output_shapes
        self.auxiliary_info_shape = {"old_logp": ()}

        writer = SummaryWriter(config.log_dir)
        memory = DummyOnPolicyBuffer(self.observation_space,
                                     self.action_space,
                                     self.representation_info_shape,
                                     self.auxiliary_info_shape,
                                     self.nenvs,
                                     self.nsteps,
                                     self.nminibatch,
                                     self.gamma,
                                     self.lam)
        learner = PPOCLIP_Learner(policy,
                                  optimizer,
                                  scheduler,
                                  writer,
                                  config.model_dir,
                                  config.vf_coef,
                                  config.ent_coef,
                                  0.2)

        self.obs_rms = RunningMeanStd(shape=space2shape(self.observation_space), comm=self.comm, use_mpi=False)
        self.ret_rms = RunningMeanStd(shape=(), comm=self.comm, use_mpi=False)
        super(PPOKL_Agent, self).__init__(envs, policy, memory, learner, writer, config.log_dir,
                                          config.model_dir)

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

    def _action(self, obs):
        states, act_probs, vs = self.policy(ms.Tensor(obs))
        acts = self.policy.actor.sample(act_probs)
        logps = self.policy.actor.log_prob(value=acts, probs=act_probs)
        if context._get_mode() == 0:
            return {"state": states[0].asnumpy()}, acts.asnumpy(), vs.asnumpy(), logps.asnumpy()
        else:
            for key in states.keys():
                states[key] = states[key].asnumpy()
            return states, acts.asnumpy(), vs.asnumpy(), logps.asnumpy()

    def train(self, train_steps=10000):
        episodes = np.zeros((self.nenvs,), np.int32)
        scores = np.zeros((self.nenvs,), np.float32)
        returns = np.zeros((self.nenvs,), np.float32)

        obs = self.envs.reset()
        for step in tqdm(range(train_steps)):
            self.obs_rms.update(obs)
            obs = self._process_observation(obs)
            states, acts, rets, logps = self._action(obs)
            next_obs, rewards, dones, infos = self.envs.step(acts)
            if self.render: self.envs.render()
            self.memory.store(obs, acts, self._process_reward(rewards), rets, dones, states, {"old_logp": logps})
            if self.memory.full:
                _, _, vals, _ = self._action(self._process_observation(next_obs))
                for i in range(self.nenvs):
                    self.memory.finish_path(vals[i], i)
                for _ in range(self.nminibatch * self.nepoch):
                    obs_batch, act_batch, ret_batch, adv_batch, _, aux_batch = self.memory.sample()
                    self.learner.update(obs_batch, act_batch, ret_batch, adv_batch, aux_batch['old_logp'])
                self.memory.clear()
            scores += rewards
            returns = self.gamma * returns + rewards
            obs = next_obs
            for i in range(self.nenvs):
                if dones[i] == True:
                    self.ret_rms.update(returns[i:i + 1])
                    self.memory.finish_path(0, i)
                    self.writer.add_scalars("returns-episode", {"env-%d" % i: scores[i]}, episodes[i])
                    self.writer.add_scalars("returns-step", {"env-%d" % i: scores[i]}, step)
                    scores[i] = 0
                    returns[i] = 0
                    episodes[i] += 1

            if step % 50000 == 0 or step == train_steps - 1:
                self.save_model()
                np.save(self.model_dir + "/obs_rms.npy",
                        {'mean': self.obs_rms.mean, 'std': self.obs_rms.std, 'count': self.obs_rms.count})

    def test(self, test_episodes=100000):
        self.load_model(self.model_dir)
        scores = np.zeros((self.nenvs,), np.float32)
        returns = np.zeros((self.nenvs,), np.float32)

        obs = self.envs.reset()
        for _ in tqdm(range(test_episodes)):
            self.obs_rms.update(obs)
            obs = self._process_observation(obs)
            states, acts, rets, logps = self._action(obs)
            next_obs, rewards, dones, infos = self.envs.step(acts)
            self.envs.render()

            scores += rewards
            returns = self.gamma * returns + rewards
            obs = next_obs
            for i in range(self.nenvs):
                if dones[i] == True:
                    scores[i], returns[i] = 0, 0

    def evaluate(self):
        pass
