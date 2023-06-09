from xuanpolicy.torch.agents import *


class A2C_Agent(Agent):
    def __init__(self,
                 config: Namespace,
                 envs: VecEnv,
                 policy: nn.Module,
                 optimizer: torch.optim.Optimizer,
                 scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
                 device: Optional[Union[int, str, torch.device]] = None):
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
        self.clip_grad = config.clip_grad

        self.observation_space = envs.observation_space
        self.action_space = envs.action_space
        self.representation_info_shape = policy.representation.output_shapes
        self.auxiliary_info_shape = {}

        memory = DummyOnPolicyBuffer(self.observation_space,
                                     self.action_space,
                                     self.representation_info_shape,
                                     self.auxiliary_info_shape,
                                     self.nenvs,
                                     self.nsteps,
                                     self.nminibatch,
                                     self.gamma,
                                     self.lam)
        learner = A2C_Learner(policy,
                              optimizer,
                              scheduler,
                              config.device,
                              config.modeldir,
                              config.vf_coef,
                              config.ent_coef,
                              config.clip_grad)

        self.obs_rms = RunningMeanStd(shape=space2shape(self.observation_space), comm=self.comm, use_mpi=False)
        self.ret_rms = RunningMeanStd(shape=(), comm=self.comm, use_mpi=False)
        super(A2C_Agent, self).__init__(config, envs, policy, memory, learner, device, config.logdir, config.modeldir)

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
        states, dists, vs = self.policy(obs)
        acts = dists.stochastic_sample()
        for key in states.keys():
            states[key] = states[key].detach().cpu().numpy()
        acts = acts.detach().cpu().numpy()
        vs = vs.detach().cpu().numpy()
        return states, acts, vs

    def train(self, train_steps=10000):
        episodes = np.zeros((self.nenvs,), np.int32)
        scores = np.zeros((self.nenvs,), np.float32)
        returns = np.zeros((self.nenvs,), np.float32)

        obs, infos = self.envs.reset()
        for step in tqdm(range(train_steps)):
            step_info, episode_info = {}, {}
            self.obs_rms.update(obs)
            obs = self._process_observation(obs)
            states, acts, rets = self._action(obs)
            next_obs, rewards, terminals, trunctions, infos = self.envs.step(acts)
            self.memory.store(obs, acts, self._process_reward(rewards), rets, terminals, states, {})
            if self.memory.full:
                _, _, vals = self._action(self._process_observation(next_obs))
                for i in range(self.nenvs):
                    self.memory.finish_path(vals[i], i)
                for _ in range(self.nminibatch * self.nepoch):
                    obs_batch, act_batch, ret_batch, adv_batch, _, _ = self.memory.sample()
                    step_info = self.learner.update(obs_batch, act_batch, ret_batch, adv_batch)
                self.memory.clear()
            scores += rewards
            returns = self.gamma * returns + rewards
            obs = next_obs
            for i in range(self.nenvs):
                if terminals[i] or trunctions[i]:
                    self.ret_rms.update(returns[i:i + 1])
                    self.memory.finish_path(0, i)
                    step_info["returns-step"] = {"env-%d" % i: scores[i]}
                    episode_info["returns-episode"] = {"env-%d" % i: scores[i]}
                    scores[i], returns[i] = 0, 0
                    episodes[i] += 1
                    self.log_infos(step_info, step)
                    self.log_infos(episode_info, episodes[i])

            if step % self.config.save_model_frequency == 0 or step == train_steps - 1:
                self.save_model()
                np.save(self.modeldir + "/obs_rms.npy",
                        {'mean': self.obs_rms.mean, 'std': self.obs_rms.std, 'count': self.obs_rms.count})

    def test(self, test_steps=100):
        self.load_model(self.modeldir)
        scores = np.zeros((self.nenvs,), np.float32)
        returns = np.zeros((self.nenvs,), np.float32)
        obs, infos = self.envs.reset()
        videos = [[] for _ in range(self.nenvs)]
        for step in tqdm(range(test_steps)):
            self.obs_rms.update(obs)
            obs = self._process_observation(obs)
            states, acts, rets = self._action(obs)
            next_obs, rewards, terminals, trunctions, infos = self.envs.step(acts)
            if self.config.render and self.config.render_mode == "rgb_array":
                images = self.envs.render(self.config.render_mode)
                for idx, img in enumerate(images):
                    videos[idx].append(img)

            scores += rewards
            returns = self.gamma * returns + rewards
            obs = next_obs
            for i in range(self.nenvs):
                if terminals[i] or trunctions[i]:
                    scores[i], returns[i] = 0, 0

        if self.config.render and self.config.render_mode == "rgb_array":
            # batch, time, height, width, channel -> batch, time, channel, height, width
            videos_info = {"Videos_Test": np.array(videos, dtype=np.uint8).transpose((0, 1, 4, 2, 3))}
            self.log_videos(info=videos_info, fps=50)
