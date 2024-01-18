from xuance.mindspore.agents import *


class TD3_Agent(Agent):
    def __init__(self,
                 config: Namespace,
                 envs: VecEnv,
                 policy: nn.Cell,
                 optimizer: Sequence[nn.Optimizer],
                 scheduler: Optional[nn.exponential_decay_lr] = None):
        self.render = config.render
        self.n_envs = envs.num_envs

        self.gamma = config.gamma
        self.train_frequency = config.training_frequency
        self.start_training = config.start_training
        self.start_noise = config.start_noise
        self.end_noise = config.end_noise
        self.noise_scale = config.start_noise

        self.observation_space = envs.observation_space
        self.action_space = envs.action_space
        self.auxiliary_info_shape = {}

        memory = DummyOffPolicyBuffer(self.observation_space,
                                      self.action_space,
                                      self.auxiliary_info_shape,
                                      self.n_envs,
                                      config.n_size,
                                      config.batch_size)
        learner = TD3_Learner(policy,
                              optimizer,
                              scheduler,
                              config.model_dir,
                              config.gamma,
                              config.tau,
                              config.actor_update_delay)
        super(TD3_Agent, self).__init__(config, envs, policy, memory, learner, config.log_dir, config.model_dir)

    def _action(self, obs, noise_scale=0.0):
        _, action = self.policy.action(ms.Tensor(obs))
        action = action.asnumpy() + np.random.normal(size=action.shape) * noise_scale
        return np.clip(action, -1, 1)

    def train(self, train_steps):
        obs = self.envs.buf_obs
        for _ in tqdm(range(train_steps)):
            step_info = {}
            self.obs_rms.update(obs)
            obs = self._process_observation(obs)
            acts = self._action(obs, self.noise_scale)
            if self.current_step < self.start_training:
                acts = [self.action_space.sample() for _ in range(self.n_envs)]
            next_obs, rewards, terminals, trunctions, infos = self.envs.step(acts)
            self.memory.store(obs, acts, self._process_reward(rewards), terminals, self._process_observation(next_obs))
            if self.current_step > self.start_training and self.current_step % self.train_frequency == 0:
                obs_batch, act_batch, rew_batch, terminal_batch, next_batch = self.memory.sample()
                step_info = self.learner.update(obs_batch, act_batch, rew_batch, next_batch, terminal_batch)
                step_info["noise_scale"] = self.noise_scale
                self.log_infos(step_info, self.current_step)

            self.returns = self.gamma * self.returns + rewards
            obs = next_obs
            for i in range(self.n_envs):
                if terminals[i] or trunctions[i]:
                    obs[i] = infos[i]["reset_obs"]
                    self.ret_rms.update(self.returns[i:i + 1])
                    self.returns[i] = 0.0
                    self.current_episode[i] += 1
                    if self.use_wandb:
                        step_info["Episode-Steps/env-%d" % i] = infos[i]["episode_step"]
                        step_info["Train-Episode-Rewards/env-%d" % i] = infos[i]["episode_score"]
                    else:
                        step_info["Episode-Steps"] = {"env-%d" % i: infos[i]["episode_step"]}
                        step_info["Train-Episode-Rewards"] = {"env-%d" % i: infos[i]["episode_score"]}
                    self.log_infos(step_info, self.current_step)

            self.current_step += self.n_envs
            if self.noise_scale >= self.end_noise:
                self.noise_scale = self.noise_scale - (self.start_noise - self.end_noise) / self.config.running_steps

    def test(self, env_fn, test_episodes):
        test_envs = env_fn()
        num_envs = test_envs.num_envs
        videos, episode_videos = [[] for _ in range(num_envs)], []
        current_episode, scores, best_score = 0, [], -np.inf
        obs, infos = test_envs.reset()
        if self.config.render_mode == "rgb_array" and self.render:
            images = test_envs.render(self.config.render_mode)
            for idx, img in enumerate(images):
                videos[idx].append(img)

        while current_episode < test_episodes:
            self.obs_rms.update(obs)
            obs = self._process_observation(obs)
            acts = self._action(obs, noise_scale=0.0)
            next_obs, rewards, terminals, trunctions, infos = test_envs.step(acts)
            if self.config.render_mode == "rgb_array" and self.render:
                images = test_envs.render(self.config.render_mode)
                for idx, img in enumerate(images):
                    videos[idx].append(img)

            obs = next_obs
            for i in range(num_envs):
                if terminals[i] or trunctions[i]:
                    obs[i] = infos[i]["reset_obs"]
                    scores.append(infos[i]["episode_score"])
                    current_episode += 1
                    if best_score < infos[i]["episode_score"]:
                        best_score = infos[i]["episode_score"]
                        episode_videos = videos[i].copy()
                    if self.config.test_mode:
                        print("Episode: %d, Score: %.2f" % (current_episode, infos[i]["episode_score"]))

        if self.config.render_mode == "rgb_array" and self.render:
            # time, height, width, channel -> time, channel, height, width
            videos_info = {"Videos_Test": np.array([episode_videos], dtype=np.uint8).transpose((0, 1, 4, 2, 3))}
            self.log_videos(info=videos_info, fps=self.fps, x_index=self.current_step)

        if self.config.test_mode:
            print("Best Score: %.2f" % (best_score))

        test_info = {
            "Test-Episode-Rewards/Mean-Score": np.mean(scores),
            "Test-Episode-Rewards/Std-Score": np.std(scores)
        }
        self.log_infos(test_info, self.current_step)

        test_envs.close()

        return scores
