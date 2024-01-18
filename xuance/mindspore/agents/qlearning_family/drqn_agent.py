from xuance.mindspore.agents import *


class DRQN_Agent(Agent):
    def __init__(self,
                 config: Namespace,
                 envs: DummyVecEnv_Gym,
                 policy: nn.Cell,
                 optimizer: nn.Optimizer,
                 scheduler):
        self.render = config.render
        self.n_envs = envs.num_envs

        self.gamma = config.gamma
        self.train_frequency = config.training_frequency
        self.start_training = config.start_training
        self.start_greedy = config.start_greedy
        self.end_greedy = config.end_greedy
        self.egreedy = config.start_greedy

        self.observation_space = envs.observation_space
        self.action_space = envs.action_space
        self.auxiliary_info_shape = {}

        self.atari = True if config.env_name == "Atari" else False
        memory = RecurrentOffPolicyBuffer(self.observation_space,
                                          self.action_space,
                                          self.auxiliary_info_shape,
                                          self.n_envs,
                                          config.n_size,
                                          config.batch_size,
                                          episode_length=envs.max_episode_length,
                                          lookup_length=config.lookup_length)
        learner = DRQN_Learner(policy,
                               optimizer,
                               scheduler,
                               config.model_dir,
                               config.gamma,
                               config.sync_frequency)
        super(DRQN_Agent, self).__init__(config, envs, policy, memory, learner, config.log_dir, config.model_dir)
        self.lstm = True if config.rnn == "LSTM" else False

    def _action(self, obs, egreedy=0.0, rnn_hidden=None):
        _, argmax_action, _, rnn_hidden_next = self.policy(Tensor(obs[:, np.newaxis]), *rnn_hidden)
        random_action = np.random.choice(self.action_space.n, self.n_envs)
        if np.random.rand() < egreedy:
            action = random_action
        else:
            action = argmax_action.asnumpy()
        return action, rnn_hidden_next

    def train(self, train_steps):
        obs = self.envs.buf_obs
        episode_data = [EpisodeBuffer() for _ in range(self.n_envs)]
        for i_env in range(self.n_envs):
            episode_data[i_env].obs.append(self._process_observation(obs[i_env]))
        self.rnn_hidden = self.policy.init_hidden(self.n_envs)
        dones = [False for _ in range(self.n_envs)]
        for _ in tqdm(range(train_steps)):
            step_info = {}
            self.obs_rms.update(obs)
            obs = self._process_observation(obs)
            acts, self.rnn_hidden = self._action(obs, self.egreedy, self.rnn_hidden)
            next_obs, rewards, terminals, trunctions, infos = self.envs.step(acts)

            if (self.current_step > self.start_training) and (self.current_step % self.train_frequency == 0):
                # training
                if self.memory.can_sample():
                    obs_batch, act_batch, rew_batch, terminal_batch = self.memory.sample()
                    step_info = self.learner.update(obs_batch, act_batch, rew_batch, terminal_batch)
                    step_info["epsilon-greedy"] = self.egreedy
                    self.log_infos(step_info, self.current_step)

            obs = next_obs
            for i in range(self.n_envs):
                episode_data[i].put([self._process_observation(obs[i]), acts[i], self._process_reward(rewards[i]), terminals[i]])
                if terminals[i] or trunctions[i]:
                    if self.atari and (~trunctions[i]):
                        pass
                    else:
                        self.rnn_hidden = self.policy.init_hidden_item(self.rnn_hidden, i)
                        dones[i] = True
                        self.current_episode[i] += 1
                        if self.use_wandb:
                            step_info["Episode-Steps/env-%d" % i] = infos[i]["episode_step"]
                            step_info["Train-Episode-Rewards/env-%d" % i] = infos[i]["episode_score"]
                        else:
                            step_info["Episode-Steps"] = {"env-%d" % i: infos[i]["episode_step"]}
                            step_info["Train-Episode-Rewards"] = {"env-%d" % i: infos[i]["episode_score"]}
                        self.log_infos(step_info, self.current_step)
                        self.memory.store(episode_data[i])
                        episode_data[i] = EpisodeBuffer()
                        obs[i] = infos[i]["reset_obs"]
                        episode_data[i].obs.append(self._process_observation(obs[i]))

            self.current_step += self.n_envs
            if self.egreedy > self.end_greedy:
                self.egreedy = self.egreedy - (self.start_greedy - self.end_greedy) / self.config.decay_step_greedy

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

        rnn_hidden = self.policy.init_hidden(num_envs)
        while current_episode < test_episodes:
            self.obs_rms.update(obs)
            obs = self._process_observation(obs)
            acts, rnn_hidden = self._action(obs, egreedy=0.0, rnn_hidden=rnn_hidden)
            next_obs, rewards, terminals, trunctions, infos = test_envs.step(acts)
            if self.config.render_mode == "rgb_array" and self.render:
                images = test_envs.render(self.config.render_mode)
                for idx, img in enumerate(images):
                    videos[idx].append(img)

            obs = next_obs
            for i in range(num_envs):
                if terminals[i] or trunctions[i]:
                    if self.atari and (~trunctions[i]):
                        pass
                    else:
                        obs[i] = infos[i]["reset_obs"]
                        rnn_hidden = self.policy.init_hidden_item(rnn_hidden, i)
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
