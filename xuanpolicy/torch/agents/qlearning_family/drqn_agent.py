from xuanpolicy.torch.agents import *
from collections import deque


class DRQN_Agent(Agent):
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

        self.observation_space = envs.observation_space
        self.action_space = envs.action_space
        self.representation_info_shape = policy.representation.output_shapes
        self.auxiliary_info_shape = {}
        self.sequence_length = config.sequence_length

        memory = RecurrentOffPolicyBuffer(self.observation_space,
                                          self.action_space,
                                          self.representation_info_shape,
                                          self.auxiliary_info_shape,
                                          self.nenvs,
                                          config.nsize,
                                          config.batchsize,
                                          episode_length=envs.max_episode_length,
                                          sequence_length=config.sequence_length,
                                          rnn_state_dim=config.recurrent_hidden_size)
        learner = DRQN_Learner(policy,
                               optimizer,
                               scheduler,
                               config.device,
                               config.modeldir,
                               config.gamma,
                               config.sync_frequency)

        self.obs_rms = RunningMeanStd(shape=space2shape(self.observation_space), comm=self.comm, use_mpi=False)
        self.ret_rms = RunningMeanStd(shape=(), comm=self.comm, use_mpi=False)
        super(DRQN_Agent, self).__init__(config, envs, policy, memory, learner, device, config.logdir, config.modeldir)

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

    def _action(self, obs, hidden=None, egreedy=0.0):
        _, argmax_action, _, hidden_n = self.policy(obs, hidden)
        random_action = np.random.choice(self.action_space.n, self.nenvs)
        if np.random.rand() < egreedy:
            action = random_action
        else:
            action = argmax_action.detach().cpu().numpy()
        return action, hidden_n.detach().cpu().numpy()

    def train(self, train_steps):
        obs = self.envs.buf_obs
        for i in range(self.nenvs):
            self.memory.episode_start(obs[i], i)
        for _ in tqdm(range(train_steps), position=1, desc="Step ", leave=False, colour='white'):
            step_info = {}
            self.obs_rms.update(obs)
            obs = self._process_observation(obs)
            acts, rnn_state = self._action(self.memory.obs_queue, self.memory.rnn_state_queue, self.egreedy)
            next_obs, rewards, terminals, trunctions, infos = self.envs.step(acts)

            self.memory.store(obs, rnn_state, acts, self._process_reward(rewards), terminals)
            if self.current_step > self.start_training and self.current_step % self.train_frequency == 0:
                # training
                obs_batch, rnn_batch, act_batch, rew_batch, next_batch, terminal_batch = self.memory.sample()
                step_info = self.learner.update(obs_batch, rnn_batch, act_batch, rew_batch, next_batch, terminal_batch)

            for i in range(self.nenvs):
                if terminals[i] or trunctions[i]:
                    if self.atari and (~trunctions[i]):
                        pass
                    else:
                        self.current_episode[i] += 1
                        self.memory.finish_path(i, infos[i]["episode_step"])
                        self.memory.episode_start(self.envs.buf_obs[i], i)
                        step_info["Episode-Steps"] = infos[i]["episode_step"]
                        if self.use_wandb:
                            step_info["Train-Episode-Rewards/env-%d" % i] = infos[i]["episode_score"]
                        else:
                            step_info["Train-Episode-Rewards"] = {"env-%d" % i: infos[i]["episode_score"]}
                        self.log_infos(step_info, self.current_step)

            obs = next_obs
            self.memory.local_append(obs, rnn_state)
            self.current_step += 1
            self.egreedy = self.egreedy - (self.start_greedy - self.end_greedy) / self.config.training_steps

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
            states, acts = self._action(obs, egreedy=0.0)
            next_obs, rewards, terminals, trunctions, infos = test_envs.step(acts)
            if self.config.render_mode == "rgb_array" and self.render:
                images = test_envs.render(self.config.render_mode)
                for idx, img in enumerate(images):
                    videos[idx].append(img)

            for i in range(num_envs):
                if terminals[i] or trunctions[i]:
                    if self.atari and (~trunctions[i]):
                        pass
                    else:
                        scores.append(infos[i]["episode_score"])
                        current_episode += 1
                        if best_score < infos[i]["episode_score"]:
                            best_score = infos[i]["episode_score"]
                            episode_videos = videos[i].copy()
                        if self.config.test_mode:
                            print("Episode: %d, Score: %.2f" % (current_episode, infos[i]["episode_score"]))
            obs = next_obs

        if self.config.render_mode == "rgb_array" and self.render:
            # time, height, width, channel -> time, channel, height, width
            videos_info = {"Videos_Test": np.array([episode_videos], dtype=np.uint8).transpose((0, 1, 4, 2, 3))}
            self.log_videos(info=videos_info, fps=50)

        if self.config.test_mode:
            print("Best Score: %.2f" % (best_score))

        test_info = {"Test-Episode-Rewards/Mean-Score": np.mean(scores)}
        self.log_infos(test_info, self.current_step)

        test_envs.close()

        return scores
