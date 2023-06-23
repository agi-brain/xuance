import numpy as np

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

        if config.max_episode_length == -1:
            self.max_episode_length = envs.max_episode_length
        else:
            self.max_episode_length = config.max_episode_length
        memory = RecurrentOffPolicyBuffer(self.observation_space,
                                          self.action_space,
                                          self.representation_info_shape,
                                          self.auxiliary_info_shape,
                                          self.nenvs,
                                          config.nsize,
                                          config.batchsize,
                                          episode_length=self.max_episode_length)
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
        self.current_episode = 0

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

    def _action(self, obs, egreedy=0.0):
        _, argmax_action, _ = self.policy(obs[:, np.newaxis])
        random_action = np.random.choice(self.action_space.n, self.nenvs)
        if np.random.rand() < egreedy:
            action = random_action
        else:
            action = argmax_action.detach().cpu().numpy()
        return action

    def run_episode(self):
        episode_info = {}
        obs, infos = self.envs.reset()
        dones = [False for _ in range(self.nenvs)]
        obs_queue = np.zeros((self.nenvs, self.max_episode_length + 1) + self.observation_space.shape)
        obs_queue[:, 0] = self._process_observation(obs)
        act_queue = np.zeros((self.nenvs, self.max_episode_length) + space2shape(self.action_space))
        rew_queue = np.zeros((self.nenvs, self.max_episode_length, ))
        terminal_queue = np.zeros((self.nenvs, self.max_episode_length, ), np.bool)
        filled_queue = np.zeros((self.nenvs, self.max_episode_length, ), np.bool)
        step = 0
        self.policy.init_hidden(self.nenvs)
        while not all(dones):
            self.obs_rms.update(obs)
            obs = self._process_observation(obs)
            acts = self._action(obs, self.egreedy)
            next_obs, rewards, terminals, trunctions, infos = self.envs.step(acts)

            for i in range(self.nenvs):
                if not dones[i]:
                    filled_queue[i, step] = 1
                if terminals[i] or trunctions[i] or step >= (self.max_episode_length-1):
                    if self.atari and (~trunctions[i]):
                        pass
                    else:
                        self.policy.init_hidden(self.nenvs, i)
                        dones[i] = True
                        if self.use_wandb:
                            episode_info["Episode-Steps/env-%d" % i] = infos[i]["episode_step"]
                            episode_info["Train-Episode-Rewards/env-%d" % i] = infos[i]["episode_score"]
                        else:
                            episode_info["Episode-Steps"] = {"env-%d" % i: infos[i]["episode_step"]}
                            episode_info["Train-Episode-Rewards"] = {"env-%d" % i: infos[i]["episode_score"]}

            obs = next_obs
            obs_queue[:, step + 1], act_queue[:, step] = self._process_observation(obs), acts
            rew_queue[:, step], terminal_queue[:, step] = self._process_reward(rewards), terminals
            self.current_step += 1
            step += 1

        return obs_queue, act_queue, rew_queue, terminal_queue, filled_queue, episode_info

    def train(self, train_episodes):
        train_info = {}
        for _ in tqdm(range(train_episodes)):
            obs_queue, act_queue, rew_queue, terminal_queue, filled_queue, episode_info = self.run_episode()

            self.memory.store(obs_queue, act_queue, rew_queue, terminal_queue, filled_queue)
            if self.current_step > self.start_training and self.current_step % self.train_frequency == 0:
                # training
                for _ in range(self.config.n_trains_per_episode):
                    obs_batch, act_batch, rew_batch, terminal_batch, fill_batch = self.memory.sample()
                    train_info = self.learner.update(obs_batch, act_batch, rew_batch, terminal_batch, fill_batch)

            self.current_episode += 1
            if self.egreedy > self.end_greedy:
                self.egreedy = self.egreedy - (self.start_greedy - self.end_greedy) / self.config.decay_episodes_greedy
            episode_info["epsilon-greedy"] = self.egreedy

            self.log_infos(episode_info, self.current_episode)
            self.log_infos(train_info, self.current_episode)

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

        self.policy.init_hidden(num_envs)
        while current_episode < test_episodes:
            self.obs_rms.update(obs)
            obs = self._process_observation(obs)
            acts = self._action(obs, egreedy=0.0)
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
                        self.policy.init_hidden(num_envs, i)
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
            self.log_videos(info=videos_info, fps=50, x_index=self.current_episode)

        if self.config.test_mode:
            print("Best Score: %.2f" % (best_score))

        test_info = {"Test-Episode-Rewards/Mean-Score": np.mean(scores)}
        self.log_infos(test_info, self.current_episode)

        test_envs.close()

        return scores
