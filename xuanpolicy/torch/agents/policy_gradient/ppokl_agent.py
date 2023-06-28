from xuanpolicy.torch.agents import *


class PPOKL_Agent(Agent):
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

        self.observation_space = envs.observation_space
        self.action_space = envs.action_space
        self.representation_info_shape = policy.representation_actor.output_shapes
        self.auxiliary_info_shape = {"old_dist": None}

        memory = DummyOnPolicyBuffer(self.observation_space,
                                     self.action_space,
                                     self.representation_info_shape,
                                     self.auxiliary_info_shape,
                                     self.nenvs,
                                     self.nsteps,
                                     self.nminibatch,
                                     self.gamma,
                                     self.lam)
        learner = PPOKL_Learner(policy,
                                optimizer,
                                scheduler,
                                config.device,
                                config.modeldir,
                                config.vf_coef,
                                config.ent_coef,
                                config.target_kl)

        self.obs_rms = RunningMeanStd(shape=space2shape(self.observation_space), comm=self.comm, use_mpi=False)
        self.ret_rms = RunningMeanStd(shape=(), comm=self.comm, use_mpi=False)
        super(PPOKL_Agent, self).__init__(config, envs, policy, memory, learner, device, config.logdir,
                                          config.modeldir)
        if self.atari:
            self.memory = DummyOnPolicyBuffer_Atari(self.observation_space,
                                                    self.action_space,
                                                    self.representation_info_shape,
                                                    self.auxiliary_info_shape,
                                                    self.nenvs,
                                                    self.nsteps,
                                                    self.nminibatch,
                                                    self.gamma,
                                                    self.lam)

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
        _, dists, vs = self.policy(obs)
        acts = dists.stochastic_sample()
        vs = vs.detach().cpu().numpy()
        acts = acts.detach().cpu().numpy()
        return acts, vs, split_distributions(dists)

    def train(self, train_steps):
        obs = self.envs.buf_obs
        for _ in tqdm(range(train_steps)):
            step_info = {}
            self.obs_rms.update(obs)
            obs = self._process_observation(obs)
            acts, rets, dists = self._action(obs)
            next_obs, rewards, terminals, trunctions, infos = self.envs.step(acts)

            self.memory.store(obs, acts, self._process_reward(rewards), rets, terminals, {"old_dist": dists})
            if self.memory.full:
                _, vals, _ = self._action(self._process_observation(next_obs))
                for i in range(self.nenvs):
                    self.memory.finish_path(vals[i], i)
                for _ in range(self.nminibatch * self.nepoch):
                    obs_batch, act_batch, ret_batch, adv_batch, aux_batch = self.memory.sample()
                    step_info = self.learner.update(obs_batch, act_batch, ret_batch, adv_batch, aux_batch['old_dist'])
                self.memory.clear()
                self.log_infos(step_info, self.current_step)

            obs = next_obs
            for i in range(self.nenvs):
                if terminals[i] or trunctions[i]:
                    if self.atari and (~trunctions[i]):
                        pass
                    else:
                        obs[i] = infos[i]["reset_obs"]
                        self.memory.finish_path(0, i)
                        self.current_episode[i] += 1
                        if self.use_wandb:
                            step_info["Episode-Steps/env-%d" % i] = infos[i]["episode_step"]
                            step_info["Train-Episode-Rewards/env-%d" % i] = infos[i]["episode_score"]
                        else:
                            step_info["Episode-Steps"] = {"env-%d" % i: infos[i]["episode_step"]}
                            step_info["Train-Episode-Rewards"] = {"env-%d" % i: infos[i]["episode_score"]}
                        self.log_infos(step_info, self.current_step)

            self.current_step += 1

    def test(self, env_fn, test_episode):
        test_envs = env_fn()
        num_envs = test_envs.num_envs
        videos, episode_videos = [[] for _ in range(num_envs)], []
        current_episode, scores, best_score = 0, [], -np.inf
        obs, infos = test_envs.reset()
        if self.config.render_mode == "rgb_array" and self.render:
            images = test_envs.render(self.config.render_mode)
            for idx, img in enumerate(images):
                videos[idx].append(img)

        while current_episode < test_episode:
            self.obs_rms.update(obs)
            obs = self._process_observation(obs)
            acts, rets, logps = self._action(obs)
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
            self.log_videos(info=videos_info, fps=50, x_index=self.current_step)

        if self.config.test_mode:
            print("Best Score: %.2f" % (best_score))

        test_info = {"Test-Episode-Rewards/Mean-Score": np.mean(scores)}
        self.log_infos(test_info, self.current_step)

        test_envs.close()

        return scores
