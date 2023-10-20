from xuance.tensorflow.agents import *


class PG_Agent(Agent):
    def __init__(self,
                 config: Namespace,
                 envs: VecEnv,
                 policy: tk.Model,
                 optimizer: tk.optimizers.Optimizer,
                 device: str = 'cpu'):
        self.render = config.render
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

        self.atari = True if config.env_name == "Atari" else False
        Buffer = DummyOnPolicyBuffer_Atari if self.atari else DummyOnPolicyBuffer

        memory = Buffer(self.observation_space,
                        self.action_space,
                        self.representation_info_shape,
                        self.auxiliary_info_shape,
                        self.nenvs,
                        self.nsteps,
                        self.nminibatch,
                        self.gamma,
                        self.lam)
        learner = PG_Learner(policy,
                             optimizer,
                             config.device,
                             config.modeldir,
                             config.ent_coef,
                             config.clip_grad)
        super(PG_Agent, self).__init__(config, envs, policy, memory, learner, device, config.logdir, config.modeldir)

    def _action(self, obs):
        _, _ = self.policy(obs)
        acts = self.policy.actor.dist.stochastic_sample()
        acts = acts.numpy()
        return acts

    def train(self, train_steps):
        obs = self.envs.buf_obs
        for _ in tqdm(range(train_steps)):
            step_info = {}
            self.obs_rms.update(obs)
            obs = self._process_observation(obs)
            acts = self._action(obs)
            next_obs, rewards, terminals, trunctions, infos = self.envs.step(acts)
            self.memory.store(obs, acts, self._process_reward(rewards), 0, terminals)
            if self.memory.full:
                for i in range(self.nenvs):
                    self.memory.finish_path(rewards[i], i)
                for _ in range(self.nminibatch * self.nepoch):
                    obs_batch, act_batch, ret_batch, _, _ = self.memory.sample()
                    step_info = self.learner.update(obs_batch, act_batch, ret_batch)
                self.memory.clear()

            self.returns = self.gamma * self.returns + rewards
            obs = next_obs
            for i in range(self.nenvs):
                if terminals[i] or trunctions[i]:
                    if self.atari and (~trunctions[i]):
                        pass
                    else:
                        obs[i] = infos[i]["reset_obs"]
                        self.ret_rms.update(self.returns[i:i + 1])
                        self.returns[i] = 0.0
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
            acts = self._action(obs)
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
