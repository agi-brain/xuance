from xuance.tensorflow.agents import *


class PPG_Agent(Agent):
    def __init__(self,
                 config: Namespace,
                 envs: DummyVecEnv_Gym,
                 policy: tk.Model,
                 optimizer: tk.optimizers.Optimizer,
                 device: str = 'cpu'):
        self.render = config.render
        self.n_envs = envs.num_envs
        self.n_steps = config.n_steps
        self.n_minibatch = config.n_minibatch
        self.n_epoch = config.n_epoch
        self.policy_nepoch = config.policy_nepoch
        self.value_nepoch = config.value_nepoch
        self.aux_nepoch = config.aux_nepoch

        self.gamma = config.gamma
        self.gae_lam = config.gae_lambda
        self.observation_space = envs.observation_space
        self.action_space = envs.action_space
        self.representation_info_shape = policy.actor_representation.output_shapes
        self.auxiliary_info_shape = {"old_dist": None}

        self.buffer_size = self.n_envs * self.n_steps
        self.batch_size = self.buffer_size // self.n_epoch
        memory = DummyOnPolicyBuffer(self.observation_space,
                                     self.action_space,
                                     self.auxiliary_info_shape,
                                     self.n_envs,
                                     self.n_steps,
                                     config.use_gae,
                                     config.use_advnorm,
                                     self.gamma,
                                     self.gae_lam)
        learner = PPG_Learner(policy,
                              optimizer,
                              config.device,
                              config.model_dir,
                              config.ent_coef,
                              config.clip_range,
                              config.kl_beta)
        super(PPG_Agent, self).__init__(config, envs, policy, memory, learner, device, config.log_dir, config.model_dir)

    def _action(self, obs):
        _, _, vs, _ = self.policy(obs)
        dists = self.policy.actor.dist
        acts = dists.stochastic_sample()
        vs = vs.numpy()
        acts = acts.numpy()
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
                for i in range(self.n_envs):
                    self.memory.finish_path(vals[i], i)
                # policy update
                indexes = np.arange(self.buffer_size)
                for _ in range(self.policy_nepoch):
                    np.random.shuffle(indexes)
                    for start in range(0, self.buffer_size, self.batch_size):
                        end = start + self.batch_size
                        sample_idx = indexes[start:end]
                        obs_batch, act_batch, ret_batch, _, adv_batch, aux_batch = self.memory.sample(sample_idx)
                        step_info.update(self.learner.update_policy(obs_batch, act_batch, ret_batch, adv_batch,
                                                                    aux_batch['old_dist']))
                # critic update
                for _ in range(self.value_nepoch):
                    np.random.shuffle(indexes)
                    for start in range(0, self.buffer_size, self.batch_size):
                        end = start + self.batch_size
                        sample_idx = indexes[start:end]
                        obs_batch, act_batch, ret_batch, _, adv_batch, aux_batch = self.memory.sample(sample_idx)
                        step_info.update(self.learner.update_critic(obs_batch, act_batch, ret_batch, adv_batch,
                                                                    aux_batch['old_dist']))

                # update old_prob
                buffer_obs_shape = self.memory.observations.shape
                buffer_obs = self.memory.observations.reshape([-1, buffer_obs_shape[-1]])
                buffer_act = self.memory.actions
                _, new_logits, _, _ = self.policy(buffer_obs)
                try:
                    self.policy.actor.dist.set_param(tf.reshape(new_logits, buffer_obs_shape[0:-1] + (-1,)))
                except:
                    new_std = tf.math.exp(self.policy.actor.logstd)
                    self.policy.actor.dist.set_param(tf.reshape(new_logits, buffer_obs_shape[0:-1] + (-1,)), new_std)
                new_dist = self.policy.actor.dist
                self.memory.auxiliary_infos['old_dist'] = split_distributions(new_dist)
                for _ in range(self.aux_nepoch):
                    np.random.shuffle(indexes)
                    for start in range(0, self.buffer_size, self.batch_size):
                        end = start + self.batch_size
                        sample_idx = indexes[start:end]
                        obs_batch, act_batch, ret_batch, _, adv_batch, aux_batch = self.memory.sample(sample_idx)
                        step_info.update(self.learner.update_auxiliary(obs_batch, act_batch, ret_batch, adv_batch,
                                                                       aux_batch['old_dist']))
                self.log_infos(step_info, self.current_step)
                self.memory.clear()

            obs = next_obs
            for i in range(self.n_envs):
                if terminals[i] or trunctions[i]:
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

            self.current_step += self.n_envs

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
            acts, rets, logps = self._action(obs)
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
