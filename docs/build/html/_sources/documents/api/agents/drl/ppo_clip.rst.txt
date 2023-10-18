PPOCLIP_Agent
=====================================

.. raw:: html

    <br><hr>

**PyTorch:**

.. py:class:: 
  xuanpolicy.torch.agent.policy_gradient.ppoclip_agent.PPOCLIP_Agent(config, envs, policy, optimizer, scheduler, device)

  :param config: Provides hyper parameters.
  :type config: Namespace
  :param envs: The vectorized environments.
  :type envs: xuanpolicy.environments.vector_envs.vector_env.VecEnv
  :param policy: The policy that provides actions and values.
  :type policy: nn.Module
  :param optimizer: The optimizer that updates the parameters.
  :type optimizer: torch.optim.Optimizer
  :param scheduler: Implement the learning rate decay.
  :type scheduler: torch.optim.lr_scheduler._LRScheduler
  :param device: Choose CPU or GPU to train the model.
  :type device: str, int, torch.device

.. py:function:: 
  xuanpolicy.torch.agent.policy_gradient.ppoclip_agent.PPOCLIP_Agent._action(obs)
  
  Calculate actions according to the observations.

  :param obs: The observation of current step.
  :type obs: numpy.ndarray
  :return: **action**, **vs**, **logps** - The actions to be executed. The values calculated by critic network. Log of probabilities of actions.
  :rtype: np.ndarray, np.ndarray
  
.. py:function:: 
  xuanpolicy.torch.agent.policy_gradient.ppoclip_agent.PPOCLIP_Agent.train(train_steps)
  
  Train the PPO agent.

  :param train_steps: The number of steps for training.
  :type train_steps: int

.. py:function:: 
  xuanpolicy.torch.agent.policy_gradient.ppoclip_agent.PPOCLIP_Agent.test(env_fn, test_episodes)
  
  Test the trained model.

  :param env_fn: The function of making environments.
  :param test_episodes: The number of testing episodes.
  :type test_episodes: int
  :return: **scores** - The accumulated scores of these episodes.
  :rtype: list

.. raw:: html

    <br><hr>

**TensorFlow:**

.. raw:: html

    <br><hr>

**MindSpore:**

.. raw:: html

    <br><hr>

源码
-----------------

.. tabs::
  
  .. group-tab:: PyTorch
    
    .. code-block:: python3

        from xuanpolicy.torch.agents import *

        class PPOCLIP_Agent(Agent):
            def __init__(self,
                        config: Namespace,
                        envs: DummyVecEnv_Gym,
                        policy: nn.Module,
                        optimizer: torch.optim.Optimizer,
                        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
                        device: Optional[Union[int, str, torch.device]] = None):
                self.render = config.render
                self.n_envs = envs.num_envs
                self.n_steps = config.n_steps
                self.n_minibatch = config.n_minibatch
                self.n_epoch = config.n_epoch

                self.gamma = config.gamma
                self.gae_lam = config.gae_lambda
                self.observation_space = envs.observation_space
                self.action_space = envs.action_space
                self.auxiliary_info_shape = {"old_logp": ()}

                self.atari = True if config.env_name == "Atari" else False
                Buffer = DummyOnPolicyBuffer_Atari if self.atari else DummyOnPolicyBuffer
                self.buffer_size = self.n_envs * self.n_steps
                self.batch_size = self.buffer_size // self.n_minibatch
                memory = Buffer(self.observation_space,
                                self.action_space,
                                self.auxiliary_info_shape,
                                self.n_envs,
                                self.n_steps,
                                config.use_gae,
                                config.use_advnorm,
                                self.gamma,
                                self.gae_lam)
                learner = PPOCLIP_Learner(policy,
                                        optimizer,
                                        scheduler,
                                        config.device,
                                        config.model_dir,
                                        vf_coef=config.vf_coef,
                                        ent_coef=config.ent_coef,
                                        clip_range=config.clip_range,
                                        clip_grad_norm=config.clip_grad_norm,
                                        use_grad_clip=config.use_grad_clip)
                super(PPOCLIP_Agent, self).__init__(config, envs, policy, memory, learner, device,
                                                    config.log_dir, config.model_dir)

            def _action(self, obs):
                _, dists, vs = self.policy(obs)
                acts = dists.stochastic_sample()
                logps = dists.log_prob(acts)
                vs = vs.detach().cpu().numpy()
                acts = acts.detach().cpu().numpy()
                logps = logps.detach().cpu().numpy()
                return acts, vs, logps

            def train(self, train_steps):
                obs = self.envs.buf_obs
                for _ in tqdm(range(train_steps)):
                    step_info = {}
                    self.obs_rms.update(obs)
                    obs = self._process_observation(obs)
                    acts, value, logps = self._action(obs)
                    next_obs, rewards, terminals, trunctions, infos = self.envs.step(acts)

                    self.memory.store(obs, acts, self._process_reward(rewards), value, terminals, {"old_logp": logps})
                    if self.memory.full:
                        _, vals, _ = self._action(self._process_observation(next_obs))
                        for i in range(self.n_envs):
                            if terminals[i]:
                                self.memory.finish_path(0.0, i)
                            else:
                                self.memory.finish_path(vals[i], i)
                        indexes = np.arange(self.buffer_size)
                        for _ in range(self.n_epoch):
                            np.random.shuffle(indexes)
                            for start in range(0, self.buffer_size, self.batch_size):
                                end = start + self.batch_size
                                sample_idx = indexes[start:end]
                                obs_batch, act_batch, ret_batch, value_batch, adv_batch, aux_batch = self.memory.sample(sample_idx)
                                step_info = self.learner.update(obs_batch, act_batch, ret_batch, value_batch, adv_batch, aux_batch['old_logp'])
                        self.log_infos(step_info, self.current_step)
                        self.memory.clear()

                    self.returns = (1 - terminals) * self.gamma * self.returns + rewards
                    obs = next_obs
                    for i in range(self.n_envs):
                        if terminals[i] or trunctions[i]:
                            self.ret_rms.update(self.returns[i:i + 1])
                            self.returns[i] = 0.0
                            if self.atari and (~trunctions[i]):
                                pass
                            else:
                                if terminals[i]:
                                    self.memory.finish_path(0.0, i)
                                else:
                                    _, vals, _ = self._action(self._process_observation(next_obs))
                                    self.memory.finish_path(vals[i], i)
                                obs[i] = infos[i]["reset_obs"]
                                self.current_episode[i] += 1
                                if self.use_wandb:
                                    step_info["Episode-Steps/env-%d" % i] = infos[i]["episode_step"]
                                    step_info["Train-Episode-Rewards/env-%d" % i] = infos[i]["episode_score"]
                                else:
                                    step_info["Episode-Steps"] = {"env-%d" % i: infos[i]["episode_step"]}
                                    step_info["Train-Episode-Rewards"] = {"env-%d" % i: infos[i]["episode_score"]}
                                self.log_infos(step_info, self.current_step)

                    self.current_step += self.n_envs

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

                test_info = {
                    "Test-Episode-Rewards/Mean-Score": np.mean(scores),
                    "Test-Episode-Rewards/Std-Score": np.std(scores)
                }
                self.log_infos(test_info, self.current_step)

                test_envs.close()

                return scores




  .. group-tab:: TensorFlow

    .. code-block:: python3

  .. group-tab:: MindSpore

    .. code-block:: python3