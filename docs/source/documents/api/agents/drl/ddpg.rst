DDPG_Agent
=====================================

.. raw:: html

    <br><hr>

PyTorch
------------------------------------------

.. py:class:: 
  xuance.torch.agent.policy_gradient.ddpg_agent.DDPG_Agent(config, envs, policy, optimizer, scheduler, device)

  :param config: Provides hyper parameters.
  :type config: Namespace
  :param envs: The vectorized environments.
  :type envs: xuance.environments.vector_envs.vector_env.VecEnv
  :param policy: The policy that provides actions and values.
  :type policy: nn.Module
  :param optimizer: The optimizers of actor and critic that update the parameters.
  :type optimizer: list[torch.optim.Optimizer]
  :param scheduler: Implement the learning rate decay.
  :type scheduler: torch.optim.lr_scheduler._LRScheduler
  :param device: Choose CPU or GPU to train the model.
  :type device: str, int, torch.device

.. py:function:: 
  xuance.torch.agent.policy_gradient.ddpg_agent.DDPG_Agent._action(obs, noise_scale)
  
  Calculate actions according to the observations.

  :param obs: The observation of current step.
  :type obs: np.ndarray
  :param noise_scale: The variance of Gaussian noises.
  :type noise_scale: float
  :return: **action** - The actions to be executed.
  :rtype: np.ndarray
  
.. py:function:: 
  xuance.torch.agent.policy_gradient.ddpg_agent.DDPG_Agent.train(train_steps)
  
  Train the DDPG agent.

  :param train_steps: The number of steps for training.
  :type train_steps: int

.. py:function:: 
  xuance.torch.agent.policy_gradient.ddpg_agent.DDPG_Agent.test(env_fn, test_episodes)
  
  Test the trained model.

  :param env_fn: The function of making environments.
  :param test_episodes: The number of testing episodes.
  :type test_episodes: int
  :return: **scores** - The accumulated scores of these episodes.
  :rtype: list

.. raw:: html

    <br><hr>

TensorFlow
------------------------------------------

.. py:class::
  xuance.tensorflow.agent.policy_gradient.ddpg_agent.DDPG_Agent(config, envs, policy, optimizer, device)

  :param config: Provides hyper parameters.
  :type config: Namespace
  :param envs: The vectorized environments.
  :type envs: xuance.environments.vector_envs.vector_env.VecEnv
  :param policy: The policy that provides actions and values.
  :type policy: nn.Module
  :param optimizer: The optimizers of actor and critic that update the parameters.
  :type optimizer: list[torch.optim.Optimizer]
  :param device: Choose CPU or GPU to train the model.
  :type device: str, int, torch.device

.. py:function::
  xuance.tensorflow.agent.policy_gradient.ddpg_agent.DDPG_Agent._action(obs, noise_scale)

  Calculate actions according to the observations.

  :param obs: The observation of current step.
  :type obs: np.ndarray
  :param noise_scale: The variance of Gaussian noises.
  :type noise_scale: float
  :return: **action** - The actions to be executed.
  :rtype: np.ndarray

.. py:function::
  xuance.tensorflow.agent.policy_gradient.ddpg_agent.DDPG_Agent.train(train_steps)

  Train the DDPG agent.

  :param train_steps: The number of steps for training.
  :type train_steps: int

.. py:function::
  xuance.tensorflow.agent.policy_gradient.ddpg_agent.DDPG_Agent.test(env_fn, test_episodes)

  Test the trained model.

  :param env_fn: The function of making environments.
  :param test_episodes: The number of testing episodes.
  :type test_episodes: int
  :return: **scores** - The accumulated scores of these episodes.
  :rtype: list

.. raw:: html

    <br><hr>

MindSpore
------------------------------------------

.. py:class::
    xuance.mindspore.agents.policy_gradient.ddpg_agent.DDPG_Agent(config, envs, policy, optimizer, scheduler)

    :param config: Provides hyper parameters.
    :type config: Namespace
    :param envs: The vectorized environments.
    :type envs: xuance.environments.vector_envs.vector_env.VecEnv
    :param policy: The policy that provides actions and values.
    :type policy: nn.Module
    :param optimizer: The optimizer that updates the parameters.
    :type optimizer: torch.optim.Optimizer
    :param scheduler: Implement the learning rate decay.
    :type scheduler: torch.optim.lr_scheduler._LRScheduler

.. py:function::
    xuance.mindspore.agents.policy_gradient.ddpg_agent.DDPG_Agent._action(obs,noise_scale)

    :param obs: The observation variables.
    :type obs: np.ndarray
    :param noise_scale: The scale value of the Gaussian noise.
    :type noise_scale: float
    :return: the selected actions after applying exploration noise and clipping.
    :rtype: np.ndarray

.. py:function::
    xuance.mindspore.agents.policy_gradient.ddpg_agent.DDPG_Agent.train(train_steps)

    :param train_steps: The number of steps for training.
    :type train_steps: int

.. py:function::
    xuance.mindspore.agents.policy_gradient.ddpg_agent.DDPG_Agent.test(env_fn,test_episodes)

    :param env_fn: The function of making environments.
    :param test_episodes: The number of testing episodes.
    :type test_episodes: int
    :return: The accumulated scores of these episodes.
    :rtype: list

.. raw:: html

    <br><hr>

Source Code
-----------------

.. tabs::
  
  .. group-tab:: PyTorch
    
    .. code-block:: python

        from xuance.torch.agents import *


        class DDPG_Agent(Agent):
            """The implementation of DDPG agent.

            Args:
                config: the Namespace variable that provides hyper-parameters and other settings.
                envs: the vectorized environments.
                policy: the neural network modules of the agent.
                optimizer: the method of optimizing.
                scheduler: the learning rate decay scheduler.
                device: the calculating device of the model, such as CPU or GPU.
            """
            def __init__(self,
                        config: Namespace,
                        envs: DummyVecEnv_Gym,
                        policy: nn.Module,
                        optimizer: Sequence[torch.optim.Optimizer],
                        scheduler: Optional[Sequence[torch.optim.lr_scheduler._LRScheduler]] = None,
                        device: Optional[Union[int, str, torch.device]] = None):
                self.render = config.render
                self.n_envs = envs.num_envs

                self.gamma = config.gamma
                self.train_frequency = config.training_frequency
                self.start_training = config.start_training
                self.start_noise = config.start_noise
                self.end_noise = config.end_noise
                self.noise_scale = config.start_noise
                self.delta_noise = (self.start_noise - self.end_noise) / (config.running_steps / self.n_envs)

                self.observation_space = envs.observation_space
                self.action_space = envs.action_space
                self.auxiliary_info_shape = {}

                memory = DummyOffPolicyBuffer(self.observation_space,
                                            self.action_space,
                                            self.auxiliary_info_shape,
                                            self.n_envs,
                                            config.n_size,
                                            config.batch_size)
                learner = DDPG_Learner(policy,
                                    optimizer,
                                    scheduler,
                                    config.device,
                                    config.model_dir,
                                    config.gamma,
                                    config.tau)
                super(DDPG_Agent, self).__init__(config, envs, policy, memory, learner, device,
                                                config.log_dir, config.model_dir)

            def _action(self, obs, noise_scale=0.0):
                _, action = self.policy(obs)
                action = action.detach().cpu().numpy()
                action = action + np.random.normal(size=action.shape) * noise_scale
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
                        self.noise_scale = self.noise_scale - self.delta_noise

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

    .. code-block:: python

        from xuance.tensorflow.agents import *


        class DDPG_Agent(Agent):
            def __init__(self,
                         config: Namespace,
                         envs: DummyVecEnv_Gym,
                         policy: tk.Model,
                         optimizer: Sequence[tk.optimizers.Optimizer],
                         device: str = 'cpu'):
                self.render = config.render
                self.n_envs = envs.num_envs

                self.gamma = config.gamma
                self.train_frequency = config.training_frequency
                self.start_training = config.start_training
                self.start_noise = config.start_noise
                self.end_noise = config.end_noise
                self.noise_scale = config.start_noise
                self.delta_noise = (self.start_noise - self.end_noise) / (config.running_steps / self.n_envs)

                self.observation_space = envs.observation_space
                self.action_space = envs.action_space
                self.auxiliary_info_shape = {}

                memory = DummyOffPolicyBuffer(self.observation_space,
                                              self.action_space,
                                              self.auxiliary_info_shape,
                                              self.n_envs,
                                              config.n_size,
                                              config.batch_size)
                learner = DDPG_Learner(policy,
                                       optimizer,
                                       config.device,
                                       config.model_dir,
                                       config.gamma,
                                       config.tau)
                super(DDPG_Agent, self).__init__(config, envs, policy, memory, learner, device, config.log_dir, config.model_dir)

            def _action(self, obs, noise_scale=0.0):
                _, action = self.policy(obs)
                action = action.numpy()
                action = action + np.random.normal(size=action.shape) * noise_scale
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
                        self.noise_scale = self.noise_scale - self.delta_noise

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


  .. group-tab:: MindSpore

    .. code-block:: python

        from xuance.mindspore.agents import *


        class DDPG_Agent(Agent):
            def __init__(self,
                         config: Namespace,
                         envs: DummyVecEnv_Gym,
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
                self.delta_noise = (self.start_noise - self.end_noise) / (config.running_steps / self.n_envs)

                self.observation_space = envs.observation_space
                self.action_space = envs.action_space
                self.auxiliary_info_shape = {}

                self.observation_space = envs.observation_space
                self.action_space = envs.action_space
                self.representation_info_shape = policy.representation.output_shapes
                self.auxiliary_info_shape = {}

                memory = DummyOffPolicyBuffer(self.observation_space,
                                              self.action_space,
                                              self.auxiliary_info_shape,
                                              self.n_envs,
                                              config.n_size,
                                              config.batch_size)
                learner = DDPG_Learner(policy,
                                       optimizer,
                                       scheduler,
                                       config.model_dir,
                                       config.gamma,
                                       config.tau)
                super(DDPG_Agent, self).__init__(config, envs, policy, memory, learner, config.log_dir, config.model_dir)

            def _action(self, obs, noise_scale=0.0):
                _, action = self.policy(ms.Tensor(obs))
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
                        self.noise_scale = self.noise_scale - self.delta_noise

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
