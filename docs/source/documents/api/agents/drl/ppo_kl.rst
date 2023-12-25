PPOKL_Agent
=====================================

.. raw:: html

    <br><hr>

PyTorch
------------------------------------------

.. py:class::
  xuance.torch.agent.policy_gradient.ppokl_agent.PPOKL_Agent(config, envs, policy, optimizer, scheduler, device)

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
  :param device: Choose CPU or GPU to train the model.
  :type device: str, int, torch.device

.. py:function:: 
  xuance.torch.agent.policy_gradient.ppokl_agent.PPOKL_Agent._action(obs)
  
  Calculate actions according to the observations.

  :param obs: The observation of current step.
  :type obs: np.ndarray
  :return: **action**, **vs**, **dist** - The actions to be executed. The values calculated by critic network. The distributions of the stochastic policy.
  :rtype: np.ndarray, np.ndarray
  
.. py:function:: 
  xuance.torch.agent.policy_gradient.ppokl_agent.PPOKL_Agent.train(train_steps)
  
  Train the PPO agent.

  :param train_steps: The number of steps for training.
  :type train_steps: int

.. py:function:: 
  xuance.torch.agent.policy_gradient.ppokl_agent.PPOKL_Agent.test(env_fn, test_episodes)
  
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
  xuance.tensorflow.agent.policy_gradient.ppokl_agent.PPOKL_Agent(config, envs, policy, optimizer, device)

  :param config: Provides hyper parameters.
  :type config: Namespace
  :param envs: The vectorized environments.
  :type envs: xuance.environments.vector_envs.vector_env.VecEnv
  :param policy: The policy that provides actions and values.
  :type policy: nn.Module
  :param optimizer: The optimizer that updates the parameters.
  :type optimizer: torch.optim.Optimizer
  :param device: Choose CPU or GPU to train the model.
  :type device: str, int, torch.device

.. py:function::
  xuance.tensorflow.agent.policy_gradient.ppokl_agent.PPOKL_Agent._action(obs)

  Calculate actions according to the observations.

  :param obs: The observation of current step.
  :type obs: np.ndarray
  :return: **action**, **vs**, **dist** - The actions to be executed. The values calculated by critic network. The distributions of the stochastic policy.
  :rtype: np.ndarray, np.ndarray

.. py:function::
  xuance.tensorflow.agent.policy_gradient.ppokl_agent.PPOKL_Agent.train(train_steps)

  Train the PPO agent.

  :param train_steps: The number of steps for training.
  :type train_steps: int

.. py:function::
  xuance.tensorflow.agent.policy_gradient.ppokl_agent.PPOKL_Agent.test(env_fn, test_episode)

  Test the trained model.

  :param env_fn: The function of making environments.
  :param test_episode: The number of testing episodes.
  :type test_episode: int
  :return: **scores** - The accumulated scores of these episodes.
  :rtype: list

.. raw:: html

    <br><hr>

MindSpore
------------------------------------------

.. py:class::
    xuance.mindspore.agents.policy_gradient.ppokl_agent.PPOKL_Agent(config, envs, policy, optimizer, scheduler)

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
    xuance.mindspore.agents.policy_gradient.ppokl_agent.PPOKL_Agent._action(obs)

    :param obs: The observation variables.
    :type obs: np.ndarray
    :return: A tuple that includes the hidden states, executable actions, values, and log of actions.
    :rtype: tuple

.. py:function::
    xuance.mindspore.agents.policy_gradient.ppokl_agent.PPOKL_Agent.train(train_steps)

    :param train_steps: The number of steps for training.
    :type train_steps: int

.. py:function::
    xuance.mindspore.agents.policy_gradient.ppokl_agent.PPOKL_Agent.test(test_episodes)

    :param test_episodes: The number of testing episodes.
    :type test_episodes: int

.. py:function::
    xuance.mindspore.agents.policy_gradient.ppokl_agent.PPOKL_Agent.evaluate()

.. raw:: html

    <br><hr>

Source Code
-----------------

.. tabs::
  
  .. group-tab:: PyTorch
    
    .. code-block:: python

        from xuance.torch.agents import *


        class PPOKL_Agent(Agent):
            """The implementation of PPO agent with KL divergence.

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
                self.representation_info_shape = policy.representation_actor.output_shapes
                self.auxiliary_info_shape = {"old_dist": None}

                self.atari = True if config.env_name == "Atari" else False
                Buffer = DummyOnPolicyBuffer_Atari if self.atari else DummyOnPolicyBuffer_Atari
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
                learner = PPOKL_Learner(policy,
                                        optimizer,
                                        scheduler,
                                        config.device,
                                        config.model_dir,
                                        config.vf_coef,
                                        config.ent_coef,
                                        config.target_kl)
                super(PPOKL_Agent, self).__init__(config, envs, policy, memory, learner, device,
                                                config.log_dir, config.model_dir)

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
                    acts, values, dists = self._action(obs)
                    next_obs, rewards, terminals, trunctions, infos = self.envs.step(acts)

                    self.memory.store(obs, acts, self._process_reward(rewards), values, terminals, {"old_dist": dists})
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
                                obs_batch, act_batch, ret_batch, value_batch, adv_batch, aux_batch = self.memory.sample(
                                    sample_idx)
                                step_info = self.learner.update(obs_batch, act_batch, ret_batch, value_batch, adv_batch,
                                                                aux_batch['old_logp'])
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

    .. code-block:: python

        from xuance.tensorflow.agents import *


        class PPOKL_Agent(Agent):
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

                self.gamma = config.gamma
                self.gae_lam = config.gae_lambda
                self.observation_space = envs.observation_space
                self.action_space = envs.action_space
                self.representation_info_shape = policy.representation_actor.output_shapes
                self.auxiliary_info_shape = {"old_dist": None}

                self.atari = True if config.env_name == "Atari" else False
                Buffer = DummyOnPolicyBuffer_Atari if self.atari else DummyOnPolicyBuffer_Atari
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
                learner = PPOKL_Learner(policy,
                                        optimizer,
                                        config.device,
                                        config.model_dir,
                                        config.vf_coef,
                                        config.ent_coef,
                                        config.target_kl)
                super(PPOKL_Agent, self).__init__(config, envs, policy, memory, learner, device, config.log_dir, config.model_dir)

            def _action(self, obs):
                _, _, vs = self.policy(obs)
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
                    acts, values, dists = self._action(obs)
                    next_obs, rewards, terminals, trunctions, infos = self.envs.step(acts)

                    self.memory.store(obs, acts, self._process_reward(rewards), values, terminals, {"old_dist": dists})
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
                                obs_batch, act_batch, ret_batch, value_batch, adv_batch, aux_batch = self.memory.sample(
                                    sample_idx)
                                step_info = self.learner.update(obs_batch, act_batch, ret_batch, value_batch, adv_batch,
                                                                aux_batch['old_logp'])
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


  .. group-tab:: MindSpore

    .. code-block:: python

        from xuance.mindspore.agents import *


        class PPOKL_Agent(Agent):
            def __init__(self,
                         config: Namespace,
                         envs: VecEnv,
                         policy: nn.Cell,
                         optimizer: nn.Optimizer,
                         scheduler: Optional[nn.exponential_decay_lr] = None):
                self.config = config
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
                self.representation_info_shape = policy.representation.output_shapes
                self.auxiliary_info_shape = {"old_logp": ()}

                writer = SummaryWriter(config.log_dir)
                memory = DummyOnPolicyBuffer(self.observation_space,
                                             self.action_space,
                                             self.representation_info_shape,
                                             self.auxiliary_info_shape,
                                             self.nenvs,
                                             self.nsteps,
                                             self.nminibatch,
                                             self.gamma,
                                             self.lam)
                learner = PPOCLIP_Learner(policy,
                                          optimizer,
                                          scheduler,
                                          writer,
                                          config.model_dir,
                                          config.vf_coef,
                                          config.ent_coef,
                                          0.2)

                self.obs_rms = RunningMeanStd(shape=space2shape(self.observation_space), comm=self.comm, use_mpi=False)
                self.ret_rms = RunningMeanStd(shape=(), comm=self.comm, use_mpi=False)
                super(PPOKL_Agent, self).__init__(envs, policy, memory, learner, writer, config.log_dir,
                                                  config.model_dir)

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
                states, act_probs, vs = self.policy(ms.Tensor(obs))
                acts = self.policy.actor.sample(act_probs)
                logps = self.policy.actor.log_prob(value=acts, probs=act_probs)
                if context._get_mode() == 0:
                    return {"state": states[0].asnumpy()}, acts.asnumpy(), vs.asnumpy(), logps.asnumpy()
                else:
                    for key in states.keys():
                        states[key] = states[key].asnumpy()
                    return states, acts.asnumpy(), vs.asnumpy(), logps.asnumpy()

            def train(self, train_steps=10000):
                episodes = np.zeros((self.nenvs,), np.int32)
                scores = np.zeros((self.nenvs,), np.float32)
                returns = np.zeros((self.nenvs,), np.float32)

                obs = self.envs.reset()
                for step in tqdm(range(train_steps)):
                    self.obs_rms.update(obs)
                    obs = self._process_observation(obs)
                    states, acts, rets, logps = self._action(obs)
                    next_obs, rewards, dones, infos = self.envs.step(acts)
                    if self.render: self.envs.render()
                    self.memory.store(obs, acts, self._process_reward(rewards), rets, dones, states, {"old_logp": logps})
                    if self.memory.full:
                        _, _, vals, _ = self._action(self._process_observation(next_obs))
                        for i in range(self.nenvs):
                            self.memory.finish_path(vals[i], i)
                        for _ in range(self.nminibatch * self.nepoch):
                            obs_batch, act_batch, ret_batch, adv_batch, _, aux_batch = self.memory.sample()
                            self.learner.update(obs_batch, act_batch, ret_batch, adv_batch, aux_batch['old_logp'])
                        self.memory.clear()
                    scores += rewards
                    returns = self.gamma * returns + rewards
                    obs = next_obs
                    for i in range(self.nenvs):
                        if dones[i] == True:
                            self.ret_rms.update(returns[i:i + 1])
                            self.memory.finish_path(0, i)
                            self.writer.add_scalars("returns-episode", {"env-%d" % i: scores[i]}, episodes[i])
                            self.writer.add_scalars("returns-step", {"env-%d" % i: scores[i]}, step)
                            scores[i] = 0
                            returns[i] = 0
                            episodes[i] += 1

                    if step % 50000 == 0 or step == train_steps - 1:
                        self.save_model()
                        np.save(self.model_dir + "/obs_rms.npy",
                                {'mean': self.obs_rms.mean, 'std': self.obs_rms.std, 'count': self.obs_rms.count})

            def test(self, test_episodes=100000, load_model=None):
                self.load_model(self.model_dir)
                scores = np.zeros((self.nenvs,), np.float32)
                returns = np.zeros((self.nenvs,), np.float32)

                obs = self.envs.reset()
                for _ in tqdm(range(test_episodes)):
                    self.obs_rms.update(obs)
                    obs = self._process_observation(obs)
                    states, acts, rets, logps = self._action(obs)
                    next_obs, rewards, dones, infos = self.envs.step(acts)
                    self.envs.render()

                    scores += rewards
                    returns = self.gamma * returns + rewards
                    obs = next_obs
                    for i in range(self.nenvs):
                        if dones[i] == True:
                            scores[i], returns[i] = 0, 0

            def evaluate(self):
                pass
