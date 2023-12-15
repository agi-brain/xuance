from xuance.tensorflow.agents import *
import gym
from gym import spaces


class PDQN_Agent(Agent):
    def __init__(self,
                 config: Namespace,
                 envs: Gym_Env,
                 policy: tk.Model,
                 optimizer: Sequence[tk.optimizers.Optimizer],
                 device: str = 'cpu'):
        self.envs = envs
        self.render = config.render
        self.n_envs = envs.num_envs

        self.gamma = config.gamma
        self.train_frequency = config.training_frequency
        self.start_training = config.start_training
        self.start_greedy = config.start_greedy
        self.end_greedy = config.end_greedy
        self.egreedy = config.start_greedy

        self.train_frequency = config.training_frequency
        self.start_training = config.start_training
        self.start_noise = config.start_noise
        self.end_noise = config.end_noise
        self.noise_scale = config.start_noise

        self.observation_space = envs.observation_space.spaces[0]
        old_as = envs.action_space
        num_disact = old_as.spaces[0].n
        self.action_space = gym.spaces.Tuple((old_as.spaces[0], *(gym.spaces.Box(old_as.spaces[1].spaces[i].low,
                                                                                 old_as.spaces[1].spaces[i].high,
                                                                                 dtype=np.float32) for i in
                                                                  range(0, num_disact))))
        self.action_high = [self.action_space.spaces[i].high for i in range(1, num_disact + 1)]
        self.action_low = [self.action_space.spaces[i].low for i in range(1, num_disact + 1)]
        self.action_range = [self.action_space.spaces[i].high - self.action_space.spaces[i].low for i in
                             range(1, num_disact + 1)]
        self.representation_info_shape = {'state': (envs.observation_space.spaces[0].shape)}
        self.auxiliary_info_shape = {}
        self.nenvs = 1
        self.epsilon = 1.0
        self.epsilon_steps = 1000
        self.epsilon_initial = 1.0
        self.epsilon_final = 0.1
        self.buffer_action_space = spaces.Box(np.zeros(4), np.ones(4), dtype=np.float64)

        memory = DummyOffPolicyBuffer(self.observation_space,
                                      self.buffer_action_space,
                                      self.auxiliary_info_shape,
                                      self.n_envs,
                                      config.n_size,
                                      config.batch_size)
        learner = PDQN_Learner(policy,
                               optimizer,
                               config.device,
                               config.model_dir,
                               config.gamma,
                               config.tau)

        self.num_disact = self.action_space.spaces[0].n
        self.conact_sizes = np.array([self.action_space.spaces[i].shape[0] for i in range(1, self.num_disact+1)])
        self.conact_size = int(self.conact_sizes.sum())

        super(PDQN_Agent, self).__init__(config, envs, policy, memory, learner, device,
                                         config.log_dir, config.model_dir)

    def _action(self, obs):
        with tf.device(self.device):
            obs = tf.convert_to_tensor(obs, tf.float32)
            obs = tf.expand_dims(obs, axis=0)
            con_actions = self.policy.con_action(obs)
            con_actions = tf.stop_gradient(con_actions)
            rnd = np.random.rand()
            if rnd < self.epsilon:
                disaction = np.random.choice(self.num_disact)
            else:
                q = self.policy.Qeval(obs, con_actions)
                q = tf.stop_gradient(q)
                q = q.numpy()
                disaction = np.argmax(q)

        con_actions = con_actions.numpy()
        con_actions = np.squeeze(con_actions, axis=0)
        offset = np.array([self.conact_sizes[i] for i in range(disaction)], dtype=int).sum()
        conaction = con_actions[offset:offset+self.conact_sizes[disaction]]

        return disaction, conaction, con_actions

    def pad_action(self, disaction, conaction):
        con_actions = [np.zeros((1,), dtype=np.float32), np.zeros((1,), dtype=np.float32), np.zeros((1,), dtype=np.float32)]
        con_actions[disaction][:] = conaction
        return (disaction, con_actions)

    def train(self, train_steps=10000):
        episodes = np.zeros((self.nenvs,), np.int32)
        scores = np.zeros((self.nenvs,), np.float32)
        obs, _ = self.envs.reset()
        for _ in tqdm(range(train_steps)):
            step_info = {}
            disaction, conaction, con_actions = self._action(obs)
            action = self.pad_action(disaction, conaction)
            action[1][disaction] = self.action_range[disaction] * (action[1][disaction] + 1) / 2. + self.action_low[disaction]
            (next_obs, steps), rewards, terminal, _ = self.envs.step(action)
            if self.render: self.envs.render("human")
            acts = np.concatenate(([disaction], con_actions), axis=0).ravel()
            self.memory.store(obs, acts, rewards, terminal, next_obs)
            if self.current_step > self.start_training and self.current_step % self.train_frequency == 0:
                obs_batch, act_batch, rew_batch, terminal_batch, next_batch = self.memory.sample()
                step_info.update(self.learner.update(obs_batch, act_batch, rew_batch, next_batch, terminal_batch))

            scores += rewards
            obs = next_obs
            self.noise_scale = self.start_noise - (self.start_noise - self.end_noise) / train_steps
            if terminal == True:
                step_info["returns-step"] = scores
                scores = 0
                returns = 0
                episodes += 1
                self.end_episode(episodes)
                obs, _ = self.envs.reset()
                self.log_infos(step_info, self.current_step)

            self.current_step += self.n_envs
            if self.egreedy >= self.end_greedy:
                self.egreedy = self.egreedy - (self.start_greedy - self.end_greedy) / self.config.decay_step_greedy

    def test(self, env_fn, test_episodes):
        test_envs = env_fn()
        episode_score = 0
        current_episode, scores, best_score = 0, [], -np.inf
        obs, _ = self.envs.reset()

        while current_episode < test_episodes:
            disaction, conaction, con_actions = self._action(obs)
            action = self.pad_action(disaction, conaction)
            action[1][disaction] = self.action_range[disaction] * (action[1][disaction] + 1) / 2. + self.action_low[disaction]
            (next_obs, steps), rewards, terminal, _ = self.envs.step(action)
            self.envs.render("human")
            episode_score += rewards
            obs = next_obs
            if terminal == True:
                scores.append(episode_score)
                obs, _ = self.envs.reset()
                current_episode += 1
                if best_score < episode_score:
                    best_score = episode_score
                episode_score = 0
                if self.config.test_mode:
                    print("Episode: %d, Score: %.2f" % (current_episode, episode_score))

        if self.config.test_mode:
            print("Best Score: %.2f" % (best_score))

        test_info = {
            "Test-Episode-Rewards/Mean-Score": np.mean(scores),
            "Test-Episode-Rewards/Std-Score": np.std(scores)
        }
        self.log_infos(test_info, self.current_step)

        test_envs.close()

        return scores

    def end_episode(self, episode):
        if episode < self.epsilon_steps:
            self.epsilon = self.epsilon_initial - (self.epsilon_initial - self.epsilon_final) * (
                    episode / self.epsilon_steps)
        else:
            self.epsilon = self.epsilon_final
