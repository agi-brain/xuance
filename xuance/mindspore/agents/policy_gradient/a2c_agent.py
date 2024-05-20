# This is the main file for an advantage actor critic (A2C) algorithm.
# The agent random sample a batch in the replay buffer, and optimize the policy gradient and value function loss.
# This can be a first RL algorithm code for the starters.
from xuance.mindspore.agents import *


class A2C_Agent(Agent):
    def __init__(self,
                 config: Namespace,
                 envs: DummyVecEnv,
                 policy: nn.Cell,
                 optimizer: nn.Optimizer,
                 scheduler: Optional[nn.exponential_decay_lr] = None):
        self.render = config.render
        self.n_envs = envs.num_envs
        self.n_steps = config.n_steps
        self.n_epoch = config.n_epoch
        self.n_minibatch = config.n_minibatch

        self.gamma = config.gamma
        self.gae_lam = config.gae_lambda
        self.clip_grad = config.clip_grad

        self.observation_space = envs.observation_space
        self.action_space = envs.action_space
        self.auxiliary_info_shape = {}
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
        learner = A2C_Learner(policy,
                              optimizer,
                              scheduler,
                              config.model_dir,
                              config.vf_coef,
                              config.ent_coef,
                              config.clip_grad,
                              config.clip_type)
        super(A2C_Agent, self).__init__(config, envs, policy, memory, learner, config.log_dir, config.model_dir)

    def _action(self, obs):
        _, act_probs, vs = self.policy(ms.Tensor(obs))
        acts = self.policy.actor.sample(act_probs).asnumpy()
        return acts, vs.asnumpy()

    def train(self, train_steps):
        obs = self.envs.buf_obs
        for _ in tqdm(range(train_steps)):
            step_info = {}
            self.obs_rms.update(obs)
            obs = self._process_observation(obs)
            acts, vals = self._action(obs)
            next_obs, rewards, terminals, trunctions, infos = self.envs.step(acts)
            self.memory.store(obs, acts, self._process_reward(rewards), vals, terminals)
            if self.memory.full:
                _, vals = self._action(self._process_observation(next_obs))
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
                        obs_batch, act_batch, ret_batch, _, adv_batch, _ = self.memory.sample(sample_idx)
                        step_info = self.learner.update(obs_batch, act_batch, ret_batch, adv_batch)
                self.log_infos(step_info, self.current_step)
                self.memory.clear()

            self.returns = self.gamma * self.returns + rewards
            obs = next_obs
            for i in range(self.n_envs):
                if terminals[i] or trunctions[i]:
                    self.ret_rms.update(self.returns[i:i + 1])
                    self.returns[i] = 0.0
                    if self.atari and (~trunctions[i]):
                        pass
                    else:
                        obs[i] = infos[i]["reset_obs"]
                        if terminals[i]:
                            self.memory.finish_path(0, i)
                        else:
                            _, vals = self._action(self._process_observation(next_obs))
                            self.memory.finish_path(vals[i], i)
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
            acts, rets = self._action(obs)
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
