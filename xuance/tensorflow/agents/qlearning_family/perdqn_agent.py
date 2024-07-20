from tqdm import tqdm
from copy import deepcopy
from argparse import Namespace
from xuance.environment import DummyVecEnv
from xuance.tensorflow.agents.qlearning_family import DQN_Agent
from xuance.common import PerOffPolicyBuffer


class PerDQN_Agent(DQN_Agent):
    """The implementation of Per-DQN agent.

    Args:
        config: the Namespace variable that provides hyper-parameters and other settings.
        envs: the vectorized environments.
    """

    def __init__(self,
                 config: Namespace,
                 envs: DummyVecEnv):
        super(PerDQN_Agent, self).__init__(config, envs)
        self.PER_beta0 = config.PER_beta0
        self.PER_beta = config.PER_beta0

        # Create experience replay buffer.
        self.auxiliary_info_shape = {}
        self.atari = True if config.env_name == "Atari" else False
        self.memory = PerOffPolicyBuffer(observation_space=self.observation_space,
                                         action_space=self.action_space,
                                         auxiliary_shape=self.auxiliary_info_shape,
                                         n_envs=self.n_envs,
                                         buffer_size=config.buffer_size,
                                         batch_size=config.batch_size,
                                         alpha=config.PER_alpha)
        self.learner = self._build_learner(self.config, self.policy)

    def train_epochs(self, n_epochs=1):
        train_info = {}
        for _ in range(n_epochs):
            samples = self.memory.sample(self.PER_beta)
            td_error, step_info = self.learner.update(**samples)
            self.memory.update_priorities(samples['step_choices'], td_error)
        train_info["epsilon-greedy"] = self.e_greedy
        return train_info

    def train(self, train_steps):
        obs = self.envs.buf_obs
        for _ in tqdm(range(train_steps)):
            step_info = {}
            self.obs_rms.update(obs)
            obs = self._process_observation(obs)
            policy_out = self.action(obs, test_mode=False)
            acts = policy_out['actions']
            next_obs, rewards, terminals, trunctions, infos = self.envs.step(acts)

            self.memory.store(obs, acts, self._process_reward(rewards), terminals, self._process_observation(next_obs))
            if self.current_step > self.start_training and self.current_step % self.training_frequency == 0:
                train_info = self.train_epochs(n_epochs=self.n_epochs)
                self.log_infos(train_info, self.current_step)
                self.PER_beta += (1 - self.PER_beta0) / train_steps

            obs = deepcopy(next_obs)
            for i in range(self.n_envs):
                if terminals[i] or trunctions[i]:
                    if self.atari and (~trunctions[i]):
                        pass
                    else:
                        obs[i] = infos[i]["reset_obs"]
                        self.envs.buf_obs[i] = obs[i]
                        self.current_episode[i] += 1
                        if self.use_wandb:
                            step_info["Episode-Steps/env-%d" % i] = infos[i]["episode_step"]
                            step_info["Train-Episode-Rewards/env-%d" % i] = infos[i]["episode_score"]
                        else:
                            step_info["Episode-Steps"] = {"env-%d" % i: infos[i]["episode_step"]}
                            step_info["Train-Episode-Rewards"] = {"env-%d" % i: infos[i]["episode_score"]}
                        self.log_infos(step_info, self.current_step)

            self.current_step += self.n_envs
            if self.e_greedy > self.end_greedy:
                self.e_greedy -= self.delta_egreedy
