from tqdm import tqdm
from copy import deepcopy
from argparse import Namespace
from xuance.common import Union, Optional, PerOffPolicyBuffer
from xuance.environment import DummyVecEnv, SubprocVecEnv
from xuance.torch.agents import BaseCallback
from xuance.torch.agents.qlearning_family import DQN_Agent


class PerDQN_Agent(DQN_Agent):
    """The implementation of Per-DQN agent.

    Args:
        config: the Namespace variable that provides hyperparameters and other settings.
        envs: the vectorized environments.
        callback: A user-defined callback function object to inject custom logic during training.
    """

    def __init__(self,
                 config: Namespace,
                 envs: Union[DummyVecEnv, SubprocVecEnv],
                 callback: Optional[BaseCallback] = None):
        super(PerDQN_Agent, self).__init__(config, envs, callback)
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
        self.learner = self._build_learner(self.config, self.policy, self.callback)

    def train_epochs(self, n_epochs=1):
        train_info = {}
        for _ in range(n_epochs):
            samples = self.memory.sample(self.PER_beta)
            td_error, step_info = self.learner.update(**samples)
            self.memory.update_priorities(samples['step_choices'], td_error)
        train_info["epsilon-greedy"] = self.e_greedy
        return train_info

    def train(self, train_steps):
        train_info = {}
        obs = self.envs.buf_obs
        for _ in tqdm(range(train_steps)):
            self.obs_rms.update(obs)
            obs = self._process_observation(obs)
            policy_out = self.action(obs, test_mode=False)
            acts = policy_out['actions']
            next_obs, rewards, terminals, truncations, infos = self.envs.step(acts)

            self.callback.on_train_step(self.current_step, envs=self.envs, policy=self.policy,
                                        obs=obs, policy_out=policy_out, acts=acts, next_obs=next_obs, rewards=rewards,
                                        terminals=terminals, truncations=truncations, infos=infos,
                                        train_steps=train_steps)

            self.memory.store(obs, acts, self._process_reward(rewards), terminals, self._process_observation(next_obs))
            if self.current_step > self.start_training and self.current_step % self.training_frequency == 0:
                update_info = self.train_epochs(n_epochs=self.n_epochs)
                self.log_infos(update_info, self.current_step)
                train_info.update(update_info)
                self.PER_beta += (1 - self.PER_beta0) / train_steps
                self.callback.on_train_epochs_end(self.current_step, policy=self.policy, memory=self.memory,
                                                  current_episode=self.current_episode, train_steps=train_steps,
                                                  update_info=update_info, per_beta=self.PER_beta)

            obs = deepcopy(next_obs)
            for i in range(self.n_envs):
                if terminals[i] or truncations[i]:
                    if self.atari and (~truncations[i]):
                        pass
                    else:
                        obs[i] = infos[i]["reset_obs"]
                        self.envs.buf_obs[i] = obs[i]
                        self.current_episode[i] += 1
                        if self.use_wandb:
                            episode_info = {
                                f"Episode-Steps/rank_{self.rank}/env-{i}": infos[i]["episode_step"],
                                f"Train-Episode-Rewards/rank_{self.rank}/env-{i}": infos[i]["episode_score"]
                            }
                        else:
                            episode_info = {
                                f"Episode-Steps/rank_{self.rank}": {f"env-{i}": infos[i]["episode_step"]},
                                f"Train-Episode-Rewards/rank_{self.rank}": {f"env-{i}": infos[i]["episode_score"]}
                            }
                        self.log_infos(episode_info, self.current_step)
                        train_info.update(episode_info)
                        self.callback.on_train_episode_info(envs=self.envs, policy=self.policy, env_id=i,
                                                            infos=infos, rank=self.rank, use_wandb=self.use_wandb,
                                                            current_step=self.current_step,
                                                            current_episode=self.current_episode,
                                                            train_steps=train_steps)

            self.current_step += self.n_envs
            if self.e_greedy > self.end_greedy:
                self.e_greedy -= self.delta_egreedy
            self.callback.on_train_step_end(self.current_step, envs=self.envs, policy=self.policy,
                                            train_steps=train_steps, train_info=train_info)
        return train_info
