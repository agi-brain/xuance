import numpy as np
from tqdm import tqdm
from copy import deepcopy
from argparse import Namespace
from xuance.common import Union, Optional, RecurrentOffPolicyBuffer, EpisodeBuffer, BaseCallback
from xuance.environment import DummyVecEnv, SubprocVecEnv
from xuance.mindspore import Module, Tensor
from xuance.mindspore.utils import NormalizeFunctions, ActivationFunctions, InitializeFunctions
from xuance.mindspore.policies import REGISTRY_Policy
from xuance.mindspore.agents import OffPolicyAgent


class DRQN_Agent(OffPolicyAgent):
    """The implementation of Deep Recurrent Q-Netowrk (DRQN) agent.

    Args:
        config: the Namespace variable that provides hyperparameters and other settings.
        envs: the vectorized environments.
        callback: A user-defined callback function object to inject custom logic during training.
    """

    def __init__(self,
                 config: Namespace,
                 envs: Union[DummyVecEnv, SubprocVecEnv],
                 callback: Optional[BaseCallback] = None):
        super(DRQN_Agent, self).__init__(config, envs, callback)

        self.start_greedy, self.end_greedy = config.start_greedy, config.end_greedy
        self.egreedy = config.start_greedy
        self.delta_egreedy = (self.start_greedy - self.end_greedy) / (config.decay_step_greedy / self.n_envs)

        self.policy = self._build_policy()  # build policy
        self.auxiliary_info_shape = {}
        self.memory = self._build_memory(auxiliary_info_shape=self.auxiliary_info_shape)  # build memory
        self.learner = self._build_learner(self.config, self.policy, self.callback)  # build learner
        self.lstm = True if config.rnn == "LSTM" else False

    def _build_memory(self, auxiliary_info_shape=None):
        self.atari = True if self.config.env_name == "Atari" else False
        Buffer = RecurrentOffPolicyBuffer(self.observation_space,
                                          self.action_space,
                                          auxiliary_info_shape,
                                          self.n_envs,
                                          self.config.buffer_size,
                                          self.config.batch_size,
                                          episode_length=self.episode_length,
                                          lookup_length=self.config.lookup_length)
        return Buffer

    def _build_policy(self) -> Module:
        normalize_fn = NormalizeFunctions[self.config.normalize] if hasattr(self.config, "normalize") else None
        initializer = InitializeFunctions[self.config.initialize] if hasattr(self.config, "initialize") else None
        activation = ActivationFunctions[self.config.activation]

        # build representation.
        representation = self._build_representation(self.config.representation, self.observation_space, self.config)

        # build policy.
        if self.config.policy == "DRQN_Policy":
            policy = REGISTRY_Policy["DRQN_Policy"](
                action_space=self.action_space, representation=representation,
                rnn=self.config.rnn, recurrent_hidden_size=self.config.recurrent_hidden_size,
                recurrent_layer_N=self.config.recurrent_layer_N, dropout=self.config.dropout,
                normalize=normalize_fn, initialize=initializer, activation=activation,
                use_distributed_training=self.distributed_training)
        else:
            raise AttributeError(
                f"{self.config.agent} currently does not support the policy named {self.config.policy}.")

        return policy

    def action(self, obs, egreedy=0.0, rnn_hidden=None):
        if self.lstm:
            rnn_hidden = (Tensor(rnn_hidden[0]), Tensor(rnn_hidden[1]))
        else:
            rnn_hidden = Tensor(rnn_hidden)
        _, argmax_action, _, rnn_hidden_next = self.policy(Tensor(obs[:, None]), *rnn_hidden)
        random_action = np.random.choice(self.action_space.n, self.n_envs)
        if np.random.rand() < egreedy:
            actions = random_action
        else:
            actions = argmax_action.numpy()
        if self.lstm:
            rnn_hidden_next_np = (rnn_hidden_next[0].numpy(), rnn_hidden_next[1].numpy())
        else:
            rnn_hidden_next_np = rnn_hidden_next.numpy()
        return {"actions": actions, "rnn_hidden_next": rnn_hidden_next_np}

    def train(self, train_steps):
        train_info = {}
        obs = self.envs.buf_obs
        episode_data = [EpisodeBuffer() for _ in range(self.n_envs)]
        for i_env in range(self.n_envs):
            episode_data[i_env].obs.append(self._process_observation(obs[i_env]))
        self.rnn_hidden = self.policy.init_hidden(self.n_envs)
        dones = [False for _ in range(self.n_envs)]
        for _ in tqdm(range(train_steps)):
            self.obs_rms.update(obs)
            obs = self._process_observation(obs)
            policy_out = self.action(obs, self.egreedy, self.rnn_hidden)
            acts, self.rnn_hidden = policy_out['actions'], policy_out['rnn_hidden_next']
            next_obs, rewards, terminals, truncations, infos = self.envs.step(acts)

            self.callback.on_train_step(self.current_step, envs=self.envs, policy=self.policy,
                                        obs=obs, policy_out=policy_out, acts=acts, next_obs=next_obs, rewards=rewards,
                                        terminals=terminals, truncations=truncations, infos=infos,
                                        train_steps=train_steps, rnn_hidden=self.rnn_hidden)

            if (self.current_step > self.start_training) and (self.current_step % self.training_frequency == 0):
                # training
                update_info = self.train_epochs(n_epochs=1)
                self.log_infos(update_info, self.current_step)
                train_info.update(update_info)

            obs = deepcopy(next_obs)
            for i in range(self.n_envs):
                episode_data[i].put(
                    [self._process_observation(obs[i]), acts[i], self._process_reward(rewards[i]), terminals[i]])
                if terminals[i] or truncations[i]:
                    if self.atari and (~truncations[i]):
                        pass
                    else:
                        self.rnn_hidden = self.policy.init_hidden_item(self.rnn_hidden, i)
                        dones[i] = True
                        self.current_episode[i] += 1
                        if self.use_wandb:
                            episode_info = {
                                f"Episode-Steps/env-{i}": infos[i]["episode_step"],
                                f"Train-Episode-Rewards/env-{i}": infos[i]["episode_score"]
                            }
                        else:
                            episode_info = {
                                f"Episode-Steps": {f"env-{i}": infos[i]["episode_step"]},
                                f"Train-Episode-Rewards": {f"env-{i}": infos[i]["episode_score"]}
                            }
                        self.log_infos(episode_info, self.current_step)
                        train_info.update(episode_info)
                        self.memory.store(episode_data[i])
                        episode_data[i] = EpisodeBuffer()
                        obs[i] = infos[i]["reset_obs"]
                        self.envs.buf_obs[i] = obs[i]
                        episode_data[i].obs.append(self._process_observation(obs[i]))

                        self.callback.on_train_episode_info(envs=self.envs, policy=self.policy, env_id=i,
                                                            memory=self.memory,
                                                            infos=infos, use_wandb=self.use_wandb,
                                                            current_step=self.current_step,
                                                            current_episode=self.current_episode,
                                                            train_steps=train_steps)

            self.current_step += self.n_envs
            if self.egreedy > self.end_greedy:
                self.egreedy = self.egreedy - self.delta_egreedy
            self.callback.on_train_step_end(self.current_step, envs=self.envs, policy=self.policy,
                                            train_steps=train_steps, train_info=train_info)
        return train_info

    def test(self, env_fn, test_episodes):
        test_envs = env_fn()
        num_envs = test_envs.num_envs
        videos, episode_videos, images = [[] for _ in range(num_envs)], [], None
        current_episode, current_step, scores, best_score = 0, 0, [], -np.inf
        obs, infos = test_envs.reset()
        if self.config.render_mode == "rgb_array" and self.render:
            images = test_envs.render(self.config.render_mode)
            for idx, img in enumerate(images):
                videos[idx].append(img)

        rnn_hidden = self.policy.init_hidden(num_envs)
        while current_episode < test_episodes:
            self.obs_rms.update(obs)
            obs = self._process_observation(obs)
            policy_out = self.action(obs, egreedy=0.0, rnn_hidden=rnn_hidden)
            acts, rnn_hidden = policy_out['actions'], policy_out['rnn_hidden_next']
            next_obs, rewards, terminals, truncations, infos = test_envs.step(acts)
            if self.config.render_mode == "rgb_array" and self.render:
                images = test_envs.render(self.config.render_mode)
                for idx, img in enumerate(images):
                    videos[idx].append(img)

            self.callback.on_test_step(envs=test_envs, policy=self.policy, images=images, rnn_hidden=rnn_hidden,
                                       obs=obs, policy_out=policy_out, next_obs=next_obs, rewards=rewards,
                                       terminals=terminals, truncations=truncations, infos=infos,
                                       current_train_step=self.current_step,
                                       current_step=current_step, current_episode=current_episode)

            obs = deepcopy(next_obs)
            for i in range(num_envs):
                if terminals[i] or truncations[i]:
                    if self.atari and (~truncations[i]):
                        pass
                    else:
                        obs[i] = infos[i]["reset_obs"]
                        rnn_hidden = self.policy.init_hidden_item(rnn_hidden, i)
                        scores.append(infos[i]["episode_score"])
                        current_episode += 1
                        if best_score < infos[i]["episode_score"]:
                            best_score = infos[i]["episode_score"]
                            episode_videos = videos[i].copy()
                        if self.config.test_mode:
                            print("Episode: %d, Score: %.2f" % (current_episode, infos[i]["episode_score"]))
            current_step += num_envs

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

        self.callback.on_test_end(envs=test_envs, policy=self.policy,
                                  current_train_step=self.current_step,
                                  current_step=current_step, current_episode=current_episode,
                                  scores=scores, best_score=best_score)

        test_envs.close()

        return scores
