import torch
from copy import deepcopy
from xuance.common import List, Union, SequentialReplayBuffer
from xuance.environment import DummyVecEnv, SubprocVecEnv
from xuance.torch.agents import OffPolicyAgent, BaseCallback
from xuance.torch import REGISTRY_Representation, REGISTRY_Policy
from xuance.torch.utils import ActivationFunctions

# '.': import from __init__
from xuance.torch.representations.world_model import DreamerV3WorldModel, PlayerDV3
from xuance.torch.policies import DreamerV3Policy

import numpy as np
from tqdm import tqdm
import gymnasium as gym
from argparse import Namespace
from xuance.common import Optional


class DreamerV3Agent(OffPolicyAgent):
    def __init__(self,
                 config: Namespace,
                 envs: Union[DummyVecEnv, SubprocVecEnv],
                 callback: Optional[BaseCallback] = None):
        super(DreamerV3Agent, self).__init__(config, envs, callback)
        # special judge for atari env
        self.atari = True if self.config.env_name == "Atari" else False

        # continuous or not
        self.is_continuous = (isinstance(self.envs.action_space, gym.spaces.Box))
        self.is_multidiscrete = isinstance(self.envs.action_space, gym.spaces.MultiDiscrete)
        self.config.is_continuous = self.is_continuous  # add to config

        # obs_shape & act_shape
        self.obs_shape = self.observation_space.shape
        """
        hwc 2 chw: 
        agent & memory both uses 'hwc'
        obs needed to be transformed to 'chw' and be normalized before sample & taking an action
        """
        if self.config.pixel:
            self.obs_shape = (self.obs_shape[2], ) + self.obs_shape[:2]
        self.act_shape = self.action_space.n if not self.is_continuous else self.action_space.shape
        self.config.act_shape = self.act_shape  # add to config

        # ratio
        self.replay_ratio = self.config.replay_ratio
        self.current_step, self.gradient_step = 0, 0

        # REGISTRY & create: representation, policy, learner
        ActivationFunctions['silu'] = torch.nn.SiLU
        REGISTRY_Representation['DreamerV3WorldModel'] = DreamerV3WorldModel
        self.model = self._build_representation(representation_key="DreamerV3WorldModel",
                                                config=None, input_space=None)

        REGISTRY_Policy["DreamerV3Policy"] = DreamerV3Policy
        self.policy = self._build_policy()
        self.memory = self._build_memory()
        self.learner = self._build_learner(self.config, self.policy, self.act_shape, self.callback)

        # train_player & train_states; make sure train & test to be independent
        self.train_player: PlayerDV3 = self.model.player
        self.train_player.init_states()
        self.train_states: List[np.ndarray] = [
            self.envs.buf_obs,  # obs: (envs, *obs_shape),
            np.zeros((self.envs.num_envs, )),  # rews
            np.zeros((self.envs.num_envs, )),  # terms
            np.zeros((self.envs.num_envs, )),  # truncs
            np.ones((self.envs.num_envs, ))  # is_first
        ]

    def _build_representation(self, representation_key: str,
                              input_space: Optional[gym.spaces.Space],
                              config: Optional[Namespace]) -> DreamerV3WorldModel:
        # specify the type in order to use code completion
        actions_dim = tuple(
            self.envs.action_space.shape
            if self.is_continuous else (
                self.envs.action_space.nvec.tolist() if self.is_multidiscrete else [self.envs.action_space.n]
            )
        )
        input_representations = dict(
            actions_dim=actions_dim,
            is_continuous=self.is_continuous,
            config=self.config,
            obs_space=self.envs.observation_space
        )
        representation = REGISTRY_Representation[representation_key](**input_representations)
        if representation_key not in REGISTRY_Representation:
            raise AttributeError(f"{representation_key} is not registered in REGISTRY_Representation.")
        return representation

    def _build_memory(self, auxiliary_info_shape=None) -> SequentialReplayBuffer:
        input_buffer = dict(observation_space=self.observation_space,
                            action_space=self.action_space,
                            auxiliary_shape=auxiliary_info_shape,
                            n_envs=self.n_envs,
                            buffer_size=self.buffer_size,
                            batch_size=self.batch_size)
        return SequentialReplayBuffer(**input_buffer)

    def _build_policy(self) -> DreamerV3Policy:
        return REGISTRY_Policy["DreamerV3Policy"](self.model, self.config)

    def action(self,
               obs: np.ndarray,
               test_mode: Optional[bool] = False,
               player: Optional[PlayerDV3] = None) -> np.ndarray:
        """Returns actions and values.

        Parameters:
            obs (np.ndarray): The observation.
            test_mode (Optional[bool]): True for testing without noises.
            player (Optional[PlayerDV3]): The player whose action is taken, default is train_player.

        Returns:
            actions: The real_actions to be executed.
        """
        if self.config.pixel:
            obs = obs.transpose(0, 3, 1, 2) / 255.0 - 0.5
        player = player if player is not None else self.train_player
        # actions_output = self.policy(observations)
        # [envs, *obs_shape] -> [1: batch, envs, *obs_shape]
        obs = torch.as_tensor(obs, device=self.device, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            actions = player.get_actions(obs, greedy=test_mode, mask=None)[0][0]
        # ont-hot -> real_actions
        if not self.is_continuous:
            actions = actions.argmax(dim=1).detach().cpu().numpy()
        else:  # [1, envs, *act_shape]
            actions = actions.reshape(obs.shape[1], *self.act_shape).detach().cpu().numpy()
            # action mapping in xuance.environment.utils.wrapper.XuanCeEnvWrapper.step
            # actions = (actions + 1.0) * 0.5 * (self.actions_high - self.actions_low) + self.actions_low  # action_scaling
        """
        for env_interaction: actions.shape, (envs, ) or (env, *act_shape)
        """
        return actions

    def train_epochs(self, n_epochs: int = 1):
        train_info = {}
        samples = self.memory.sample(self.config.seq_len)  # (envs, seq, batch, ~)
        if self.config.pixel:
            samples['obs'] = samples['obs'].transpose(0, 1, 2, 5, 3, 4) / 255.0 - 0.5
        # n_epoch(n_gradient step) scattered to each environment
        # st = np.random.choice(np.arange(self.envs.num_envs), 1).item()  # not necessary
        st = 0
        for _ in range(n_epochs):  # assert n_epochs == parallels
            cur_samples = {k: v[(st + _) % self.envs.num_envs] for k, v in samples.items()}
            train_info = self.learner.update(**cur_samples)
        return train_info

    def train(self, train_steps):  # each train still uses old rssm_states until episode end
        train_info = {}
        obs, rews, terms, truncs, is_first = self.train_states

        for _ in tqdm(range(train_steps)):
            self.obs_rms.update(obs)
            obs = self._process_observation(obs)
            if self.current_step < self.start_training:  # ramdom_sample before training
                acts = np.array([self.envs.action_space.sample() for _ in range(self.envs.num_envs)])
            else:
                acts = self.action(obs)
            if self.atari:  # use truncs to train in xc_atari
                terms = deepcopy(truncs)
            """(o1, a1, r1, term1, trunc1, is_first1), acts: real_acts"""
            self.memory.store(obs, acts, self._process_reward(rews), terms, truncs, is_first)
            next_obs, rews, terms, truncs, infos = self.envs.step(acts)

            self.callback.on_train_step(self.current_step, envs=self.envs, policy=self.policy,
                                        obs=obs, acts=acts, next_obs=next_obs, rewards=rews,
                                        terminals=terms, truncations=truncs, infos=infos,
                                        train_steps=train_steps)

            """
            set to zeros after the first step
            (o2, a1, r2, term2, trunc2, is_first2)
            """
            is_first = np.zeros_like(terms)
            obs = next_obs
            self.returns = self.gamma * self.returns + rews
            done_idxes = []
            for i in range(self.n_envs):
                if terms[i] or truncs[i]:
                    if self.atari and (~truncs[i]):  # do not term until trunc
                        pass
                    else:
                        # carry the reset procedure to the outside
                        done_idxes.append(i)
                        self.ret_rms.update(self.returns[i:i + 1])
                        self.returns[i] = 0.0
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
            if len(done_idxes) > 0:
                """
                store the last data and reset all
                (o_t, a_t = 0 for dones, r_t, term_t, trunc_t, is_first_t)
                """
                extra_shape = () if not self.is_continuous else self.act_shape
                acts[done_idxes] = np.zeros((len(done_idxes),) + extra_shape)
                if self.atari:  # use truncs to train in xc_atari
                    terms = deepcopy(truncs)
                self.memory.store(obs, acts, self._process_reward(rews), terms, truncs, is_first)

                """reset DreamerV3 Player's states"""
                obs[done_idxes] = np.stack([infos[idx]["reset_obs"] for idx in done_idxes])  # reset obs
                self.envs.buf_obs[done_idxes] = obs[done_idxes]
                rews[done_idxes] = np.zeros((len(done_idxes), ))
                terms[done_idxes] = np.zeros((len(done_idxes), ))
                truncs[done_idxes] = np.zeros((len(done_idxes), ))
                is_first[done_idxes] = np.ones_like(terms[done_idxes])
                self.train_player.init_states(done_idxes)
            """
            start training 
            replay_ratio = self.gradient_step / self.current_step
            """
            if self.current_step > self.start_training:
                # count current_step when start_training
                n_epochs = max(int((self.current_step - self.start_training) * self.replay_ratio - self.gradient_step), 0)
                update_info = self.train_epochs(n_epochs=n_epochs)
                self.gradient_step += n_epochs
                if train_info is not None:
                    self.log_infos(update_info, self.current_step)
                    train_info.update(update_info)
                    self.callback.on_train_epochs_end(self.current_step, policy=self.policy, memory=self.memory,
                                                      current_episode=self.current_episode, train_steps=train_steps,
                                                      update_info=update_info)

            self.callback.on_train_step_end(self.current_step, envs=self.envs, policy=self.policy,
                                            train_steps=train_steps, train_info=train_info)
        # save the train_states for next train
        self.train_states = [obs, rews, terms, truncs, is_first]
        return train_info

    def test(self, env_fn, test_episodes: int) -> list:
        test_envs = env_fn()
        num_envs = test_envs.num_envs
        # copy the total network for test
        test_player = deepcopy(self.train_player)
        test_player.init_states(num_envs=num_envs)
        videos, episode_videos = [[] for _ in range(num_envs)], []
        current_episode, scores, best_score = 0, [], -np.inf
        obs, infos = test_envs.reset()
        if self.config.render_mode == "rgb_array" and self.render:
            images = test_envs.render(self.config.render_mode)
            for idx, img in enumerate(images):
                videos[idx].append(img)
        is_done = np.zeros(num_envs)
        while is_done.sum() < test_episodes:
            self.obs_rms.update(obs)
            obs = self._process_observation(obs)
            acts = self.action(obs, test_mode=True, player=test_player)
            next_obs, rews, terms, truncs, infos = test_envs.step(acts)
            if self.config.render_mode == "rgb_array" and self.render:
                images = test_envs.render(self.config.render_mode)
                for idx, img in enumerate(images):
                    videos[idx].append(img)

            obs = deepcopy(next_obs)
            done_idxes = []
            for i in range(num_envs):
                if terms[i] or truncs[i]:
                    if self.atari and (~truncs[i]):
                        pass
                    else:
                        done_idxes.append(i)
                        obs[i] = infos[i]["reset_obs"]
                        if is_done[i] != 1:
                            is_done[i] = 1
                            scores.append(infos[i]["episode_score"])
                        if best_score < infos[i]["episode_score"]:
                            best_score = infos[i]["episode_score"]
                            episode_videos = videos[i].copy()
                        if self.config.test_mode:
                            print("Episode: %d, Score: %.2f" % (current_episode, infos[i]["episode_score"]))
            if len(done_idxes) > 0:
                test_player.init_states(reset_envs=done_idxes, num_envs=num_envs)

        if self.config.render_mode == "rgb_array" and self.render:
            # time, height, width, channel -> time, channel, height, width
            videos_info = {"Videos_Test": np.array([episode_videos], dtype=np.uint8).transpose((0, 1, 4, 2, 3))}
            self.log_videos(info=videos_info, fps=self.fps, x_index=self.current_step)  # fps cannot work

        if self.config.test_mode:
            print("Best Score: %.2f" % best_score)

        test_info = {
            "Test-Episode-Rewards/Mean-Score": np.mean(scores),
            "Test-Episode-Rewards/Std-Score": np.std(scores)
        }
        self.log_infos(test_info, self.current_step)

        test_envs.close()


        return scores

