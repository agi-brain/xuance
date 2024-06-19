import torch
import numpy as np
from tqdm import tqdm
from copy import deepcopy
from argparse import Namespace
from xuance.environment import DummyVecEnv
from xuance.torch.utils import NormalizeFunctions, ActivationFunctions
from xuance.torch.policies import REGISTRY_Policy
from xuance.torch.learners import PPG_Learner
from xuance.torch.agents import Agent
from xuance.common import DummyOnPolicyBuffer, DummyOnPolicyBuffer_Atari
from xuance.torch.utils import split_distributions


class PPG_Agent(Agent):
    """The implementation of PPG agent.

    Args:
        config: the Namespace variable that provides hyper-parameters and other settings.
        envs: the vectorized environments.
    """
    def __init__(self,
                 config: Namespace,
                 envs: DummyVecEnv):
        super(PPG_Agent, self).__init__(config, envs)
        self.horizon_size = config.horizon_size
        self.n_minibatch = config.n_minibatch
        self.n_epochs = config.n_epochs
        self.gae_lam = config.gae_lambda
        self.policy_nepoch = config.policy_nepoch
        self.value_nepoch = config.value_nepoch
        self.aux_nepoch = config.aux_nepoch

        # build policy, optimizer, lr_scheduler.
        self.policy = self._build_policy()
        optimizer = torch.optim.Adam(self.policy.parameters(), self.config.learning_rate, eps=1e-5)
        lr_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.0,
                                                         total_iters=self.config.running_steps)
        
        self.auxiliary_info_shape = {"old_dist": None}
        self.atari = True if config.env_name == "Atari" else False
        Buffer = DummyOnPolicyBuffer_Atari if self.atari else DummyOnPolicyBuffer
        self.buffer_size = self.n_envs * self.horizon_size
        self.batch_size = self.buffer_size // self.n_epochs
        input_buffer = dict(observation_space=self.observation_space,
                            action_space=self.action_space,
                            auxiliary_shape=self.auxiliary_info_shape,
                            n_envs=self.n_envs,
                            horizon_size=self.horizon_size,
                            use_gae=config.use_gae,
                            use_advnorm=config.use_advnorm,
                            gamma=self.gamma,
                            gae_lam=self.gae_lam)
        self.memory = Buffer(**input_buffer)
        self.learner = self._build_learner(self.config, envs.max_episode_steps, self.policy, optimizer, lr_scheduler)

    def _build_policy(self):
        normalize_fn = NormalizeFunctions[self.config.normalize] if hasattr(self.config, "normalize") else None
        initializer = torch.nn.init.orthogonal_
        activation = ActivationFunctions[self.config.activation]
        device = self.device

        # build representation.
        representation = self._build_representation(self.config.representation, self.config)

        # build policy.
        if self.config.policy == "Categorical_PPG":
            policy = REGISTRY_Policy["Categorical_PPG"](
                action_space=self.action_space, representation=representation,
                actor_hidden_size=self.config.actor_hidden_size, critic_hidden_size=self.config.critic_hidden_size,
                normalize=normalize_fn, initialize=initializer, activation=activation, device=device)
        elif self.config.policy == "Gaussian_PPG":
            policy = REGISTRY_Policy["Gaussian_PPG"](
                action_space=self.action_space, representation=representation,
                actor_hidden_size=self.config.actor_hidden_size, critic_hidden_size=self.config.critic_hidden_size,
                normalize=normalize_fn, initialize=initializer, activation=activation, device=device,
                activation_action=ActivationFunctions[self.config.activation_action])
        else:
            raise AttributeError(f"PPG currently does not support the policy named {self.config.policy}.")

        return policy

    def _build_learner(self, *args):
        return PPG_Learner(*args)

    def action(self, obs):
        _, dists, vs, _ = self.policy(obs)
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
            acts, rets, dists = self.action(obs)
            next_obs, rewards, terminals, trunctions, infos = self.envs.step(acts)

            self.memory.store(obs, acts, self._process_reward(rewards), rets, terminals, {"old_dist": dists})
            if self.memory.full:
                _, vals, _ = self.action(self._process_observation(next_obs))
                for i in range(self.n_envs):
                    self.memory.finish_path(vals[i], i)
                # policy update
                indexes = np.arange(self.buffer_size)
                for _ in range(self.policy_nepoch):
                    np.random.shuffle(indexes)
                    for start in range(0, self.buffer_size, self.batch_size):
                        end = start + self.batch_size
                        sample_idx = indexes[start:end]
                        samples = self.memory.sample(sample_idx)
                        step_info.update(self.learner.update_policy(**samples))
                # critic update
                for _ in range(self.value_nepoch):
                    np.random.shuffle(indexes)
                    for start in range(0, self.buffer_size, self.batch_size):
                        end = start + self.batch_size
                        sample_idx = indexes[start:end]
                        samples = self.memory.sample(sample_idx)
                        step_info.update(self.learner.update_critic(**samples))
                    
                # update old_prob
                buffer_obs = self.memory.observations
                buffer_act = self.memory.actions
                _, new_dist, _, _ = self.policy(buffer_obs)
                self.memory.auxiliary_infos['old_dist'] = split_distributions(new_dist)
                for _ in range(self.aux_nepoch):
                    np.random.shuffle(indexes)
                    for start in range(0, self.buffer_size, self.batch_size):
                        end = start + self.batch_size
                        sample_idx = indexes[start:end]
                        samples = self.memory.sample(sample_idx)
                        step_info.update(self.learner.update_auxiliary(**samples))
                self.log_infos(step_info, self.current_step)
                self.memory.clear()

            obs = deepcopy(next_obs)
            for i in range(self.n_envs):
                if terminals[i] or trunctions[i]:
                    obs[i] = infos[i]["reset_obs"]
                    self.envs.buf_obs[i] = obs[i]
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
            acts, rets, logps = self.action(obs)
            next_obs, rewards, terminals, trunctions, infos = test_envs.step(acts)
            if self.config.render_mode == "rgb_array" and self.render:
                images = test_envs.render(self.config.render_mode)
                for idx, img in enumerate(images):
                    videos[idx].append(img)

            obs = deepcopy(next_obs)
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
