import torch
import numpy as np
from copy import deepcopy
from argparse import Namespace
from xuance.common import Union, Optional
from xuance.environment import DummyVecEnv, SubprocVecEnv
from xuance.torch import Module, REGISTRY_Learners
from xuance.torch.agents import OfflineAgent, BaseCallback
from xuance.torch.policies import REGISTRY_Policy
from xuance.torch.learners import TD3_BC_Learner
from xuance.torch.utils import NormalizeFunctions, ActivationFunctions
try:
    import d4rl
except:
    pass


class TD3_BC_Agent(OfflineAgent):
    """The implementation of TD3_BC agent.

    Args:
        config: the Namespace variable that provides hyperparameters and other settings.
        envs: the vectorized environments.
        callback: A user-defined callback function object to inject custom logic during training.
    """
    def __init__(self,
                 config: Namespace,
                 envs: Union[DummyVecEnv, SubprocVecEnv],
                 callback: Optional[BaseCallback] = None):
        super(TD3_BC_Agent, self).__init__(config, envs, callback)
        self.policy = self._build_policy()
        REGISTRY_Learners["TD3_BC_Learner"] = TD3_BC_Learner
        self.learner = self._build_learner(self.config, self.policy, self.callback)  # build learner
        self.dataset = None

    def load_dataset(self, dataset):
        self.dataset = dataset
        self.memory.d4rl2buffer(dataset=self.dataset)

    def _build_policy(self) -> Module:
        normalize_fn = NormalizeFunctions[self.config.normalize] if hasattr(self.config, "normalize") else None
        initializer = torch.nn.init.orthogonal_
        activation = ActivationFunctions[self.config.activation]
        device = self.device

        # build representations.
        representation = self._build_representation(self.config.representation, self.observation_space, self.config)

        # build policy
        if self.config.policy == "TD3_Policy":
            policy = REGISTRY_Policy["TD3_Policy"](
                action_space=self.action_space, representation=representation,
                actor_hidden_size=self.config.actor_hidden_size, critic_hidden_size=self.config.critic_hidden_size,
                normalize=normalize_fn, initialize=initializer, device=device,
                use_distributed_training=self.distributed_training,
                activation=activation, activation_action=ActivationFunctions[self.config.activation_action])
        else:
            raise AttributeError(f"TD3_BC currently does not support the policy named {self.config.policy}.")

        return policy

    def action(self, observations: np.ndarray):
        with torch.no_grad():
            _, actions = self.policy(observations)
            actions = actions.cpu().numpy()
        return {"actions": actions}

    def test(self, env_fn, test_episodes):
        test_envs = env_fn()
        num_envs = test_envs.num_envs
        videos, episode_videos, images = [[] for _ in range(num_envs)], [], None
        current_episode, current_step, scores, best_score = 0, 0, [], -np.inf
        for env_test in test_envs.envs:
            env_test.env.env.seed(self.config.env_seed)
        obs, infos = test_envs.reset()

        if self.config.render_mode == "rgb_array" and self.render:
            images = test_envs.render(self.config.render_mode)
            for idx, img in enumerate(images):
                videos[idx].append(img)

        while current_episode < test_episodes:
            self.obs_rms.update(obs)
            actions = self.action(obs)
            next_obs, rewards, terminated, truncated, infos = test_envs.step(actions['actions'])
            if self.config.render_mode == "rgb_array" and self.render:
                images = test_envs.render(self.config.render_mode)
                for idx, img in enumerate(images):
                    videos[idx].append(img)

            self.callback.on_test_step(envs=test_envs, policy=self.policy, images=images,
                                       obs=obs, actions=actions, next_obs=next_obs, rewards=rewards,
                                       terminals=terminated, truncations=truncated, infos=infos,
                                       current_train_step=self.current_step,
                                       current_step=current_step, current_episode=current_episode)

            obs = deepcopy(next_obs)
            for i in range(self.n_envs):
                if terminated[i] or truncated[i]:
                    obs[i] = infos[i]["reset_obs"]
                    scores.append(infos[i]["episode_score"])
                    current_episode += 1

                    if best_score < infos[i]["episode_score"]:
                        best_score = infos[i]["episode_score"]
                        episode_videos = videos[i].copy()

                    if self.config.test_mode:
                        print("Episode: %d, Score: %.2f" % (current_episode, infos[i]["episode_score"]))
            current_step += num_envs

        if self.config.render_mode == "rgb_array" and self.render:
            videos_info = {"Videos_Test": np.array([episode_videos], dtype=np.uint8).transpose((0, 1, 4, 2, 3))}
            self.log_videos(info=videos_info, fps=self.fps, x_index=self.current_step)

        if self.config.test_mode:
            print("Best Score: %.2f" % best_score)
        scores = np.array(scores)
        print(f"Test-Episode-Rewards:{scores}")
        print(f"Mean-Test-Episode-Rewards: %.3f" % np.mean(scores))
        normalized_returns = d4rl.get_normalized_score(self.config.dataset, scores) * 100.0

        test_info = {
            "Mean-Test-Episode-Rewards": np.mean(scores),
            "Std-Rewards": np.std(scores),
            "D4RL-Score": np.mean(normalized_returns),
            "Normalized_Returns_Std": np.std(normalized_returns)
        }
        self.log_infos(test_info, self.current_step)

        self.callback.on_test_end(envs=test_envs, policy=self.policy,
                                  current_train_step=self.current_step,
                                  current_step=current_step, current_episode=current_episode,
                                  scores=scores, best_score=best_score)

        test_envs.close()
        return normalized_returns
