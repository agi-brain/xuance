import torch
from tqdm import tqdm
from copy import deepcopy
from argparse import Namespace
from xuance.common import Union, Optional
from xuance.environment import DummyVecEnv, SubprocVecEnv
from xuance.torch import Module
from xuance.torch.utils import NormalizeFunctions, ActivationFunctions
from xuance.torch.policies import REGISTRY_Policy
from xuance.torch.agents import OnPolicyAgent, BaseCallback


class PPOCLIP_Agent(OnPolicyAgent):
    """The implementation of PPO agent.

    Args:
        config: the Namespace variable that provides hyperparameters and other settings.
        envs: the vectorized environments.
        callback: A user-defined callback function object to inject custom logic during training.
    """

    def __init__(self,
                 config: Namespace,
                 envs: Union[DummyVecEnv, SubprocVecEnv],
                 callback: Optional[BaseCallback] = None):
        super(PPOCLIP_Agent, self).__init__(config, envs, callback)
        self.auxiliary_info_shape = {"old_logp": ()}
        self.memory = self._build_memory(self.auxiliary_info_shape)  # build memory
        self.policy = self._build_policy()  # build policy
        self.learner = self._build_learner(self.config, self.policy, self.callback)  # build learner

    def _build_policy(self) -> Module:
        normalize_fn = NormalizeFunctions[self.config.normalize] if hasattr(self.config, "normalize") else None
        initializer = torch.nn.init.orthogonal_
        activation = ActivationFunctions[self.config.activation]
        device = self.device

        # build representation.
        representation = self._build_representation(self.config.representation, self.observation_space, self.config)

        # build policy.
        if self.config.policy == "Categorical_AC":
            policy = REGISTRY_Policy["Categorical_AC"](
                action_space=self.action_space, representation=representation,
                actor_hidden_size=self.config.actor_hidden_size, critic_hidden_size=self.config.critic_hidden_size,
                normalize=normalize_fn, initialize=initializer, activation=activation, device=device,
                use_distributed_training=self.distributed_training)
        elif self.config.policy == "Gaussian_AC":
            policy = REGISTRY_Policy["Gaussian_AC"](
                action_space=self.action_space, representation=representation,
                actor_hidden_size=self.config.actor_hidden_size, critic_hidden_size=self.config.critic_hidden_size,
                normalize=normalize_fn, initialize=initializer, activation=activation, device=device,
                use_distributed_training=self.distributed_training,
                activation_action=ActivationFunctions[self.config.activation_action])
        else:
            raise AttributeError(f"PPO_CLIP currently does not support the policy named {self.config.policy}.")

        return policy

    def get_aux_info(self, policy_output: dict = None):
        """Returns auxiliary information.

        Parameters:
            policy_output (dict): The output information of the policy.

        Returns:
            aux_info (dict): The auxiliary information.
        """
        aux_info = {"old_logp": policy_output['log_pi']}
        return aux_info

    def train(self, train_steps):
        train_info = {}
        obs = self.envs.buf_obs
        for _ in tqdm(range(train_steps)):
            self.obs_rms.update(obs)
            obs = self._process_observation(obs)
            policy_out = self.action(obs, return_dists=False, return_logpi=True)
            acts, value, logps = policy_out['actions'], policy_out['values'], policy_out['log_pi']
            next_obs, rewards, terminals, truncations, infos = self.envs.step(acts)
            aux_info = self.get_aux_info(policy_out)

            self.callback.on_train_step(self.current_step, envs=self.envs, policy=self.policy,
                                        obs=obs, policy_out=policy_out, acts=acts, vals=value, next_obs=next_obs,
                                        rewards=rewards, terminals=terminals, truncations=truncations,
                                        infos=infos, aux_info=aux_info, train_steps=train_steps)

            self.memory.store(obs, acts, self._process_reward(rewards), value, terminals, aux_info)
            if self.memory.full:
                vals = self.get_terminated_values(next_obs)
                for i in range(self.n_envs):
                    if terminals[i]:
                        self.memory.finish_path(0.0, i)
                    else:
                        self.memory.finish_path(vals[i], i)
                update_info = self.train_epochs(self.n_epochs)
                self.log_infos(update_info, self.current_step)
                train_info.update(update_info)
                self.callback.on_train_epochs_end(self.current_step, policy=self.policy, memory=self.memory,
                                                  current_episode=self.current_episode, train_steps=train_steps,
                                                  update_info=update_info)
                self.memory.clear()

            self.returns = self.gamma * self.returns + rewards
            obs = deepcopy(next_obs)
            for i in range(self.n_envs):
                if terminals[i] or truncations[i]:
                    self.ret_rms.update(self.returns[i:i + 1])
                    self.returns[i] = 0.0
                    if self.atari and (~truncations[i]):
                        pass
                    else:
                        if terminals[i]:
                            self.memory.finish_path(0.0, i)
                        else:
                            vals = self.get_terminated_values(next_obs)
                            self.memory.finish_path(vals[i], i)
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
            self.callback.on_train_step_end(self.current_step, envs=self.envs, policy=self.policy,
                                            train_steps=train_steps, train_info=train_info)
        return train_info
