import torch
from typing import Union
from torch import Tensor
from xuance.environment.vector_envs import DummyVecEnv, SubprocVecEnv, DummyVecMultiAgentEnv, SubprocVecMultiAgentEnv


class TensorEnvWrapper:
    def __init__(
            self,
            vec_envs: Union[DummyVecEnv, SubprocVecEnv],
            device: torch.device = torch.device("cpu")
    ):
        self.envs = vec_envs
        self.device = device

    def reset(self) -> (Tensor, dict):
        obs, infos = self.envs.reset()
        obs_tensor = Tensor(obs).to(self.device)
        return obs_tensor, infos

    def step(self, actions: Tensor) -> (Tensor, Tensor, Tensor, Tensor, dict):
        actions_np = actions.detach().cpu().numpy()
        next_obs, rewards, terminals, truncations, infos = self.envs.step(actions_np)
        next_obs_tensor = Tensor(next_obs).to(self.device)
        rewards_tensor = Tensor(rewards).to(self.device)
        terminals_tensor = Tensor(terminals).to(self.device)
        truncations_tensor = Tensor(truncations).to(self.device)
        return next_obs_tensor, rewards_tensor, terminals_tensor, truncations_tensor, infos


class TensorMultiAgentEnvWrapper:
    def __init__(
            self,
            vec_envs: Union[DummyVecMultiAgentEnv, SubprocVecMultiAgentEnv],
            device: torch.device = torch.device("cpu"),
    ):
        self.envs = vec_envs
        self.device = device

    def reset(self):
        # Convert numpy data to tensor data.
        return self.envs.reset()

    def step(self, actions):
        # Convert tensor data to numpy data, and then convert numpy data to tensor data.
        return self.envs.step(actions)
