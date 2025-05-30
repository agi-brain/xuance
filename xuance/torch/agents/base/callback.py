from abc import ABC
from argparse import Namespace


class BaseCallback(ABC):

    def on_train_step(self, current_step, obs, acts, next_obs, rewards, terminals, truncations, infos, **kwargs):
        return

    def on_train_step_end(self, current_step, infos, return_info):
        return

    def on_train_episode_info(self, env_infos, env_id, rank, use_wandb):
        episode_info = {}
        if use_wandb:
            episode_info[f"Episode-Steps/rank_{rank}/env-{env_id}"] = env_infos[env_id]["episode_step"]
            episode_info[f"Train-Episode-Rewards/rank_{rank}/env-{env_id}"] = env_infos[env_id]["episode_score"]
        else:
            episode_info[f"Episode-Steps/rank_{rank}"] = {f"env-{env_id}": env_infos[env_id]["episode_step"]}
            episode_info[f"Train-Episode-Rewards/rank_{rank}"] = {f"env-{env_id}": env_infos[env_id]["episode_score"]}
        return episode_info

    def on_test_step(self, *args, **kwargs):
        return

    def on_test_end(self, *args, **kwargs):
        return
