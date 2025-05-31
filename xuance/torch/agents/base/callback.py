from abc import ABC


class BaseCallback(ABC):
    """Base class for callback hooks in reinforcement learning training and testing.

    Users can inherit this class to implement custom logic during different stages
    of training and evaluation.
    """
    def on_update_start(self, iterations, **kwargs):
        """Called before the policy update begins.

        Args:
            iterations (int): Number of update iterations that have performed.
            **kwargs: Additional optional keyword arguments.
        """
        info = {}
        return info

    def on_update_end(self, info, **kwargs):
        """Called after the policy update is completed.

        Args:
            info: Information of the update of the policy.
            **kwargs: Optional keyword arguments.
        """
        return info

    def on_train_step(self, current_step, obs, acts, next_obs, rewards, terminals, truncations, infos, **kwargs):
        """Called after each training step (i.e., after collecting one transition).

        Args:
            current_step (int): The current global training step.
            obs: Observations from the environment.
            acts: Actions taken by the agent.
            next_obs: Observations after the actions.
            rewards: Rewards received.
            terminals: Whether the state is terminal.
            truncations: Whether the episode was truncated.
            infos: Extra information from the environment.
            **kwargs: Additional optional information.
        """
        return

    def on_train_step_end(self, current_step, infos, return_info):
        """Called after a training step is completed (includes update, logging, etc.).

        Args:
            current_step (int): The current global training step.
            infos: Information collected during the step.
            return_info: Return values from the step or update.
        """
        return

    def on_train_episode_info(self, env_infos, env_id, rank, use_wandb):
        """Called at the termination or truncation of one episode for an environment.

        Args:
            env_infos: Raw information from the environment.
            env_id: ID of the current environment instance.
            rank: Worker or process ID (for distributed training).
            use_wandb (bool): Whether to log the data to Weights & Biases.

        Returns:
            Dict[str, Union[Dict[str, Any], Any]]: A dictionary of episode-level metrics.
        """
        return

    def on_test_step(self, *args, **kwargs):
        """Called during each step in the testing phase.

        Args:
            *args: Optional positional arguments.
            **kwargs: Optional keyword arguments.
        """
        return

    def on_test_end(self, *args, **kwargs):
        """Called at the end of the testing phase.

        Args:
            *args: Optional positional arguments.
            **kwargs: Optional keyword arguments.
        """
        return
