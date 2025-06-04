from abc import ABC


class BaseCallback(ABC):
    """Base class for callback hooks in reinforcement learning training and testing.

    Users can inherit this class to implement custom logic during different stages
    of training and evaluation.
    """
    def __init__(self, *args, **kwargs):
        self.logger = kwargs.get('logger')

    def on_update_start(self, iterations, **kwargs):
        """Called before the policy update begins.

        Args:
            iterations (int): Number of update iterations that have performed.
            **kwargs: Additional optional keyword arguments.
        """
        return {}

    def on_update_end(self, iterations, **kwargs):
        """Called after the policy update is completed.

        Args:
            iterations (int): Number of update iterations that have performed.
            **kwargs: Optional keyword arguments.
        """
        return {}

    def on_train_step(self, current_step, **kwargs):
        """Called after each training step (i.e., after collecting one transition).

        Args:
            current_step (int): The current global training step.
            **kwargs: Additional optional information.
        """
        return

    def on_train_epochs_end(self, current_step, **kwargs):
        """Called after each training epoch (i.e., after collecting one transition).
        Args:
            current_step (int): The current global training step.
            **kwargs: Additional optional information.
        """
        return

    def on_train_episode_info(self, **kwargs):
        """Called at the termination or truncation of one episode for an environment.
        """
        return

    def on_train_step_end(self, current_step, **kwargs):
        """Called after a training step is completed (includes update, logging, etc.).

        Args:
            current_step (int): The current global training step.
            envs_info: Environment information.
            train_info: Training information.
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


class MultiAgentBaseCallback(BaseCallback):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def on_update_agent_wise(self, iterations, agent_key, **kwargs):
        return {}
