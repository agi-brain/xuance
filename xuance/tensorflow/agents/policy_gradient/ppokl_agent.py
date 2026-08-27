import numpy as np
from xuance.tensorflow import tf
from xuance.tensorflow.rl_models.modules import ActionOutput, split_distributions
from xuance.tensorflow.agents.policy_gradient.ppo_agent import PPO_Agent


class PPOKL_Agent(PPO_Agent):
    """The implementation of PPO agent with KL divergence.

    Args:
        config: the Namespace variable that provides hyperparameters and other settings.
        envs: the vectorized environments.
        callback: A user-defined callback function object to inject custom logic during training.
    """

    @property
    def auxiliary_info_shape(self):
        return {"old_dist": None}

    def get_aux_info(self, policy_output: ActionOutput = None):
        """Returns auxiliary information.

        Parameters:
            policy_output (dict): The output information of the policy.

        Returns:
            aux_info (dict): The auxiliary information.
        """
        aux_info = {"old_dist": policy_output.distributions}
        return aux_info

    def get_actions(
            self,
            observations: np.ndarray,
            deterministic: bool = False,
            **kwargs
    ) -> ActionOutput:
        """Returns actions and values.

        Parameters:
            observations (np.ndarray): The observation.
            deterministic (bool): True for deterministic policy and False for stochastic policy.

        Returns:
            actions: The actions to be executed.
            values: The evaluated values.
            dists: The policy distributions.
            log_pi: Log of stochastic actions.
        """
        observations = tf.convert_to_tensor(observations, dtype=tf.float32)
        model_output = self.model(observations)
        policy_dists, values = model_output.distributions, model_output.values
        actions = policy_dists.deterministic_sample() if deterministic else policy_dists.stochastic_sample()
        dists = split_distributions(policy_dists)
        actions = actions.numpy()
        values = values.numpy()
        return ActionOutput(
            env_actions=actions,
            values=values,
            distributions=dists,
        )

