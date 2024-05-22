from abc import ABC, abstractmethod


class Trainer(ABC):
    @abstractmethod
    def __init__(
        self, args, num_agents, policies, policy_mapping_fn, device, episode_length
    ):
        """
        Abstract trainer class. Performs gradient updates to policies.
        :param args: (Namespace) contains parameters needed to perform training updates.
        :param num_agents: (int) number of agents in environment.
        :param policies: (dict) maps policy_id to a policy instance (see recurrent_policy and mlp_policy).
        :param policy_mapping_fn: (function) given an agent_id, returns the policy_id of the policy controlling the agent.
        :param device: (str) device on which to perform gradient updates.
        """
        raise NotImplementedError

    @abstractmethod
    def train_policy_on_batch(self, update_policy_id, batch):
        """
        Performs a gradient update for the specified policy using a batch of sampled data.
        :param update_policy_id: (str) id of policy to update.
        :param batch: (Tuple) batch of data sampled from buffer. Batch contains observations, global observations,
                      actions, rewards, terminal states, available actions, and priority weights (for PER)
        """
        raise NotImplementedError

    @abstractmethod
    def prep_training(self):
        """Sets all networks to training mode."""
        raise NotImplementedError

    @abstractmethod
    def prep_rollout(self):
        """Sets all networks to eval mode."""
        raise NotImplementedError
