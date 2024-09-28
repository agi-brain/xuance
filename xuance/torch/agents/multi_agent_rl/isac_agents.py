import torch
from argparse import Namespace
from xuance.common import List, Optional
from xuance.environment import DummyVecMultiAgentEnv
from xuance.torch import Module
from xuance.torch.utils import NormalizeFunctions, ActivationFunctions
from xuance.torch.policies import REGISTRY_Policy
from xuance.torch.agents import OffPolicyMARLAgents


class ISAC_Agents(OffPolicyMARLAgents):
    """The implementation of Independent SAC agents.

    Args:
        config: the Namespace variable that provides hyper-parameters and other settings.
        envs: the vectorized environments.
    """

    def __init__(self,
                 config: Namespace,
                 envs: DummyVecMultiAgentEnv):
        super(ISAC_Agents, self).__init__(config, envs)
        # build policy, optimizers, schedulers
        self.policy = self._build_policy()  # build policy
        self.memory = self._build_memory()  # build memory
        self.learner = self._build_learner(self.config, self.model_keys, self.agent_keys, self.policy)

    def _build_policy(self) -> Module:
        """
        Build representation(s) and policy(ies) for agent(s)

        Returns:
            policy (torch.nn.Module): A dict of policies.
        """
        normalize_fn = NormalizeFunctions[self.config.normalize] if hasattr(self.config, "normalize") else None
        initializer = torch.nn.init.orthogonal_
        activation = ActivationFunctions[self.config.activation]
        device = self.device
        agent = self.config.agent

        # build representations
        A_representation = self._build_representation(self.config.representation, self.observation_space, self.config)
        C_representation = self._build_representation(self.config.representation, self.observation_space, self.config)

        # build policies
        if self.config.policy == "Gaussian_ISAC_Policy":
            policy = REGISTRY_Policy["Gaussian_ISAC_Policy"](
                action_space=self.action_space, n_agents=self.n_agents,
                actor_representation=A_representation, critic_representation=C_representation,
                actor_hidden_size=self.config.actor_hidden_size,
                critic_hidden_size=self.config.critic_hidden_size,
                normalize=normalize_fn, initialize=initializer, activation=activation,
                activation_action=ActivationFunctions[self.config.activation_action],
                device=device, use_distributed_training=self.distributed_training,
                use_parameter_sharing=self.use_parameter_sharing, model_keys=self.model_keys,
                use_rnn=self.use_rnn, rnn=self.config.rnn if self.use_rnn else None)
            self.continuous_control = True
        else:
            raise AttributeError(f"{agent} currently does not support the policy named {self.config.policy}.")

        return policy

    def action(self,
               obs_dict: List[dict],
               avail_actions_dict: Optional[List[dict]] = None,
               rnn_hidden: Optional[dict] = None,
               test_mode: Optional[bool] = False):
        """
        Returns actions for agents.

        Parameters:
            obs_dict (List[dict]): Observations for each agent in self.agent_keys.
            avail_actions_dict (Optional[List[dict]]): Actions mask values, default is None.
            rnn_hidden (Optional[dict]): The hidden variables of the RNN.
            test_mode (Optional[bool]): True for testing without noises.

        Returns:
            rnn_hidden_state (dict): The new hidden states for RNN (if self.use_rnn=True).
            actions_dict (dict): The output actions.
        """
        batch_size = len(obs_dict)

        obs_input, agents_id, avail_actions_input = self._build_inputs(obs_dict)
        hidden_state, actions, _ = self.policy(observation=obs_input, agent_ids=agents_id, rnn_hidden=rnn_hidden)

        if self.use_parameter_sharing:
            key = self.model_keys[0]
            actions[key] = actions[key].reshape(batch_size, self.n_agents, -1).cpu().detach().numpy()
            actions_dict = [{k: actions[key][e, i] for i, k in enumerate(self.agent_keys)} for e in range(batch_size)]
        else:
            for key in self.agent_keys:
                actions[key] = actions[key].reshape(batch_size, -1).cpu().detach().numpy()
            actions_dict = [{k: actions[k][i] for k in self.agent_keys} for i in range(batch_size)]

        return {"hidden_state": hidden_state, "actions": actions_dict}
