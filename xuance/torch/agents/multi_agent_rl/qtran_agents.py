import torch
from argparse import Namespace
from xuance.common import List, Optional
from xuance.environment import DummyVecMultiAgentEnv
from xuance.torch import Module
from xuance.torch.utils import NormalizeFunctions, ActivationFunctions
from xuance.torch.policies import REGISTRY_Policy, QTRAN_base, QTRAN_alt, VDN_mixer
from xuance.torch.agents import OffPolicyMARLAgents


class QTRAN_Agents(OffPolicyMARLAgents):
    """The implementation of QTRAN agents.

    Args:
        config: the Namespace variable that provides hyper-parameters and other settings.
        envs: the vectorized environments.
    """

    def __init__(self,
                 config: Namespace,
                 envs: DummyVecMultiAgentEnv):
        super(QTRAN_Agents, self).__init__(config, envs)
        self.state_space = envs.state_space
        self.use_global_state = True

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

        # build representations
        representation = self._build_representation(self.config.representation, self.observation_space, self.config)

        # build policies
        dim_state = self.state_space.shape[-1]
        action_space = self.action_space
        mixer = VDN_mixer()
        if self.config.agent == "QTRAN_base":
            qtran_mixer = QTRAN_base(dim_state, action_space, self.config.qtran_net_hidden_dim, self.config.n_agents,
                                     self.config.q_hidden_size[0], self.use_parameter_sharing, device)
        elif self.config.agent == "QTRAN_alt":
            qtran_mixer = QTRAN_alt(dim_state, action_space, self.config.qtran_net_hidden_dim, self.config.n_agents,
                                    self.config.q_hidden_size[0], self.use_parameter_sharing, device)
        else:
            raise ValueError("Mixer {} not recognised.".format(self.config.agent))

        if self.config.policy == "Qtran_Mixing_Q_network":
            policy = REGISTRY_Policy["Qtran_Mixing_Q_network"](
                action_space=self.action_space, n_agents=self.n_agents, representation=representation,
                mixer=mixer, qtran_mixer=qtran_mixer,
                hidden_size=self.config.q_hidden_size,
                normalize=normalize_fn, initialize=initializer, activation=activation,
                device=device, use_distributed_training=self.distributed_training,
                use_parameter_sharing=self.use_parameter_sharing, model_keys=self.model_keys,
                use_rnn=self.use_rnn, rnn=self.config.rnn if self.use_rnn else None)
        else:
            raise AttributeError(f"QMIX currently does not support the policy named {self.config.policy}.")

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
        obs_input, agents_id, avail_actions_input = self._build_inputs(obs_dict, avail_actions_dict)
        hidden_state, _, actions, _ = self.policy(observation=obs_input,
                                                  agent_ids=agents_id,
                                                  avail_actions=avail_actions_input,
                                                  rnn_hidden=rnn_hidden)

        if self.use_parameter_sharing:
            key = self.agent_keys[0]
            actions_out = actions[key].reshape([batch_size, self.n_agents]).cpu().detach().numpy()
            actions_dict = [{k: actions_out[e, i] for i, k in enumerate(self.agent_keys)} for e in range(batch_size)]
        else:
            actions_out = {k: actions[k].reshape(batch_size).cpu().detach().numpy() for k in self.agent_keys}
            actions_dict = [{k: actions_out[k][i] for k in self.agent_keys} for i in range(batch_size)]

        if not test_mode:  # get random actions
            actions_dict = self.exploration(batch_size, actions_dict, avail_actions_dict)
        return {"hidden_state": hidden_state, "actions": actions_dict}
