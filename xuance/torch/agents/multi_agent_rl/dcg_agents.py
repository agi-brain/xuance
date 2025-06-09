import torch
import numpy as np
from torch.nn import Module
from argparse import Namespace
from operator import itemgetter
from xuance.common import List, Optional, Union
from xuance.environment import DummyVecMultiAgentEnv, SubprocVecMultiAgentEnv
from xuance.torch.utils import NormalizeFunctions, ActivationFunctions
from xuance.torch.policies import REGISTRY_Policy
from xuance.torch.agents import OffPolicyMARLAgents, BaseCallback


class DCG_Agents(OffPolicyMARLAgents):
    """The implementation of DCG agents.

    Args:
        config: the Namespace variable that provides hyperparameters and other settings.
        envs: the vectorized environments.
        callback: A user-defined callback function object to inject custom logic during training.
    """

    def __init__(self,
                 config: Namespace,
                 envs: Union[DummyVecMultiAgentEnv, SubprocVecMultiAgentEnv],
                 callback: Optional[BaseCallback] = None):
        super(DCG_Agents, self).__init__(config, envs, callback)
        self.state_space = envs.state_space
        self.use_global_state = True if config.agent == "DCG_S" else False
        self.delta_egreedy = (self.start_greedy - self.end_greedy) / config.decay_step_greedy

        # build policy, optimizers, schedulers
        self.policy = self._build_policy()  # build policy
        self.memory = self._build_memory()  # build memory
        self.learner = self._build_learner(self.config, self.model_keys, self.agent_keys, self.policy, self.callback)

    def _build_learner(self, *args):
        from xuance.torch.learners.multi_agent_rl.dcg_learner import DCG_Learner
        return DCG_Learner(*args)

    def _build_policy(self) -> Module:
        """
        Build representation(s) and policy(ies) for agent(s)

        Returns:
            policy (torch.nn.Module): A dict of policies.
        """
        normalize_fn = NormalizeFunctions[self.config.normalize] if hasattr(self.config, "normalize") else None
        initializer = torch.nn.init.orthogonal_
        activation = ActivationFunctions[self.config.activation]

        # build representation
        representation = self._build_representation(self.config.representation, self.observation_space, self.config)

        # build policies
        from xuance.torch.policies.coordination_graph import DCG_utility, DCG_payoff, Coordination_Graph
        repre_state_dim = representation[self.model_keys[0]].output_shapes['state'][0]
        max_action_dim = max([self.action_space[key].n for key in self.agent_keys])
        utility = DCG_utility(repre_state_dim, self.config.hidden_utility_dim, max_action_dim, self.device)
        payoffs = DCG_payoff(repre_state_dim * 2, self.config.hidden_payoff_dim, max_action_dim,
                             self.config.low_rank_payoff, self.config.payoff_rank, self.device)
        dcgraph = Coordination_Graph(self.n_agents, self.config.graph_type, self.device)
        dcgraph.set_coordination_graph()

        if self.config.policy == "DCG_Policy":
            policy = REGISTRY_Policy["DCG_Policy"](
                action_space=self.action_space, n_agents=self.n_agents,
                representation=representation, utility=utility, payoffs=payoffs, dcgraph=dcgraph,
                hidden_size_bias=self.config.hidden_bias_dim if self.config.agent == "DCG_S" else None,
                normalize=normalize_fn, initializer=initializer, activation=activation, device=self.device,
                use_distributed_training=self.distributed_training, use_parameter_sharing=self.use_parameter_sharing,
                model_keys=self.model_keys, use_rnn=self.use_rnn, rnn=self.config.rnn if self.use_rnn else None)
        else:
            raise AttributeError(f"DCG currently does not support the policy named {self.config.policy}.")

        return policy

    def action(self,
               obs_dict: List[dict],
               avail_actions_dict: Optional[List[dict]] = None,
               rnn_hidden: Optional[dict] = None,
               test_mode: Optional[bool] = False,
               **kwargs):
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
        with torch.no_grad():
            rnn_hidden_next, hidden_states = self.policy.get_hidden_states(batch_size, obs_input, rnn_hidden,
                                                                           use_target_net=False)
            if self.use_actions_mask:
                if self.use_parameter_sharing:
                    avail_actions_input = avail_actions_input[self.model_keys[0]].reshape(batch_size, self.n_agents, -1)
                else:
                    avail_actions_input = np.stack(itemgetter(*self.agent_keys)(avail_actions_input),
                                                   axis=-2).reshape(batch_size, self.n_agents, -1)
            hidden_states = hidden_states.reshape([batch_size, self.n_agents, -1])
            actions = self.learner.act(hidden_states, avail_actions=avail_actions_input)

        actions_out = actions.reshape([batch_size, self.n_agents]).cpu().detach().numpy()
        actions_dict = [{k: actions_out[e, i] for i, k in enumerate(self.agent_keys)} for e in range(batch_size)]

        if not test_mode:  # get random actions
            actions_dict = self.exploration(batch_size, actions_dict, avail_actions_dict)
        return {"hidden_state": rnn_hidden_next, "actions": actions_dict}
