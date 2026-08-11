import torch
import numpy as np
from torch.nn import Module
from argparse import Namespace
from operator import itemgetter
from gymnasium.spaces import Space
from xuance.common import List, Optional, MultiAgentBaseCallback
from xuance.environment import DummyVecMultiAgentEnv, SubprocVecMultiAgentEnv
from xuance.torch import ModuleDict
from xuance.torch.agents import OffPolicyMARLAgents
from xuance.torch.rl_models.heads import ValueHead, DCG_Utility, DCG_Payoff, Coordination_Graph
from xuance.torch.rl_models.architectures import DeepCoordinationGraph


class DCG_Agents(OffPolicyMARLAgents):
    """The implementation of DCG agents.

    Args:
        config: the Namespace variable that provides hyperparameters and other settings.
        envs: the vectorized environments.
        callback: A user-defined callback function object to inject custom logic during training.
    """

    def __init__(
            self,
            config: Namespace,
            envs: Optional[DummyVecMultiAgentEnv | SubprocVecMultiAgentEnv] = None,
            num_agents: Optional[int] = None,
            agent_keys: Optional[List[str]] = None,
            state_space: Optional[Space] = None,
            observation_space: Optional[Space] = None,
            action_space: Optional[Space] = None,
            callback: Optional[MultiAgentBaseCallback] = None
    ):
        super(DCG_Agents, self).__init__(
            config, envs, num_agents, agent_keys, state_space, observation_space, action_space, callback
        )
        self.state_space = envs.state_space
        self.use_global_state = True if config.agent == "DCG_S" else False
        self.delta_egreedy = (self.start_greedy - self.end_greedy) / config.decay_step_greedy

        # build policy, optimizers, schedulers
        self.model = self._build_model()  # build the MARL model
        self.memory = self._build_memory()  # build memory
        self.learner = self._build_learner(self.config, self.agent_grouping, self.model, self.callback)

    def _build_learner(self, *args):
        from xuance.torch.learners.multi_agent_rl.dcg_learner import DCG_Learner
        return DCG_Learner(*args)

    def _build_model(self) -> Module:
        """
        Build the MARL model.

        Returns:
            model (torch.nn.Module): The MARL model.
        """
        representations = ModuleDict()
        for group_key, group_agents in self.groups.items():
            reference_agent = group_agents[0]
            # build agent feature encoder as representations
            representations[group_key] = self._build_agent_feature_encoder(
                representation_choice=self.config.representation,
                group_agents=group_agents,
                input_space=self.observation_space[reference_agent]
            )

        repre_state_dim = representations[self.group_keys[0]].output_shapes['state'][0]
        max_action_dim = max([self.action_space[key].n for key in self.agent_keys])
        utility = DCG_Utility(repre_state_dim, self.config.hidden_utility_dim, max_action_dim, self.device)
        payoffs = DCG_Payoff(repre_state_dim * 2, self.config.hidden_payoff_dim, max_action_dim,
                             self.config.low_rank_payoff, self.config.payoff_rank, self.device)
        dcg_graph = Coordination_Graph(self.n_agents, self.config.graph_type, self.device)
        dcg_graph.set_coordination_graph()

        if self.config.agent == "DCG_S":
            hidden_size_bias = self.config.hidden_bias_dim
            dcg_s = True
            state_dim = self.state_space.shape[0]
            self.bias = ValueHead(
                feature_dim=state_dim,
                hidden_size=hidden_size_bias,
                normalizer=self.normalize_fn,
                initializer=self.initializer,
                activation=self.activation,
                device=self.device
            )
        else:
            dcg_s = False
            self.bias = None

        model = DeepCoordinationGraph(
            grouping=self.agent_grouping,
            action_space=self.action_space,
            representation=representations,
            utility=utility,
            payoffs=payoffs,
            dcg_graph=dcg_graph,
            dcg_s=dcg_s,
            bias=self.bias,
            use_rnn=self.use_rnn,
            device=self.device,
            use_distributed_training=self.distributed_training
        )

        return model

    def get_actions(self,
                    obs_dict: List[dict],
                    avail_actions_dict: Optional[List[dict]] = None,
                    rnn_states: Optional[dict] = None,
                    test_mode: Optional[bool] = False,
                    **kwargs):
        """
        Returns actions for agents.

        Parameters:
            obs_dict (List[dict]): Observations for each agent in self.agent_keys.
            avail_actions_dict (Optional[List[dict]]): Actions mask values, default is None.
            rnn_states (Optional[dict]): The hidden variables of the RNN.
            test_mode (Optional[bool]): True for testing without noises.

        Returns:
            rnn_states (dict): The new hidden states for RNN (if self.use_rnn=True).
            actions_dict (dict): The output actions.
        """
        batch_size = len(obs_dict)
        obs_input, agent_indices, avail_actions_input = self._build_inputs(obs_dict, avail_actions_dict)
        with torch.no_grad():
            rnn_states_new, hidden_states = self.model.get_hidden_states(observations=obs_input,
                                                                         agent_indices=agent_indices,
                                                                         rnn_states=rnn_states,
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
        return {"rnn_states": rnn_states_new, "actions": actions_dict}
