import torch
import gymnasium
from argparse import Namespace
from gymnasium.spaces import Space
from typing import List, Optional, Dict
from xuance.common import MultiAgentBaseCallback
from xuance.environment import DummyVecMultiAgentEnv, SubprocVecMultiAgentEnv
from xuance.torch import Module, ModuleDict
from xuance.torch.utils import ActivationFunctions
from xuance.torch.agents import OffPolicyMARLAgents
from xuance.torch.rl_models import (CategoricalActor, SAC_GaussianActor, TwinActionValueCritic,
                                    TwinDiscreteActionValueCritic)
from xuance.torch.rl_models.modules import RNN_State, MARLActionOutput
from xuance.torch.rl_models.architectures import IndependentSoftActorCritic


class ISAC_Agents(OffPolicyMARLAgents):
    """The implementation of Independent SAC agents.

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
        super(ISAC_Agents, self).__init__(
            config, envs, num_agents, agent_keys, state_space, observation_space, action_space, callback
        )
        # build policy, optimizers, schedulers
        self.model = self._build_model()  # build the MARL model
        self.memory = self._build_memory()  # build memory
        self.learner = self._build_learner(self.config, self.agent_grouping, self.model, self.callback)

    def _build_model(self) -> Module:
        """
        Build the MARL model.

        Returns:
            model (torch.nn.Module): The MARL model.
        """
        actor_input = dict(
            actor_hidden_size=self.config.actor_hidden_size,
            normalizer=self.normalize_fn,
            initializer=self.initializer,
            activation=self.activation,
            device=self.device
        )
        if isinstance(self.action_space[self.agent_keys[0]], gymnasium.spaces.Box):
            Actor = SAC_GaussianActor
            actor_input['activation_action'] = ActivationFunctions[self.config.activation_action]
            Critic = TwinActionValueCritic
            Architecture = IndependentSoftActorCritic
            self.continuous_control = True
        elif isinstance(self.action_space[self.agent_keys[0]], gymnasium.spaces.Discrete):
            Actor = CategoricalActor
            Critic = TwinDiscreteActionValueCritic
            # Architecture = IndependentSoftActorCriticDiscrete
            self.continuous_control = False
        else:
            raise NotImplementedError

        actor_networks = ModuleDict()
        critic_networks = ModuleDict()
        for group_key, group_agents in self.groups.items():
            reference_agent = group_agents[0]
            # build agent feature encoder as actor representations
            actor_feature_encoder = self._build_agent_feature_encoder(
                representation_choice=self.config.representation,
                group_agents=group_agents,
                input_space=self.observation_space[reference_agent]
            )
            # build inner-group shared actor-network
            actor_input['representation'] = actor_feature_encoder
            actor_input['action_space'] = self.action_space[reference_agent]
            actor_networks[group_key] = Actor(**actor_input)
            # build critic feature encoder as critic representations
            critic_feature_encoder = self._build_agent_feature_encoder(
                representation_choice=self.config.representation,
                group_agents=group_agents,
                input_space=self.observation_space[reference_agent]
            )
            # build inner-group shared critic-network
            critic_networks[group_key] = Critic(
                representation=critic_feature_encoder,
                action_space=self.action_space[reference_agent],
                critic_hidden_size=self.config.critic_hidden_size,
                normalizer=self.normalize_fn,
                initializer=self.initializer,
                activation=self.activation,
                device=self.device
            )

        # build the RL model
        model = Architecture(
            grouping=self.agent_grouping,
            actors=actor_networks,
            critics=critic_networks,
            use_rnn=self.use_rnn,
            device=self.device,
            use_distributed_training=self.distributed_training
        )

        return model

    @torch.no_grad()
    def get_actions(
            self,
            obs_list: List[dict],
            avail_actions_list: Optional[List[dict]] = None,
            rnn_states: Optional[Dict[str, RNN_State]] = None,
            test_mode: Optional[bool] = False,
            **kwargs
    ) -> MARLActionOutput:
        """
        Returns actions for agents.

        Parameters:
            obs_list (List[dict]): Observations for each agent in self.agent_keys.
            avail_actions_list (Optional[List[dict]]): Actions mask values, default is None.
            rnn_states (Optional[Dict[str, RNN_State]]): The hidden variables of the RNN.
            test_mode (Optional[bool]): True for testing without noises.

        Returns:
            rnn_states (dict): The new hidden states for RNN (if self.use_rnn=True).
            actions_list (dict): The output actions.
        """
        batch_size = len(obs_list)

        obs_input, agent_indices, avail_actions_input = self._build_inputs(obs_list, avail_actions_list)

        model_output = self.model(observations=obs_input,
                                  agent_indices=agent_indices,
                                  avail_actions=avail_actions_input,
                                  rnn_states=rnn_states)
        rnn_states_new = model_output.actor_rnn_states
        actions = model_output.actions

        if self.continuous_control:
            actions.grouped_tensor = {
                k: actions.grouped_tensor[k].reshape(batch_size, n, -1).cpu().numpy() for k, n in
                self.n_group_agents.items()
            }
            actions_list = [{
                k: actions.agent_wise[k][e].reshape([-1]) for k in self.agent_keys
            } for e in range(batch_size)]
        else:
            actions.grouped_tensor = {
                k: actions.grouped_tensor[k].reshape(batch_size, n).cpu().numpy() for k, n in
                self.n_group_agents.items()
            }
            actions_list = [{
                k: actions.agent_wise[k][e].reshape([]) for k in self.agent_keys
            } for e in range(batch_size)]

        return MARLActionOutput(
            env_actions=actions_list,
            rnn_states=rnn_states_new
        )

