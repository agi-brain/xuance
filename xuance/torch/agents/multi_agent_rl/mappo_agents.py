import torch
import gymnasium
import numpy as np
from argparse import Namespace
from gymnasium.spaces import Space
from typing import List, Optional, Dict, Tuple
from xuance.common import MultiAgentBaseCallback
from xuance.environment import DummyVecMultiAgentEnv, SubprocVecMultiAgentEnv
from xuance.torch import Module, ModuleDict
from xuance.torch.utils import ActivationFunctions
from xuance.torch.agents.multi_agent_rl import IPPO_Agents
from xuance.torch.rl_models import CategoricalActor, GaussianActor
from xuance.torch.rl_models import CentralizedStateValueCritic as Critic
from xuance.torch.rl_models.modules import RNN_State
from xuance.torch.rl_models.architectures import MultiAgentActorCritic


class MAPPO_Agents(IPPO_Agents):
    """The implementation of MAPPO agents.

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
        super(MAPPO_Agents, self).__init__(
            config, envs, num_agents, agent_keys, state_space, observation_space, action_space, callback
        )

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
            Actor = GaussianActor
            actor_input['activation_action'] = ActivationFunctions[self.config.activation_action]
            self.continuous_control = True
        elif isinstance(self.action_space[self.agent_keys[0]], gymnasium.spaces.Discrete):
            Actor = CategoricalActor
            self.continuous_control = False
        else:
            raise NotImplementedError

        actor_networks = ModuleDict()
        critic_networks = ModuleDict()
        critic_feature_encoder = ModuleDict()
        for group_key, group_agents in self.groups.items():
            reference_agent = group_agents[0]
            # build agent feature encoder as actor representations
            actor_feature_encoder = self._build_agent_feature_encoder(
                representation_choice=self.config.representation,
                group_agents=group_agents,
                input_space=self.observation_space[reference_agent]
            )
            actor_input['representation'] = actor_feature_encoder
            actor_input['action_space'] = self.action_space[reference_agent]
            # build inner-group shared actor-network
            actor_networks[group_key] = Actor(**actor_input)
            # build critic feature encoder as critic representations
            critic_feature_encoder[group_key] = self._build_agent_feature_encoder(
                representation_choice=self.config.representation,
                group_agents=group_agents,
                input_space=self.observation_space[reference_agent]
            )

        # build inner-group shared critic-network
        for group_key, group_agents in self.groups.items():
            critic_networks[group_key] = Critic(
                grouping=self.agent_grouping,
                representations=critic_feature_encoder,
                state_space=self.state_space,
                critic_hidden_size=self.config.critic_hidden_size,
                normalizer=self.normalize_fn,
                initializer=self.initializer,
                activation=self.activation,
                use_rnn=self.use_rnn,
                device=self.device
            )

        # build the RL model
        model = MultiAgentActorCritic(
            grouping=self.agent_grouping,
            actors=actor_networks,
            critics=critic_networks,
            use_rnn=self.use_rnn,
            device=self.device,
            use_distributed_training=self.distributed_training
        )

        return model

    @torch.no_grad()
    def values_next(
            self,
            i_env: int,
            obs_dict: dict,
            state: Optional[np.ndarray] = None,
            rnn_states_critic: Dict[str, Dict[str, RNN_State]] = None
    ) -> Tuple[Dict[str, Dict[str, RNN_State]], Dict[str, np.ndarray]]:
        """Compute bootstrapped critic values for an environment that reached a boundary.

        This method evaluates the critic on the terminal/next observations of a specific
        vectorized environment (`i_env`) and returns per-agent value estimates used for bootstrapping
        when finalizing trajectories (e.g., for GAE/return computation).

        Args:
            i_env (int): Index of the vectorized environment that is finishing an episode or trajectory segment.
            obs_dict (dict): Per-agent observations for the selected environment.
                This dict is keyed by `self.agent_keys`.
            state (Optional[np.ndarray]): Global state for the selected environment when `use_global_state=True`.
                If provided, it should correspond to the same `i_env` instance.
            rnn_states_critic (Optional[dict]): Current critic RNN hidden states keyed by `self.model_keys`.
                Required when `self.use_rnn` is True.

        Returns:
            Tuple[Optional[dict], dict]: A tuple of `(rnn_states_critic_new, values_dict)`:
                - rnn_states_critic_new (Optional[dict]): Updated critic hidden states for the selected environment
                    when `self.use_rnn` is True; otherwise the value returned by the critic (typically None).
                - values_dict (dict): Per-agent critic value estimates keyed by `self.agent_keys`.
        """
        if self.use_rnn:
            rnn_states_critic_i = {}
            for group, n_agents in self.n_group_agents.items():
                hidden_item_index = np.arange(i_env * n_agents, (i_env + 1) * n_agents)
                rnn_states_critic_i[group] = {k: self.model.critics[
                    group].representations[k].obs_representation.get_rnn_states_item(
                    hidden_item_index, rnn_states_critic[group][k]) for k in self.group_keys}
        else:
            rnn_states_critic_i = None

        obs_input, agent_indices, _ = self._build_inputs([obs_dict])

        values_model_output = self.model.get_values(state=state if self.use_global_state else None,
                                                    observations=obs_input,
                                                    agent_indices=agent_indices,
                                                    rnn_states=rnn_states_critic_i)
        rnn_states_critic_new_i = values_model_output.critic_rnn_states
        values = values_model_output.values
        values.grouped_tensor = {k: v.cpu().numpy() for k, v in values.grouped_tensor.items()}
        values_dict = {k: v.reshape([]) for k, v in values.agent_wise.items()}

        return rnn_states_critic_new_i, values_dict

