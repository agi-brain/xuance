import torch
import gymnasium
import numpy as np
from argparse import Namespace
from gymnasium.spaces import Space
from typing import List, Optional, Dict
from xuance.common import MultiAgentBaseCallback
from xuance.environment import DummyVecMultiAgentEnv, SubprocVecMultiAgentEnv
from xuance.torch import Module, ModuleDict
from xuance.torch.utils import ActivationFunctions
from xuance.torch.agents import OnPolicyMARLAgents
from xuance.torch.rl_models import CategoricalActor, GaussianActor
from xuance.torch.rl_models import StateValueCritic as Critic
from xuance.torch.rl_models.modules import RNN_State, MARLActionOutput
from xuance.torch.rl_models.architectures import IndependentActorCritic


class IAC_Agents(OnPolicyMARLAgents):
    """The implementation of IAC agents.

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
        super(IAC_Agents, self).__init__(
            config, envs, num_agents, agent_keys, state_space, observation_space, action_space, callback
        )
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
            critic_feature_encoder = self._build_agent_feature_encoder(
                representation_choice=self.config.representation,
                group_agents=group_agents,
                input_space=self.observation_space[reference_agent]
            )
            # build inner-group shared critic-network
            critic_networks[group_key] = Critic(
                representation=critic_feature_encoder,
                critic_hidden_size=self.config.critic_hidden_size,
                normalizer=self.normalize_fn,
                initializer=self.initializer,
                activation=self.activation,
                device=self.device
            )

        # build the RL model
        model = IndependentActorCritic(
            grouping=self.agent_grouping,
            actors=actor_networks,
            critics=critic_networks,
            use_rnn=self.use_rnn,
            device=self.device,
            use_distributed_training=self.distributed_training
        )

        return model

    def store_experience(self, obs_list, avail_actions, actions_list, log_pi_a, rewards_list, values_list,
                         terminals_list, info, **kwargs):
        """
        Store experience data into replay buffer.

        Parameters:
            obs_list (List[dict]): Observations for each agent in self.agent_keys.
            avail_actions (List[dict]): Actions mask values for each agent in self.agent_keys.
            actions_list (List[dict]): Actions for each agent in self.agent_keys.
            log_pi_a (dict): The log of pi.
            rewards_list (List[dict]): Rewards for each agent in self.agent_keys.
            values_list (dict): Critic values for each agent in self.agent_keys.
            terminals_list (List[dict]): Terminated values for each agent in self.agent_keys.
            info (List[dict]): Other information for the environment at current step.
            **kwargs: Other inputs.
        """
        experience_data = {
            'obs': {k: np.array([data[k] for data in obs_list]) for k in self.agent_keys},
            'actions': {k: np.array([data[k] for data in actions_list]) for k in self.agent_keys},
            # 'log_pi_old': log_pi_a,
            'rewards': {k: np.array([np.array(list(data.values())).mean() for data in rewards_list])
                        for k in self.agent_keys},
            'values': values_list,
            'terminals': {k: np.array([data[k] for data in terminals_list]) for k in self.agent_keys},
            'agent_mask': {k: np.array([data['agent_mask'][k] for data in info]) for k in self.agent_keys},
        }
        if self.use_rnn:
            experience_data['episode_steps'] = np.array([data['episode_step'] - 1 for data in info])
        if self.use_global_state:
            experience_data['state'] = np.array(kwargs['state'])
        if self.use_actions_mask:
            experience_data['avail_actions'] = {k: np.array([data[k] for data in avail_actions])
                                                for k in self.agent_keys}
        self.memory.store(**experience_data)

    @torch.no_grad()
    def get_actions(
            self,
            obs_list: List[dict],
            state: Optional[np.ndarray] = None,
            avail_actions_list: Optional[List[dict]] = None,
            rnn_states_actor: Optional[Dict[str, RNN_State]] = None,
            rnn_states_critic: Optional[Dict[str, RNN_State]] = None,
            test_mode: Optional[bool] = False,
            deterministic: bool = False,
            **kwargs
    ) -> MARLActionOutput:
        """
        Returns actions for agents.

        Parameters:
            obs_list (dict): Observations for each agent in self.agent_keys.
            state (Optional[np.ndarray]): The global state.
            avail_actions_list (Optional[List[dict]]): Actions mask values, default is None.
            rnn_states_actor (Optional[dict]): The RNN hidden states of actor representation.
            rnn_states_critic (Optional[dict]): The RNN hidden states of critic representation.
            test_mode (Optional[bool]): True for testing without noises.
            deterministic (bool): True for deterministic policy and False for stochastic policy.

        Returns:
            rnn_states_actor_new (dict): The new RNN hidden states of actor representation (if self.use_rnn=True).
            rnn_states_critic_new (dict): The new RNN hidden states of critic representation (if self.use_rnn=True).
            actions_list (dict): The output actions.
            log_pi_a (dict): The log of pi.
            values_dict (dict): The evaluated critic values (when test_mode is False).
        """
        batch_size = len(obs_list)
        rnn_states_critic_new, values_dict = {}, {}

        obs_input, agent_indices, avail_actions_input = self._build_inputs(obs_list, avail_actions_list)
        model_output = self.model(observations=obs_input,
                                  agent_indices=agent_indices,
                                  avail_actions=avail_actions_input,
                                  rnn_states=rnn_states_actor,
                                  deterministic=deterministic)
        rnn_states_actor_new = model_output.actor_rnn_states
        actions = model_output.actions

        actions.grouped_tensor = {k: actions.grouped_tensor[k].reshape(batch_size, n).cpu().numpy()
                                  for k, n in self.n_group_agents.items()}
        if self.continuous_control:
            actions_list = [{k: actions.agent_wise[k][e].reshape([-1]) for k in self.agent_keys}
                            for e in range(batch_size)]
        else:
            actions_list = [{k: actions.agent_wise[k][e].reshape([]) for k in self.agent_keys}
                            for e in range(batch_size)]

        if not test_mode:
            values_model_output = self.model.get_values(observations=obs_input,
                                                        agent_indices=agent_indices,
                                                        rnn_states=rnn_states_critic)
            rnn_states_critic_new = values_model_output.critic_rnn_states
            values = values_model_output.values
            values.grouped_tensor = {k: v.cpu().numpy() for k, v in values.grouped_tensor.items()}
            values_dict = {k: v.reshape(batch_size) for k, v in values.agent_wise.items()}

        return MARLActionOutput(
            env_actions=actions_list,
            values=values_dict,
            rnn_states_actor=rnn_states_actor_new,
            rnn_states_critic=rnn_states_critic_new
        )

