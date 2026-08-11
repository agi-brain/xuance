import gymnasium
import torch
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
from xuance.torch.rl_models.modules import RNN_State
from xuance.torch.rl_models.heads import VDN_Mixer, QMIX_Mixer
from xuance.torch.rl_models.architectures import ValueDecompositionActorCritic


class VDAC_Agents(OnPolicyMARLAgents):
    """The implementation of VDAC agents.

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
        super(VDAC_Agents, self).__init__(
            config, envs, num_agents, agent_keys, state_space, observation_space, action_space, callback
        )
        self.state_space = envs.state_space
        self.mixer = config.mixer

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

        if self.mixer == "VDN":
            mixer = VDN_Mixer()
        elif self.mixer == "QMIX":
            dim_state = self.state_space.shape[-1]
            mixer = QMIX_Mixer(dim_state, self.config.hidden_dim_mixing_net, self.config.hidden_dim_hyper_net,
                               self.n_agents, self.device)
            self.use_global_state = True
        else:
            raise NotImplementedError

        # build the RL model
        model = ValueDecompositionActorCritic(
            grouping=self.agent_grouping,
            actors=actor_networks,
            critics=critic_networks,
            mixer=mixer,
            use_rnn=self.use_rnn,
            device=self.device,
            use_distributed_training=self.distributed_training
        )

        return model

    def store_experience(self, obs_dict, avail_actions, actions_dict, log_pi_a, rewards_dict, values_dict,
                         terminals_dict, info, **kwargs):
        """
        Store experience data into replay buffer.

        Parameters:
            obs_dict (List[dict]): Observations for each agent in self.agent_keys.
            avail_actions (List[dict]): Actions mask values for each agent in self.agent_keys.
            actions_dict (List[dict]): Actions for each agent in self.agent_keys.
            log_pi_a (dict): The log of pi.
            rewards_dict (List[dict]): Rewards for each agent in self.agent_keys.
            values_dict (dict): Critic values for each agent in self.agent_keys.
            terminals_dict (List[dict]): Terminated values for each agent in self.agent_keys.
            info (List[dict]): Other information for the environment at current step.
            **kwargs: Other inputs.
        """
        experience_data = {
            'obs': {k: np.array([data[k] for data in obs_dict]) for k in self.agent_keys},
            'actions': {k: np.array([data[k] for data in actions_dict]) for k in self.agent_keys},
            # 'log_pi_old': log_pi_a,
            'rewards': {k: np.array([np.array(list(data.values())).mean() for data in rewards_dict])
                        for k in self.agent_keys},
            'values': values_dict,
            'terminals': {k: np.array([data[k] for data in terminals_dict]) for k in self.agent_keys},
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

    def get_actions(self,
                    obs_dict: List[dict],
                    state: Optional[np.ndarray] = None,
                    avail_actions_dict: Optional[List[dict]] = None,
                    rnn_states_actor: Dict[str, RNN_State] = None,
                    rnn_states_critic: Dict[str, RNN_State] = None,
                    test_mode: Optional[bool] = False,
                    deterministic: Optional[bool] = False,
                    **kwargs):
        """
        Returns actions for agents.

        Parameters:
            obs_dict (dict): Observations for each agent in self.agent_keys.
            state (Optional[np.ndarray]): The global state.
            avail_actions_dict (Optional[List[dict]]): Actions mask values, default is None.
            rnn_states_actor (Optional[dict]): The RNN hidden states of actor representation.
            rnn_states_critic (Optional[dict]): The RNN hidden states of critic representation.
            test_mode (Optional[bool]): True for testing without noises.
            deterministic (bool): True for deterministic policy and False for stochastic policy.

        Returns:
            rnn_states_actor_new (dict): The new RNN hidden states of actor representation (if self.use_rnn=True).
            rnn_states_critic_new (dict): The new RNN hidden states of critic representation (if self.use_rnn=True).
            actions_dict (dict): The output actions.
            log_pi_a (dict): The log of pi.
            values_dict (dict): The evaluated critic values (when test_mode is False).
        """
        n_env = len(obs_dict)
        rnn_states_critic_new, values_dict = {}, {}

        obs_input, agent_indices, avail_actions_input = self._build_inputs(obs_dict, avail_actions_dict)
        model_output = self.model(observations=obs_input,
                                  agent_indices=agent_indices,
                                  avail_actions=avail_actions_input,
                                  rnn_states=rnn_states_actor,
                                  deterministic=deterministic)
        rnn_states_actor_new = model_output.actor_rnn_states

        if not test_mode:
            values_model_output = self.model.get_values(observations=obs_input,
                                                        agent_indices=agent_indices,
                                                        rnn_states=rnn_states_critic)
            rnn_states_critic_new = values_model_output.critic_rnn_states
            state = torch.as_tensor(state, device=self.device)
            values_tot = self.model.values_tot(values_model_output.values, state).detach().cpu().numpy().reshape(n_env)
            values_dict = {k: values_tot for k in self.agent_keys}

        actions_sample = {k: v.cpu().detach().numpy() for k, v in model_output.actions.items()}
        if self.continuous_control:
            actions_dict = [{k: actions_sample[k][e].reshape([-1]) for k in self.agent_keys} for e in range(n_env)]
        else:
            actions_dict = [{k: actions_sample[k][e].reshape([]) for k in self.agent_keys} for e in range(n_env)]

        return {"rnn_states_actor": rnn_states_actor_new, "rnn_states_critic": rnn_states_critic_new,
                "actions": actions_dict, "log_pi": None, "values": values_dict}

    def values_next(self,
                    i_env: int,
                    obs_dict: dict,
                    state: Optional[np.ndarray] = None,
                    rnn_states_critic: Optional[dict] = None):
        """
        Returns critic values of one environment that finished an episode.

        Parameters:
            i_env (int): The index of environment.
            obs_dict (dict): Observations for each agent in self.agent_keys.
            state (Optional[np.ndarray]): The global state.
            rnn_states_critic (Optional[dict]): The RNN hidden states of critic representation.

        Returns:
            rnn_states_critic_new (dict): The new RNN hidden states of critic representation (if self.use_rnn=True).
            values_dict: The critic values.
        """
        if self.use_rnn:
            rnn_states_critic_i = {}
            for group, n_agents in self.n_group_agents.items():
                hidden_item_index = np.arange(i_env * n_agents, (i_env + 1) * n_agents)
                rnn_states_critic_i[group] = self.model.critics[
                    group].representation.obs_representation.get_rnn_states_item(
                    hidden_item_index, rnn_states_critic[group])
        else:
            rnn_states_critic_i = None

        obs_input, agent_indices, _ = self._build_inputs([obs_dict])
        values_model_output = self.model.get_values(observations=obs_input,
                                                    agent_indices=agent_indices,
                                                    rnn_states=rnn_states_critic_i)
        rnn_states_critic_new_i = values_model_output.critic_rnn_states
        values_tot = self.model.values_tot(values_model_output.values, state).detach().cpu().numpy().reshape([])
        values_dict = {k: values_tot for k in self.agent_keys}

        return rnn_states_critic_new_i, values_dict
