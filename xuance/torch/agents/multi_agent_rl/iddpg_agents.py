from argparse import Namespace
from gymnasium.spaces import Space
from typing import List, Optional, Dict
from xuance.common import MultiAgentBaseCallback
from xuance.environment import DummyVecMultiAgentEnv, SubprocVecMultiAgentEnv
from xuance.torch import Module, ModuleDict
from xuance.torch.utils import ActivationFunctions
from xuance.torch.agents import OffPolicyMARLAgents
from xuance.torch.rl_models import DeterministicActor, ActionValueCritic
from xuance.torch.rl_models.modules import RNN_State
from xuance.torch.rl_models.architectures import IndependentDeterministicActorCritic


class IDDPG_Agents(OffPolicyMARLAgents):
    """The implementation of Independent DDPG agents.

    Args:
        config: The Namespace variable that provides hyperparameters and other settings.
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
        super(IDDPG_Agents, self).__init__(
            config, envs, num_agents, agent_keys, state_space, observation_space, action_space, callback
        )

        self.start_noise, self.end_noise = config.start_noise, config.end_noise
        self.noise_scale = config.start_noise
        self.delta_noise = (self.start_noise - self.end_noise) / config.running_steps
        self.sigma = config.sigma

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
            actor_networks[group_key] = DeterministicActor(
                representation=actor_feature_encoder,
                actor_hidden_size=self.config.actor_hidden_size,
                action_space=self.action_space[reference_agent],
                normalizer=self.normalize_fn,
                initializer=self.initializer,
                activation=self.activation,
                activation_action=ActivationFunctions[self.config.activation_action],
                device=self.device
            )
            # build critic feature encoder as critic representations
            critic_feature_encoder = self._build_agent_feature_encoder(
                representation_choice=self.config.representation,
                group_agents=group_agents,
                input_space=self.observation_space[reference_agent]
            )
            # build inner-group shared critic-network
            critic_networks[group_key] = ActionValueCritic(
                representation=critic_feature_encoder,
                action_space=self.action_space[reference_agent],
                critic_hidden_size=self.config.critic_hidden_size,
                normalizer=self.normalize_fn,
                initializer=self.initializer,
                activation=self.activation,
                device=self.device
            )

        # build the RL model
        model = IndependentDeterministicActorCritic(
            grouping=self.agent_grouping,
            actors=actor_networks,
            critics=critic_networks,
            use_rnn=self.use_rnn,
            device=self.device,
            use_distributed_training=self.distributed_training
        )

        return model

    def get_actions(self,
                    obs_dict: List[dict],
                    avail_actions_dict: Optional[List[dict]] = None,
                    rnn_states: Optional[Dict[str, RNN_State]] = None,
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
        obs_input, agent_indices, _ = self._build_inputs(obs_dict, avail_actions_dict)
        model_output = self.model(observations=obs_input,
                                  agent_indices=agent_indices,
                                  rnn_states=rnn_states)
        rnn_states_new = model_output.actor_rnn_states
        actions = model_output.actions

        for key in self.agent_keys:
            actions[key] = actions[key].reshape(batch_size, -1).cpu().detach().numpy()

        if not test_mode:
            actions = self.exploration(batch_size, actions)

        actions_dict = [{k: actions[k][i] for k in self.agent_keys} for i in range(batch_size)]

        return {"rnn_states": rnn_states_new, "actions": actions_dict}
