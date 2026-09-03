from argparse import Namespace
from gymnasium.spaces import Space
from typing import List, Optional, Dict, Tuple
from xuance.common import MultiAgentBaseCallback
from xuance.environment import DummyVecMultiAgentEnv, SubprocVecMultiAgentEnv

from xuance.tensorflow import tf, Tensor, Module, ModuleDict
from xuance.tensorflow.utils import ActivationFunctions
from xuance.tensorflow.agents import OffPolicyMARLAgents
from xuance.tensorflow.rl_models import DeterministicActor, ActionValueCritic
from xuance.tensorflow.rl_models.modules import RNN_State, MARLActionOutput, AgentGroupedTensor
from xuance.tensorflow.rl_models.architectures import IndependentDeterministicActorCritic


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
                normalizer=self.normalizer_fn,
                initializer=self.initializer,
                activation=self.activation,
                activation_action=ActivationFunctions[self.config.activation_action]
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
                normalizer=self.normalizer_fn,
                initializer=self.initializer,
                activation=self.activation
            )

        # build the RL model
        model = IndependentDeterministicActorCritic(
            grouping=self.agent_grouping,
            actors=actor_networks,
            critics=critic_networks,
            use_rnn=self.use_rnn,
            use_distributed_training=self.distributed_training
        )

        return model

    @tf.function(reduce_retracing=True)
    def _rollout_step(
            self,
            observations: Dict[str, Tensor],
            agent_indices: Dict[str, Tensor],
            noise_scale: Tensor,
            **kwargs
    ) -> Tuple[Dict[str, Tuple], Dict[str, Tensor]]:
        observations = AgentGroupedTensor(observations, self.agent_grouping)
        agent_indices = AgentGroupedTensor(agent_indices, self.agent_grouping)
        if self.use_actions_mask:
            avail_actions = AgentGroupedTensor(kwargs["avail_actions"], self.agent_grouping)
        else:
            avail_actions = None
        if self.use_rnn:
            rnn_states = {
                k: RNN_State(hidden_states=v[0], cell_states=v[1] if len(v) > 1 else None)
                for k, v in kwargs["rnn_states"].items()
            }
        else:
            rnn_states = None

        model_output = self.model(observations=observations,
                                  agent_indices=agent_indices,
                                  avail_actions=avail_actions,
                                  rnn_states=rnn_states)
        if self.use_rnn:
            rnn_states_new = {
                k: (v.hidden_states,) if v.cell_states is None else (v.hidden_states, v.cell_states)
                for k, v in model_output.actor_rnn_states.items()
            }
        else:
            rnn_states_new = None
        pi_actions = model_output.actions.grouped_tensor

        # Exploration
        actions = {}
        for group in self.group_keys:
            group_actions = pi_actions[group]

            noise = tf.random.normal(shape=tf.shape(group_actions), dtype=group_actions.dtype)
            explore_actions = group_actions + noise * noise_scale
            actions_low = self.action_space[self.agent_grouping.agents_in(group)[0]].low
            actions_high = self.action_space[self.agent_grouping.agents_in(group)[0]].high
            explore_actions = tf.clip_by_value(explore_actions,
                                               tf.cast(actions_low, group_actions.dtype),
                                               tf.cast(actions_high, group_actions.dtype))
            actions[group] = explore_actions

        return rnn_states_new, actions

    def get_actions(self,
                    obs_list: List[dict],
                    avail_actions_list: Optional[List[dict]] = None,
                    rnn_states: Optional[Dict[str, RNN_State]] = None,
                    test_mode: Optional[bool] = False,
                    **kwargs) -> MARLActionOutput:
        """Compute actions for all agents given vectorized observations.

        This method performs a forward pass through the current multi-agent policy to obtain actions for each agent in
        each parallel environment. When RNN-based representations are enabled, it also consumes and returns recurrent
        hidden states. During training (`test_mode=False`), this method applies the configured exploration strategy
        (epsilon-greedy or additive noise); during evaluation (`test_mode=True`), exploration is disabled.

        Args:
            obs_list (List[dict]): Observations for each parallel environment.
                Each element is a dict keyed by `self.agent_keys`.
            avail_actions_list (Optional[List[dict]]): Available-action masks for each parallel environment when
                `use_actions_mask=True`. Each element is a dict keyed by `self.agent_keys`. Can be None when
                action masking is disabled.
            rnn_states (Optional[Dict[str, RNN_State]]): Current RNN hidden states keyed by `self.group_keys`.
                Required when `self.use_rnn` is True.
            test_mode (bool): Whether to run in evaluation mode. When True, exploration is disabled and actions are
                produced deterministically (or without training-time noise).

        Returns:
            dict: A dictionary containing:
                - hidden_state (Optional[dict]): Updated RNN hidden states when `self.use_rnn` is True;
                    otherwise the value returned by the policy (typically None).
                - actions (List[dict]): Actions for each parallel environment.
                    Each element is a dict keyed by `self.agent_keys`.
        """
        batch_size = len(obs_list)
        obs_input, agent_indices_input, avail_actions_input = self._build_inputs(obs_list, avail_actions_list)
        noise_scale = tf.convert_to_tensor(0.0 if test_mode else self.noise_scale, dtype=tf.float32)

        rollout_kwargs = {}
        if self.use_actions_mask:
            rollout_kwargs["avail_actions"] = avail_actions_input.grouped_tensor
        if self.use_rnn:
            rollout_kwargs["rnn_states"] = {
                k: (v.hidden_states,) if v.cell_states is None else (v.hidden_states, v.cell_states)
                for k, v in rnn_states.items()
            }

        rnn_states_new, actions = self._rollout_step(observations=obs_input.grouped_tensor,
                                                     agent_indices=agent_indices_input.grouped_tensor,
                                                     noise_scale=noise_scale,
                                                     **rollout_kwargs)
        if self.use_rnn:
            rnn_states_new = {
                k: RNN_State(hidden_states=v[0], cell_states=v[1] if len(v) > 1 else None)
                for k, v in rnn_states_new.items()
            }
        else:
            rnn_states_new = None

        actions = {k: tf.reshape(actions[k], [batch_size, n, -1]).numpy() for k, n in self.n_group_agents.items()}
        actions = AgentGroupedTensor(actions, self.agent_grouping)
        actions_list = [{k: v[i] for k, v in actions.agent_wise.items()} for i in range(batch_size)]

        return MARLActionOutput(
            env_actions=actions_list,
            rnn_states=rnn_states_new
        )
