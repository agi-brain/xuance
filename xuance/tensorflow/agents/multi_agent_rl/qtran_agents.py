from argparse import Namespace
from gymnasium.spaces import Space
from typing import List, Optional, Dict, Tuple
from xuance.common import MultiAgentBaseCallback
from xuance.environment import DummyVecMultiAgentEnv, SubprocVecMultiAgentEnv

from xuance.tensorflow import tf, Tensor, Module, ModuleDict
from xuance.tensorflow.agents import OffPolicyMARLAgents
from xuance.tensorflow.rl_models import DiscreteActionValueCritic
from xuance.tensorflow.rl_models.modules import RNN_State, MARLActionOutput, AgentGroupedTensor
from xuance.tensorflow.rl_models.heads import VDN_Mixer, QTRAN_Base, QTRAN_Alt
from xuance.tensorflow.rl_models.architectures import QTranMixingNetwork


class QTRAN_Agents(OffPolicyMARLAgents):
    """The implementation of QTRAN agents.

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
        super(QTRAN_Agents, self).__init__(
            config, envs, num_agents, agent_keys, state_space, observation_space, action_space, callback
        )
        self.state_space = envs.state_space
        self.use_global_state = True

        self.start_greedy, self.end_greedy = config.start_greedy, config.end_greedy
        self.e_greedy = self.start_greedy
        self.delta_egreedy = (self.start_greedy - self.end_greedy) / (config.decay_step_greedy / self.n_envs)

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
        q_networks = ModuleDict()
        for group_key, group_agents in self.groups.items():
            reference_agent = group_agents[0]
            # build agent feature encoder as representations
            agent_feature_encoder = self._build_agent_feature_encoder(
                representation_choice=self.config.representation,
                group_agents=group_agents,
                input_space=self.observation_space[reference_agent]
            )
            # build inner-group shared q-network
            q_networks[group_key] = DiscreteActionValueCritic(
                representation=agent_feature_encoder,
                action_space=self.action_space[reference_agent],
                critic_hidden_size=self.config.q_hidden_size,
                normalizer=self.normalizer_fn,
                initializer=self.initializer,
                activation=self.activation
            )

        # build mixers
        mixer = VDN_Mixer()

        input_qtran_mixer = dict(
            dim_state=self.state_space.shape[-1],
            action_space=self.action_space,
            dim_hidden=self.config.qtran_net_hidden_dim,
            n_agents=self.config.n_agents,
            dim_utility_hidden=self.config.q_hidden_size[0],
            use_parameter_sharing=self.use_parameter_sharing
        )
        if self.config.agent == "QTRAN_base":
            qtran_mixer = QTRAN_Base(**input_qtran_mixer)
        elif self.config.agent == "QTRAN_alt":
            qtran_mixer = QTRAN_Alt(**input_qtran_mixer)
        else:
            raise ValueError("Mixer {} not recognised.".format(self.config.agent))

        model = QTranMixingNetwork(
            grouping=self.agent_grouping,
            q_networks=q_networks,
            mixer=mixer,
            qtran_mixer=qtran_mixer,
            use_rnn=self.use_rnn,
            use_distributed_training=self.distributed_training
        )

        return model

    @tf.function(reduce_retracing=True)
    def _rollout_step(
            self,
            observations: Dict[str, Tensor],
            agent_indices: Dict[str, Tensor],
            epsilon: Tensor,
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
                for k, v in model_output.rnn_states.items()
            }
        else:
            rnn_states_new = None
        greedy_actions = model_output.actions.grouped_tensor

        # Epsilon-greedy
        actions = {}
        for group in self.group_keys:
            greedy = greedy_actions[group]

            explore_mask = tf.random.uniform(shape=tf.shape(greedy), minval=0.0, maxval=1.0,
                                             dtype=tf.float32) < epsilon
            if self.use_actions_mask:
                available = avail_actions.grouped_tensor[group]
                random_scores = tf.random.uniform(shape=tf.shape(available), minval=0.0, maxval=1.0,
                                                  dtype=tf.float32)
                random_scores = tf.where(available > 0, random_scores, tf.cast(-1.0, random_scores.dtype))
                random_actions = tf.argmax(random_scores, axis=-1, output_type=greedy.dtype)
            else:
                reference_agent = self.groups[group][0]
                n_actions = self.action_space[reference_agent].n
                random_actions = tf.random.uniform(shape=tf.shape(greedy), minval=0, maxval=n_actions,
                                                   dtype=greedy.dtype)
            actions[group] = tf.where(explore_mask, random_actions, greedy)

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
        epsilon = tf.convert_to_tensor(0.0 if test_mode else self.e_greedy, dtype=tf.float32)

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
                                                     epsilon=epsilon,
                                                     **rollout_kwargs)
        if self.use_rnn:
            rnn_states_new = {
                k: RNN_State(hidden_states=v[0], cell_states=v[1] if len(v) > 1 else None)
                for k, v in rnn_states_new.items()
            }
        else:
            rnn_states_new = None

        actions = {k: tf.reshape(actions[k], [batch_size, n]).numpy() for k, n in self.n_group_agents.items()}
        actions = AgentGroupedTensor(actions, self.agent_grouping)
        actions_list = [{k: actions.agent_wise[k][i] for k in self.agent_keys} for i in range(batch_size)]

        return MARLActionOutput(
            env_actions=actions_list,
            rnn_states=rnn_states_new
        )
