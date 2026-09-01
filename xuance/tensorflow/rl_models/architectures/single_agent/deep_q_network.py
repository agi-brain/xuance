from gymnasium.spaces import Space, Discrete
from typing import Type, Sequence, Optional, Union, Tuple
from xuance.tensorflow import tf, keras, Tensor, Module
from xuance.tensorflow.utils import zero_rnn_state_item
from xuance.tensorflow.rl_models.modules import ModelOutput, RNN_State
from xuance.tensorflow.rl_models.heads import (QValueHead, DuelingQValueHead, C51QValueHead,
                                               QuantileRegressionQValueHead, RecurrentQValueHead)


class DeepQNetwork(Module):
    q_head_cls = QValueHead

    def __init__(self,
                 representation: Module,
                 hidden_size: Sequence[int],
                 action_space: Optional[Space] = None,
                 normalizer: Optional[Type[Module]] = None,
                 initializer: Optional[keras.initializers.Initializer] = None,
                 activation: Optional[Type[Module]] = None,
                 use_distributed_training: bool = False,
                 **kwargs):
        super().__init__(**kwargs)
        if isinstance(action_space, Discrete):
            self.n_actions = action_space.n
        else:
            raise ValueError('action_space must be Discrete')
        self.representation = representation
        self.target_representation = representation.clone(copy_weights=True, trainable=False,
                                                          name="target_representation")
        self.representation_info_shape = representation.output_shapes

        self.eval_Q_head = self.q_head_cls(
            feature_dim=self.representation_info_shape['state'][0],
            hidden_size=hidden_size,
            n_actions=self.n_actions,
            normalizer=normalizer,
            initializer=initializer,
            activation=activation,
        )
        self.target_Q_head = self.eval_Q_head.clone(copy_weights=True, trainable=False, name="target_q_head")

    def call(self,
             observation: Union[Tensor, dict],
             **kwargs) -> ModelOutput:
        rep_output = self.representation(observation)
        q_values = self.eval_Q_head(rep_output.embeddings)
        greedy_actions = tf.argmax(q_values, axis=-1, output_type=tf.int32)
        return ModelOutput(
            actions=greedy_actions,
            values=q_values,
            rep_out=rep_output
        )

    def target(self,
               observation: Union[Tensor, dict],
               **kwargs) -> ModelOutput:
        target_rep_output = self.target_representation(observation)
        target_q_values = self.target_Q_head(target_rep_output.embeddings)
        return ModelOutput(values=target_q_values)

    def copy_target(self):
        for ep, tp in zip(self.representation.variables, self.target_representation.variables):
            tp.assign(ep)
        for ep, tp in zip(self.eval_Q_head.variables, self.target_Q_head.variables):
            tp.assign(ep)


class DuelingDeepQNetwork(DeepQNetwork):
    q_head_cls = DuelingQValueHead


class NoisyDeepQNetwork(DeepQNetwork):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.noise_scale = 0.0
        self.eval_noise_parameter = []
        self.target_noise_parameter = []

    def update_noise(self, noisy_bound: float = 0.0):
        """Updates the noises for network parameters."""
        self.eval_noise_parameter = []
        self.target_noise_parameter = []
        for parameter in self.eval_Q_head.variables:
            self.eval_noise_parameter.append(
                tf.random.normal(tf.shape(parameter), dtype=parameter.dtype) * noisy_bound
            )
            self.target_noise_parameter.append(
                tf.random.normal(tf.shape(parameter), dtype=parameter.dtype) * noisy_bound
            )

    def call(self,
             observation: Union[Tensor, dict],
             **kwargs) -> ModelOutput:
        self.update_noise(self.noise_scale)
        for parameter, noise_param in zip(self.eval_Q_head.variables, self.eval_noise_parameter):
            parameter.assign_add(noise_param)
        return super().call(observation, **kwargs)

    def act(self,
            observation: Union[Tensor, dict],
            **kwargs) -> Tensor:
        self.update_noise(self.noise_scale)
        for parameter, noise_param in zip(self.eval_Q_head.variables, self.eval_noise_parameter):
            parameter.assign_add(noise_param)
        return super().act(observation=observation, deterministic=True)

    def target(self,
               observation: Union[Tensor, dict],
               **kwargs) -> ModelOutput:
        self.update_noise(self.noise_scale)
        for parameter, noise_param in zip(self.target_Q_head.variables, self.target_noise_parameter):
            parameter.assign_add(noise_param)
        return super().target(observation, **kwargs)


class C51DeepQNetwork(Module):
    def __init__(self,
                 representation: Module,
                 hidden_size: Sequence[int],
                 action_space: Optional[Space],
                 atom_num: int,
                 v_min: float,
                 v_max: float,
                 normalizer: Optional[Type[Module]] = None,
                 initializer: Optional[keras.initializers.Initializer] = None,
                 activation: Optional[Type[Module]] = None,
                 use_distributed_training: bool = False,
                 **kwargs):
        super().__init__(**kwargs)
        if isinstance(action_space, Discrete):
            self.n_actions = action_space.n
        else:
            raise ValueError('action_space must be Discrete')
        self.representation = representation
        self.target_representation = representation.clone(copy_weights=True, trainable=False,
                                                          name='target_representation')
        self.representation_info_shape = representation.output_shapes

        self.atom_num = atom_num
        self.v_min = v_min
        self.v_max = v_max

        self.eval_Z_head = C51QValueHead(
            feature_dim=self.representation_info_shape['state'][0],
            hidden_size=hidden_size,
            n_actions=self.n_actions,
            atom_num=self.atom_num,
            normalizer=normalizer,
            initializer=initializer,
            activation=activation,
        )
        self.target_Z_head = self.eval_Z_head.clone(copy_weights=True, trainable=False, name='target_Z_head')
        self.supports = self.add_weight(name="supports", shape=(self.atom_num,),
                                        initializer=tf.keras.initializers.Constant(
                                            tf.linspace(self.v_min, self.v_max, self.atom_num).numpy()),
                                        trainable=False, dtype=tf.float32)
        self.delta_z = (v_max - v_min) / (atom_num - 1)

        # Prepare DDP module.
        self.distributed_training = use_distributed_training

    def call(self,
             observation: Union[Tensor, dict],
             **kwargs) -> ModelOutput:
        rep_output = self.representation(observation)
        eval_Z = self.eval_Z_head(rep_output.embeddings)
        eval_Q = tf.reduce_sum(self.supports * eval_Z, axis=-1)
        greedy_actions = tf.argmax(eval_Q, axis=-1, output_type=tf.int32)
        return ModelOutput(
            actions=greedy_actions,
            values=eval_Z,
            rep_out=rep_output
        )

    def act(self,
            observation: Union[Tensor, dict],
            deterministic: bool = True,
            epsilon_greedy: float = 0.0,
            **kwargs) -> Tensor:
        greedy_actions = self.call(observation).actions

        if deterministic or epsilon_greedy <= 0.0:
            actions = greedy_actions
        else:
            random_actions = tf.random.uniform(shape=tf.shape(greedy_actions), minval=0, maxval=self.n_actions,
                                               dtype=greedy_actions.dtype)
            random_mask = tf.random.uniform(shape=tf.shape(greedy_actions), minval=0.0, maxval=1.0,
                                            dtype=tf.float32) < epsilon_greedy
            actions = tf.where(random_mask, random_actions, greedy_actions)
        return actions

    def target(self,
               observation: Union[Tensor, dict],
               **kwargs) -> ModelOutput:
        target_rep_output = self.target_representation(observation)
        target_Z = self.target_Z_head(target_rep_output.embeddings)
        target_Q = tf.reduce_sum(self.supports * target_Z, axis=-1)
        argmax_action = tf.argmax(target_Q, axis=-1, output_type=tf.int32)
        return ModelOutput(actions=argmax_action, values=target_Z)

    def copy_target(self):
        for ep, tp in zip(self.representation.variables, self.target_representation.variables):
            tp.assign(ep)
        for ep, tp in zip(self.eval_Z_head.variables, self.target_Z_head.variables):
            tp.assign(ep)


class QRDeepQNetwork(Module):
    def __init__(self,
                 representation: Module,
                 hidden_size: Sequence[int],
                 action_space: Optional[Space],
                 quantile_num: int,
                 normalizer: Optional[Type[Module]] = None,
                 initializer: Optional[keras.initializers.Initializer] = None,
                 activation: Optional[Type[Module]] = None,
                 use_distributed_training: bool = False,
                 **kwargs):
        super().__init__(**kwargs)
        if isinstance(action_space, Discrete):
            self.n_actions = action_space.n
        else:
            raise ValueError('action_space must be Discrete')
        self.representation = representation
        self.target_representation = representation.clone(copy_weights=True, trainable=False,
                                                          name="target_representation")
        self.representation_info_shape = representation.output_shapes

        self.quantile_num = quantile_num
        self.eval_Z_head = QuantileRegressionQValueHead(
            feature_dim=self.representation_info_shape['state'][0],
            hidden_size=hidden_size,
            n_actions=self.n_actions,
            atom_num=self.quantile_num,
            normalizer=normalizer,
            initializer=initializer,
            activation=activation,
        )
        self.target_Z_head = self.eval_Z_head.clone(copy_weights=True, trainable=False,
                                                    name="target_Z_head")

        # Prepare DDP module.
        self.distributed_training = use_distributed_training

    def call(self,
             observation: Union[Tensor, dict],
             **kwargs) -> ModelOutput:
        rep_output = self.representation(observation)
        eval_Z = self.eval_Z_head(rep_output.embeddings)
        eval_Q = tf.reduce_mean(eval_Z, axis=-1)
        greedy_actions = tf.argmax(eval_Q, axis=-1, output_type=tf.int32)
        return ModelOutput(
            actions=greedy_actions,
            values=eval_Z,
            rep_out=rep_output
        )

    def act(self,
            observation: Union[Tensor, dict],
            deterministic: bool = True,
            epsilon_greedy: float = 0.0,
            **kwargs) -> Tensor:
        greedy_actions = self(observation).actions

        if deterministic or epsilon_greedy <= 0.0:
            actions = greedy_actions
        else:
            random_actions = tf.random.uniform(shape=tf.shape(greedy_actions), minval=0, maxval=self.n_actions,
                                               dtype=greedy_actions.dtype)
            random_mask = tf.random.uniform(shape=tf.shape(greedy_actions), minval=0.0, maxval=1.0,
                                            dtype=tf.float32) < epsilon_greedy
            actions = tf.where(random_mask, random_actions, greedy_actions)
        return actions

    def target(self,
               observation: Union[Tensor, dict],
               **kwargs) -> ModelOutput:
        target_rep_output = self.target_representation(observation)
        target_Z = self.target_Z_head(target_rep_output.embeddings)
        target_Q = tf.reduce_mean(target_Z, axis=-1)
        argmax_action = tf.argmax(target_Q, axis=-1, output_type=tf.int32)
        return ModelOutput(actions=argmax_action, values=target_Z)

    def copy_target(self):
        for ep, tp in zip(self.representation.variables, self.target_representation.variables):
            tp.assign(ep)
        for ep, tp in zip(self.eval_Z_head.variables, self.target_Z_head.variables):
            tp.assign(ep)


class DeepRecurrentQNetwork(Module):
    def __init__(self,
                 representation: Module,
                 recurrent_hidden_size: int,
                 recurrent_layer_N: int,
                 dropout: float,
                 action_space: Optional[Space] = None,
                 rnn: str = 'GRU',
                 initializer: Optional[keras.initializers.Initializer] = None,
                 use_distributed_training: bool = False,
                 **kwargs):
        super().__init__(**kwargs)
        if isinstance(action_space, Discrete):
            self.n_actions = action_space.n
        else:
            raise ValueError('action_space must be Discrete')
        self.representation = representation
        self.target_representation = representation.clone(copy_weights=True, trainable=False,
                                                          name="target_representation")
        self.representation_info_shape = representation.output_shapes

        self.recurrent_layer_N = recurrent_layer_N
        self.recurrent_hidden_size = recurrent_hidden_size

        self.eval_Q_head = RecurrentQValueHead(
            feature_dim=self.representation_info_shape['state'][0],
            recurrent_hidden_size=recurrent_hidden_size,
            recurrent_layer_N=recurrent_layer_N,
            dropout=dropout,
            n_actions=self.n_actions,
            rnn=rnn,
            initializer=initializer,
        )
        self.target_Q_head = self.eval_Q_head.clone(copy_weights=True, trainable=False, name="target_Q_head")

        self.lstm = self.eval_Q_head.lstm

        # Prepare DDP module.
        self.distributed_training = use_distributed_training

    def call(self,
             observation: Union[Tensor, dict],
             rnn_states: RNN_State,
             **kwargs) -> Tuple[RNN_State, ModelOutput]:
        rep_output = self.representation(observation)
        rnn_states_new, q_values = self.eval_Q_head(rep_output.embeddings, rnn_states=rnn_states)
        greedy_actions = tf.argmax(q_values[:, -1], axis=-1)
        return rnn_states_new, ModelOutput(actions=greedy_actions, values=q_values, rep_out=rep_output)

    def act(self,
            observation: Union[Tensor, dict],
            deterministic: bool = True,
            epsilon_greedy: float = 0.0,
            **kwargs) -> Tensor:
        greedy_actions = self.call(observation).actions

        if deterministic or epsilon_greedy <= 0.0:
            actions = greedy_actions
        else:
            random_actions = tf.random.uniform(shape=tf.shape(greedy_actions), minval=0, maxval=self.n_actions,
                                               dtype=greedy_actions.dtype)
            random_mask = tf.random.uniform(shape=tf.shape(greedy_actions), minval=0.0, maxval=1.0,
                                            dtype=tf.float32) < epsilon_greedy
            actions = tf.where(random_mask, random_actions, greedy_actions)
        return actions

    def target(self,
               observation: Union[Tensor, dict],
               rnn_states: RNN_State,
               **kwargs) -> Tuple[RNN_State, ModelOutput]:
        target_rep_output = self.target_representation(observation)
        target_rnn_out, target_q_values = self.target_Q_head(target_rep_output.embeddings, rnn_states=rnn_states)
        argmax_action = tf.argmax(target_q_values, axis=-1)
        return target_rnn_out, ModelOutput(actions=argmax_action, values=target_q_values)

    def init_rnn_states(self, batch: int) -> RNN_State:
        state_shape = (self.recurrent_layer_N, batch, self.recurrent_hidden_size)

        hidden_states = tf.zeros(state_shape, dtype=tf.float32)
        if self.lstm:
            cell_states = tf.zeros(state_shape, dtype=tf.float32)
        else:
            cell_states = None

        return RNN_State(
            hidden_states=hidden_states,
            cell_states=cell_states,
        )

    def init_rnn_states_item(
            self,
            rnn_states: RNN_State,
            i: int | tf.Tensor,
    ) -> RNN_State:
        hidden_states = zero_rnn_state_item(rnn_states.hidden_states, i)

        cell_states = rnn_states.cell_states
        if self.lstm:
            cell_states = zero_rnn_state_item(cell_states, i)

        return RNN_State(hidden_states=hidden_states, cell_states=cell_states)

    def copy_target(self):
        for ep, tp in zip(self.representation.variables, self.target_representation.variables):
            tp.assign(ep)
        for ep, tp in zip(self.eval_Q_head.variables, self.target_Q_head.variables):
            tp.assign(ep)
