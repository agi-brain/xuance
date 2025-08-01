import numpy as np
from copy import deepcopy
from gymnasium.spaces import Box
from xuance.common import Sequence, Optional, Union, Dict, List
from xuance.tensorflow import tf, tk, Module, Tensor
from .core import GaussianActorNet, GaussianActorNet_SAC, CriticNet


class MAAC_Policy(Module):
    """
    MAAC_Policy: Multi-Agent Actor-Critic Policy with Gaussian distributions.

    Args:
        action_space (Box): The continuous action space.
        n_agents (int): The number of agents.
        representation_actor (Optional[Dict[str, Module]]): A dict of representation modules for each agent's actor.
        representation_critic (Optional[Dict[str, Module]]): A dict of representation modules for each agent's critic.
        actor_hidden_size (Sequence[int]): A list of hidden layer sizes for actor network.
        critic_hidden_size (Sequence[int]): A list of hidden layer sizes for critic network.
        normalize (Optional[ModuleType]): The layer normalization over a minibatch of inputs.
        initialize (Optional[tk.initializers.Initializer]): The parameters initializer.
        activation (Optional[tk.layers.Layer]): The activation function for each layer.
        activation_action (Optional[tk.layers.Layer]): The activation of final layer to bound the actions.
        use_distributed_training (bool): Whether to use multi-GPU for distributed training.
        **kwargs: Other arguments.
    """

    def __init__(self,
                 action_space: Optional[Dict[str, Box]],
                 n_agents: int,
                 representation_actor: Optional[Dict[str, Module]],
                 representation_critic: Optional[Dict[str, Module]],
                 mixer: Optional[Module] = None,
                 actor_hidden_size: Sequence[int] = None,
                 critic_hidden_size: Sequence[int] = None,
                 normalize: Optional[tk.layers.Layer] = None,
                 initialize: Optional[tk.initializers.Initializer] = None,
                 activation: Optional[tk.layers.Layer] = None,
                 activation_action: Optional[tk.layers.Layer] = None,
                 use_distributed_training: bool = False,
                 **kwargs):
        super(MAAC_Policy, self).__init__()
        self.is_continuous = True
        self.action_space = action_space
        self.n_agents = n_agents
        self.use_parameter_sharing = kwargs['use_parameter_sharing']
        self.model_keys = kwargs['model_keys']
        self.lstm = True if kwargs["rnn"] == "LSTM" else False
        self.use_rnn = True if kwargs["use_rnn"] else False

        self.actor_representation = representation_actor
        self.critic_representation = representation_critic

        self.dim_input_critic = {}
        self.actor, self.critic = {}, {}
        for key in self.model_keys:
            dim_actor_in, dim_actor_out, dim_critic_in, dim_critic_out = self._get_actor_critic_input(
                self.action_space[key].shape[-1],
                self.actor_representation[key].output_shapes['state'][0],
                self.critic_representation[key].output_shapes['state'][0], n_agents)

            self.actor[key] = GaussianActorNet(dim_actor_in, dim_actor_out, actor_hidden_size,
                                               normalize, initialize, activation, activation_action)
            self.critic[key] = CriticNet(dim_critic_in, critic_hidden_size, normalize, initialize, activation)

        self.mixer = mixer

    def _get_actor_critic_input(self, dim_action, dim_actor_rep, dim_critic_rep, n_agents):
        """
        Returns the input dimensions of actor netwrok and critic networks.

        Parameters:
            dim_action: The dimension of actions (continuous), or the number of actions (discrete).
            dim_actor_rep: The dimension of the output of actor presentation.
            dim_critic_rep: The dimension of the output of critic presentation.
            n_agents: The number of agents.

        Returns:
            dim_actor_in: The dimension of input of the actor networks.
            dim_actor_out: The dimension of output of the actor networks.
            dim_critic_in: The dimension of the input of critic networks.
            dim_critic_out: The dimension of the output of critic networks.
        """
        dim_actor_in, dim_actor_out = dim_actor_rep, dim_action
        dim_critic_in, dim_critic_out = dim_critic_rep, 1
        if self.use_parameter_sharing:
            dim_actor_in += n_agents
            dim_critic_in += n_agents
        return dim_actor_in, dim_actor_out, dim_critic_in, dim_critic_out

    @tf.function
    def call(self, observation: Dict[str, Tensor], agent_ids: Optional[Tensor] = None,
             avail_actions: Dict[str, Tensor] = None, agent_key: str = None,
             rnn_hidden: Optional[Dict[str, List[Tensor]]] = None):
        """
        Returns actions of the policy.

        Parameters:
            observation (Dict[str, Tensor]): The input observations for the policies.
            agent_ids (Tensor): The agents' ids (for parameter sharing).
            avail_actions (Dict[str, Tensor]): Actions mask values, default is None.
            agent_key (str): Calculate actions for specified agent.
            rnn_hidden (Optional[Dict[str, List[Tensor]]]): The RNN hidden states of actor representation.

        Returns:
            rnn_hidden_new (Optional[Dict[str, List[Tensor]]]): The new RNN hidden states of actor representation.
            pi_dists (dict): The stochastic policy distributions.
        """
        rnn_hidden_new, pi_mu, pi_std = deepcopy(rnn_hidden), {}, {}
        agent_list = self.model_keys if agent_key is None else [agent_key]

        for key in agent_list:
            if self.use_rnn:
                outputs = self.actor_representation[key](observation[key], *rnn_hidden[key])
                rnn_hidden_new.update({key: (outputs['rnn_hidden'], outputs['rnn_cell'])})
            else:
                outputs = self.actor_representation[key](observation[key])

            if self.use_parameter_sharing:
                actor_in = tf.concat([outputs['state'], agent_ids], axis=-1)
            else:
                actor_in = outputs['state']
            pi_mu[key], pi_std[key] = self.actor[key](actor_in)

        return rnn_hidden, pi_mu, pi_std

    @tf.function
    def get_values(self, observation: Dict[str, Tensor], agent_ids: Tensor = None, agent_key: str = None,
                   rnn_hidden: Optional[Dict[str, List[Tensor]]] = None):
        """
        Get critic values via critic networks.

        Parameters:
            observation (Dict[str, Tensor]): The input observations for the policies.
            agent_ids (Tensor): The agents' ids (for parameter sharing).
            agent_key (str): Calculate actions for specified agent.
            rnn_hidden (Optional[Dict[str, List[Tensor]]]): The RNN hidden states of critic representation.

        Returns:
            rnn_hidden_new (Optional[Dict[str, List[Tensor]]]): The new RNN hidden states of critic representation.
            values (dict): The evaluated critic values.
        """
        rnn_hidden_new, values = deepcopy(rnn_hidden), {}
        agent_list = self.model_keys if agent_key is None else [agent_key]

        for key in agent_list:
            if self.use_rnn:
                outputs = self.critic_representation[key](observation[key], *rnn_hidden[key])
                rnn_hidden_new.update({key: (outputs['rnn_hidden'], outputs['rnn_cell'])})
            else:
                outputs = self.critic_representation[key](observation[key])

            if self.use_parameter_sharing:
                critic_in = tf.concat([outputs['state'], agent_ids], axis=-1)
            else:
                critic_in = outputs['state']

            values[key] = self.critic[key](critic_in)

        return rnn_hidden_new, values

    @tf.function
    def value_tot(self, values_n: Tensor, global_state=None):
        if global_state is not None:
            global_state = tf.convert_to_tensor(global_state)
        return values_n if self.mixer is None else self.mixer(values_n, global_state)


class Basic_ISAC_Policy(Module):
    """
    Basic_ISAC_Policy: The basic policy for independent soft actor-critic.

    Args:
        action_space (Optional[Dict[str, Box]]): The continuous action space.
        n_agents (int): The number of agents.
        actor_representation (Optional[Dict[str, Module]]): A dict of representation modules for each agent's actor.
        critic_representation (Optional[Dict[str, Module]]): A dict of representation modules for each agent's critic.
        actor_hidden_size (Sequence[int]): A list of hidden layer sizes for actor network.
        critic_hidden_size (Sequence[int]): A list of hidden layer sizes for critic network.
        normalize (Optional[tk.layers.Layer]): The layer normalization over a minibatch of inputs.
        initialize (Optional[tk.initializers.Initializer]): The parameters initializer.
        activation (Optional[tk.layers.Layer]): The activation function for each layer.
        activation_action (Optional[tk.layers.Layer]): The activation of final layer to bound the actions.
        use_distributed_training (bool): Whether to use multi-GPU for distributed training.
        **kwargs: Other arguments.
    """

    def __init__(self,
                 action_space: Optional[Dict[str, Box]],
                 n_agents: int,
                 actor_representation: Optional[Dict[str, Module]],
                 critic_representation: Optional[Dict[str, Module]],
                 actor_hidden_size: Sequence[int],
                 critic_hidden_size: Sequence[int],
                 normalize: Optional[tk.layers.Layer] = None,
                 initialize: Optional[tk.initializers.Initializer] = None,
                 activation: Optional[tk.layers.Layer] = None,
                 activation_action: Optional[tk.layers.Layer] = None,
                 **kwargs):
        super(Basic_ISAC_Policy, self).__init__()
        self.is_continuous = True
        self.action_space = action_space
        self.n_agents = n_agents
        self.use_parameter_sharing = kwargs['use_parameter_sharing']
        self.model_keys = kwargs['model_keys']
        self.lstm = True if kwargs["rnn"] == "LSTM" else False
        self.use_rnn = True if kwargs["use_rnn"] else False

        self.actor_representation = actor_representation
        self.critic_1_representation = critic_representation
        self.critic_2_representation = deepcopy(critic_representation)
        self.target_critic_1_representation = deepcopy(self.critic_1_representation)
        self.target_critic_2_representation = deepcopy(self.critic_2_representation)

        self.actor, self.critic_1, self.critic_2 = {}, {}, {}
        self.target_critic_1, self.target_critic_2 = {}, {}
        self.activation_action = activation_action
        for key in self.model_keys:
            dim_action = self.action_space[key].shape[-1]
            dim_actor_in, dim_actor_out, dim_critic_in = self._get_actor_critic_input(
                self.actor_representation[key].output_shapes['state'][0], dim_action,
                self.critic_1_representation[key].output_shapes['state'][0], n_agents)

            self.actor[key] = GaussianActorNet_SAC(dim_actor_in, dim_actor_out, actor_hidden_size,
                                                   normalize, initialize, activation, activation_action)
            self.critic_1[key] = CriticNet(dim_critic_in, critic_hidden_size, normalize, initialize, activation)
            self.critic_2[key] = CriticNet(dim_critic_in, critic_hidden_size, normalize, initialize, activation)
            self.target_critic_1[key] = CriticNet(dim_critic_in, critic_hidden_size, normalize, initialize, activation)
            self.target_critic_2[key] = CriticNet(dim_critic_in, critic_hidden_size, normalize, initialize, activation)
            self.target_critic_1[key].set_weights(self.critic_1[key].get_weights())
            self.target_critic_2[key].set_weights(self.critic_2[key].get_weights())

    def actor_trainable_variables(self, key):
        return self.actor_representation[key].trainable_variables + self.actor[key].trainable_variables

    def critic_trainable_variables(self, key):
        return self.critic_1_representation[key].trainable_variables + self.critic_1[key].trainable_variables + \
            self.critic_2_representation[key].trainable_variables + self.critic_2[key].trainable_variables

    def _get_actor_critic_input(self, dim_actor_rep, dim_action, dim_critic_rep, n_agents):
        """
        Returns the input dimensions of actor netwrok and critic networks.

        Parameters:
            dim_actor_rep: The dimension of the output of actor presentation.
            dim_action: The dimension of actions (continuous), or the number of actions (discrete).
            dim_critic_rep: The dimension of the output of critic presentation.
            n_agents: The number of agents.

        Returns:
            dim_actor_in: The dimension of input of the actor networks.
            dim_actor_out: The dimension of output of the actor networks.
            dim_critic_in: The dimension of the input of critic networks.
            dim_critic_out: The dimension of the output of critic networks.
        """
        dim_actor_in, dim_actor_out = dim_actor_rep, dim_action
        dim_critic_in = dim_critic_rep + dim_action
        if self.use_parameter_sharing:
            dim_actor_in += n_agents
            dim_critic_in += n_agents
        return dim_actor_in, dim_actor_out, dim_critic_in

    @tf.function
    def call(self, observation: Dict[str, Tensor], agent_ids: Tensor = None,
                avail_actions: Dict[str, Tensor] = None, agent_key: str = None,
                rnn_hidden: Optional[Dict[str, List[Tensor]]] = None):
        """
        Returns actions of the policy.

        Parameters:
            observation (Dict[np.ndarray]): The input observations for the policies.
            agent_ids (np.ndarray): The agents' ids (for parameter sharing).
            agent_key (str): Calculate actions for specified agent.
            rnn_hidden (Optional[Dict[str, List[np.ndarray]]]): The hidden variables of the RNN.

        Returns:
            rnn_hidden_new (Optional[Dict[str, List[Tensor]]]): The new hidden variables of the RNN.
            actions (Dict[Tensor]): The actions output by the policies.
        """
        rnn_hidden_new, actions_dict, log_action_prob = deepcopy(rnn_hidden), {}, {}
        agent_list = self.model_keys if agent_key is None else [agent_key]
        for key in agent_list:
            if self.use_rnn:
                outputs = self.actor_representation[key](observation[key], *rnn_hidden[key])
                rnn_hidden_new.update({key: (outputs['rnn_hidden'], outputs['rnn_cell'])})
            else:
                outputs = self.actor_representation[key](observation[key])

            if self.use_parameter_sharing:
                actor_in = tf.concat([outputs['state'], agent_ids], axis=-1)
            else:
                actor_in = outputs['state']
            pi_mu, pi_std = self.actor[key](actor_in)
            eps = tf.random.normal(shape=tf.shape(pi_mu))  # 𝜖 ~ N(0, 1)
            action_sampled = pi_mu + pi_std * eps  # Reparameterization trick
            actions_dict[key] = self.activation_action(action_sampled)
            # calculate log prob
            log_std = tf.math.log(pi_std + 1e-8)
            log_prob = -0.5 * (((action_sampled - pi_mu) / (pi_std + 1e-8)) ** 2 + 2.0 * log_std + tf.math.log(2.0 * np.pi))
            correction = - 2. * (tf.math.log(2.0) - action_sampled - tk.activations.softplus(-2. * action_sampled))
            log_prob += correction
            log_action_prob[key] = tf.reduce_sum(log_prob, axis=-1)
        return rnn_hidden_new, actions_dict, log_action_prob

    @tf.function
    def Qpolicy(self, observation: Dict[str, Tensor],
                actions: Dict[str, Tensor],
                agent_ids: Tensor = None, agent_key: str = None,
                rnn_hidden_critic_1: Optional[Dict[str, List[Tensor]]] = None,
                rnn_hidden_critic_2: Optional[Dict[str, List[Tensor]]] = None):
        """
        Returns Q^policy of current observations and actions pairs.

        Parameters:
            observation (Dict[Tensor]): The observations.
            actions (Dict[Tensor]): The actions.
            agent_ids (Dict[Tensor]): The agents' ids (for parameter sharing).
            agent_key (str): Calculate actions for specified agent.
            rnn_hidden_critic_1 (Optional[Dict[str, List[Tensor]]]): The RNN hidden states for critic_1 representation.
            rnn_hidden_critic_2 (Optional[Dict[str, List[Tensor]]]): The RNN hidden states for critic_2 representation.

        Returns:
            rnn_hidden_critic_new_1: The updated rnn states for critic_1_representation.
            rnn_hidden_critic_new_2: The updated rnn states for critic_2_representation.
            q_1: The evaluation of Q values with critic 1.
            q_2: The evaluation of Q values with critic 2.
        """
        rnn_hidden_critic_new_1, rnn_hidden_critic_new_2 = deepcopy(rnn_hidden_critic_1), deepcopy(rnn_hidden_critic_2)
        q_1, q_2 = {}, {}
        agent_list = self.model_keys if agent_key is None else [agent_key]

        for key in agent_list:
            if self.use_rnn:
                outputs_critic_1 = self.critic_1_representation[key](observation[key], *rnn_hidden_critic_1[key])
                outputs_critic_2 = self.critic_2_representation[key](observation[key], *rnn_hidden_critic_2[key])
                rnn_hidden_critic_new_1.update({key: (outputs_critic_1['rnn_hidden'], outputs_critic_1['rnn_cell'])})
                rnn_hidden_critic_new_2.update({key: (outputs_critic_2['rnn_hidden'], outputs_critic_2['rnn_cell'])})
            else:
                outputs_critic_1 = self.critic_1_representation[key](observation[key])
                outputs_critic_2 = self.critic_2_representation[key](observation[key])

            critic_1_in = tf.concat([outputs_critic_1['state'], actions[key]], axis=-1)
            critic_2_in = tf.concat([outputs_critic_2['state'], actions[key]], axis=-1)
            if self.use_parameter_sharing:
                critic_1_in = tf.concat([critic_1_in, agent_ids], axis=-1)
                critic_2_in = tf.concat([critic_2_in, agent_ids], axis=-1)
            q_1[key], q_2[key] = self.critic_1[key](critic_1_in), self.critic_2[key](critic_2_in)
        return rnn_hidden_critic_new_1, rnn_hidden_critic_new_2, q_1, q_2

    @tf.function
    def Qtarget(self, next_observation: Dict[str, Tensor],
                next_actions: Dict[str, Tensor],
                agent_ids: Tensor = None, agent_key: str = None,
                rnn_hidden_critic_1: Optional[Dict[str, List[Tensor]]] = None,
                rnn_hidden_critic_2: Optional[Dict[str, List[Tensor]]] = None):
        """
        Returns the Q^target of next observations and actions pairs.

        Parameters:
            next_observation (Dict[Tensor]): The observations of next step.
            next_actions (Dict[Tensor]): The actions of next step.
            agent_ids (Dict[Tensor]): The agents' ids (for parameter sharing).
            agent_key (str): Calculate actions for specified agent.
            rnn_hidden_critic_1 (Optional[Dict[str, List[Tensor]]]): The RNN hidden states for critic_1 representation.
            rnn_hidden_critic_2 (Optional[Dict[str, List[Tensor]]]): The RNN hidden states for critic_2 representation.

        Returns:
            rnn_hidden_critic_new_1: The updated rnn states for critic_1_representation.
            rnn_hidden_critic_new_2: The updated rnn states for critic_2_representation.
            target_q: The evaluations of Q^target.
        """
        rnn_hidden_critic_new_1, rnn_hidden_critic_new_2 = deepcopy(rnn_hidden_critic_1), deepcopy(rnn_hidden_critic_2)
        target_q = {}
        agent_list = self.model_keys if agent_key is None else [agent_key]
        for key in agent_list:
            if self.use_rnn:
                outputs_critic_1 = self.target_critic_1_representation[key](next_observation[key],
                                                                            *rnn_hidden_critic_1[key])
                outputs_critic_2 = self.target_critic_2_representation[key](next_observation[key],
                                                                            *rnn_hidden_critic_2[key])
                rnn_hidden_critic_new_1.update({key: (outputs_critic_1['rnn_hidden'], outputs_critic_1['rnn_cell'])})
                rnn_hidden_critic_new_2.update({key: (outputs_critic_2['rnn_hidden'], outputs_critic_2['rnn_cell'])})
            else:
                outputs_critic_1 = self.target_critic_1_representation[key](next_observation[key])
                outputs_critic_2 = self.target_critic_2_representation[key](next_observation[key])

            critic_1_in = tf.concat([outputs_critic_1['state'], next_actions[key]], axis=-1)
            critic_2_in = tf.concat([outputs_critic_2['state'], next_actions[key]], axis=-1)
            if self.use_parameter_sharing:
                critic_1_in = tf.concat([critic_1_in, agent_ids], axis=-1)
                critic_2_in = tf.concat([critic_2_in, agent_ids], axis=-1)
            target_q_1, target_q_2 = self.target_critic_1[key](critic_1_in), self.target_critic_2[key](critic_2_in)
            target_q[key] = tf.math.minimum(target_q_1, target_q_2)
        return rnn_hidden_critic_new_1, rnn_hidden_critic_new_2, target_q

    @tf.function
    def soft_update(self, tau=0.005):
        for key in self.model_keys:
            for ep, tp in zip(self.critic_1_representation[key].variables,
                              self.target_critic_1_representation[key].variables):
                tp.assign((1 - tau) * tp + tau * ep)
            for ep, tp in zip(self.critic_2_representation[key].variables,
                              self.target_critic_2_representation[key].variables):
                tp.assign((1 - tau) * tp + tau * ep)
            for ep, tp in zip(self.critic_1[key].variables, self.target_critic_1[key].variables):
                tp.assign((1 - tau) * tp + tau * ep)
            for ep, tp in zip(self.critic_2[key].variables, self.target_critic_2[key].variables):
                tp.assign((1 - tau) * tp + tau * ep)


class MASAC_Policy(Basic_ISAC_Policy):
    def __init__(self,
                 action_space: Optional[Dict[str, Box]],
                 n_agents: int,
                 actor_representation: Optional[Dict[str, Module]],
                 critic_representation: Optional[Dict[str, Module]],
                 actor_hidden_size: Sequence[int],
                 critic_hidden_size: Sequence[int],
                 normalize: Optional[tk.layers.Layer] = None,
                 initialize: Optional[tk.initializers.Initializer] = None,
                 activation: Optional[tk.layers.Layer] = None,
                 activation_action: Optional[tk.layers.Layer] = None,
                 **kwargs):
        super(MASAC_Policy, self).__init__(action_space, n_agents, actor_representation, critic_representation,
                                           actor_hidden_size, critic_hidden_size,
                                           normalize, initialize, activation, activation_action, **kwargs)

    def _get_actor_critic_input(self, dim_actor_rep, dim_action, dim_critic_rep, n_agents):
        """
        Returns the input dimensions of actor netwrok and critic networks.

        Parameters:
            dim_actor_rep: The dimension of the output of actor presentation.
            dim_action: The dimension of actions (continuous), or the number of actions (discrete).
            dim_critic_rep: The dimension of the output of critic presentation.
            n_agents: The number of agents.

        Returns:
            dim_actor_in: The dimension of input of the actor networks.
            dim_actor_out: The dimension of output of the actor networks.
            dim_critic_in: The dimension of the input of critic networks.
            dim_critic_out: The dimension of the output of critic networks.
        """
        dim_actor_in, dim_actor_out = dim_actor_rep, dim_action
        dim_critic_in = dim_critic_rep
        if self.use_parameter_sharing:
            dim_actor_in += n_agents
            dim_critic_in += n_agents
        return dim_actor_in, dim_actor_out, dim_critic_in

    @tf.function
    def Qpolicy(self, joint_observation: Optional[np.ndarray] = None,
                joint_actions: Optional[np.ndarray] = None,
                agent_ids: np.ndarray = None, agent_key: str = None,
                rnn_hidden_critic_1: Optional[Dict[str, List[np.ndarray]]] = None,
                rnn_hidden_critic_2: Optional[Dict[str, List[np.ndarray]]] = None):
        """
        Returns Q^policy of current observations and actions pairs.

        Parameters:
            joint_observation (Optional[np.ndarray]): The joint observations of the team.
            joint_actions (Optional[np.ndarray]): The joint actions of the team.
            agent_ids (Dict[np.ndarray]): The agents' ids (for parameter sharing).
            agent_key (str): Calculate actions for specified agent.
            rnn_hidden_critic_1 (Optional[Dict[str, List[np.ndarray]]]): The RNN hidden for critic_1 representation.
            rnn_hidden_critic_2 (Optional[Dict[str, List[np.ndarray]]]): The RNN hidden for critic_2 representation.

        Returns:
            rnn_hidden_critic_new_1: The updated rnn states for critic_1_representation.
            rnn_hidden_critic_new_2: The updated rnn states for critic_2_representation.
            q_1: The evaluations of Q^policy with critic 1.
            q_2: The evaluations of Q^policy with critic 2.
        """
        rnn_hidden_critic_new_1, rnn_hidden_critic_new_2 = deepcopy(rnn_hidden_critic_1), deepcopy(rnn_hidden_critic_2)
        q_1, q_2 = {}, {}
        agent_list = self.model_keys if agent_key is None else [agent_key]
        batch_size = joint_observation.shape[0]
        seq_len = joint_observation.shape[1] if self.use_rnn else 1

        critic_rep_in = tf.concat([joint_observation, joint_actions], axis=-1)
        if self.use_rnn:
            outputs_critic_1 = {k: self.critic_1_representation[k](critic_rep_in, *rnn_hidden_critic_1[k])
                                for k in agent_list}
            outputs_critic_2 = {k: self.critic_2_representation[k](critic_rep_in, *rnn_hidden_critic_2[k])
                                for k in agent_list}
            rnn_hidden_critic_new_1.update({k: (outputs_critic_1[k]['rnn_hidden'], outputs_critic_1[k]['rnn_cell'])
                                            for k in agent_list})
            rnn_hidden_critic_new_2.update({k: (outputs_critic_2[k]['rnn_hidden'], outputs_critic_2[k]['rnn_cell'])
                                            for k in agent_list})
        else:
            outputs_critic_1 = {k: self.critic_1_representation[k](critic_rep_in) for k in agent_list}
            outputs_critic_2 = {k: self.critic_2_representation[k](critic_rep_in) for k in agent_list}

        bs = batch_size * self.n_agents if self.use_parameter_sharing else batch_size

        for key in agent_list:
            if self.use_parameter_sharing:
                joint_rep_out_1 = tf.repeat(tf.expand_dims(outputs_critic_1[key]['state'], 1), self.n_agents, 1)
                joint_rep_out_2 = tf.repeat(tf.expand_dims(outputs_critic_2[key]['state'], 1), self.n_agents, 1)
                if self.use_rnn:
                    joint_rep_out_1 = tf.reshape(joint_rep_out_1, [bs, seq_len, -1])
                    joint_rep_out_2 = tf.reshape(joint_rep_out_2, [bs, seq_len, -1])
                else:
                    joint_rep_out_1 = tf.reshape(joint_rep_out_1, [bs, -1])
                    joint_rep_out_2 = tf.reshape(joint_rep_out_2, [bs, -1])
                critic_1_in = tf.concat([joint_rep_out_1, agent_ids], axis=-1)
                critic_2_in = tf.concat([joint_rep_out_2, agent_ids], axis=-1)
            else:
                if self.use_rnn:
                    joint_rep_out_1 = tf.reshape(outputs_critic_1[key]['state'], [bs, seq_len, -1])
                    joint_rep_out_2 = tf.reshape(outputs_critic_2[key]['state'], [bs, seq_len, -1])
                else:
                    joint_rep_out_1 = tf.reshape(outputs_critic_1[key]['state'], [bs, -1])
                    joint_rep_out_2 = tf.reshape(outputs_critic_2[key]['state'], [bs, -1])
                critic_1_in = joint_rep_out_1
                critic_2_in = joint_rep_out_2
            q_1[key] = self.critic_1[key](critic_1_in)
            q_2[key] = self.critic_2[key](critic_2_in)

        return rnn_hidden_critic_new_1, rnn_hidden_critic_new_2, q_1, q_2

    @tf.function
    def Qtarget(self, joint_observation: Optional[np.ndarray] = None,
                joint_actions: Optional[np.ndarray] = None,
                agent_ids: np.ndarray = None, agent_key: str = None,
                rnn_hidden_critic_1: Optional[Dict[str, List[np.ndarray]]] = None,
                rnn_hidden_critic_2: Optional[Dict[str, List[np.ndarray]]] = None):
        """
        Returns the Q^target of next observations and actions pairs.

        Parameters:
            joint_observation (Optional[np.ndarray]): The joint observations of the team.
            joint_actions (Optional[np.ndarray]): The joint actions of the team.
            agent_ids (Dict[np.ndarray]): The agents' ids (for parameter sharing).
            agent_key (str): Calculate actions for specified agent.
            rnn_hidden_critic_1 (Optional[Dict[str, List[np.ndarray]]]): The RNN hidden for critic_1 representation.
            rnn_hidden_critic_2 (Optional[Dict[str, List[np.ndarray]]]): The RNN hidden for critic_2 representation.

        Returns:
            rnn_hidden_critic_new_1: The updated rnn states for critic_1_representation.
            rnn_hidden_critic_new_2: The updated rnn states for critic_2_representation.
            q_target: The evaluations of Q^target.
        """
        rnn_hidden_critic_new_1, rnn_hidden_critic_new_2 = deepcopy(rnn_hidden_critic_1), deepcopy(rnn_hidden_critic_2)
        target_q = {}
        agent_list = self.model_keys if agent_key is None else [agent_key]
        batch_size = joint_observation.shape[0]
        seq_len = joint_observation.shape[1] if self.use_rnn else 1

        critic_rep_in = tf.concat([joint_observation, joint_actions], axis=-1)
        if self.use_rnn:
            outputs_critic_1 = {k: self.target_critic_1_representation[k](critic_rep_in, *rnn_hidden_critic_1[k])
                                for k in agent_list}
            outputs_critic_2 = {k: self.target_critic_2_representation[k](critic_rep_in, *rnn_hidden_critic_2[k])
                                for k in agent_list}
            rnn_hidden_critic_new_1.update({k: (outputs_critic_1[k]['rnn_hidden'], outputs_critic_1[k]['rnn_cell'])
                                            for k in agent_list})
            rnn_hidden_critic_new_2.update({k: (outputs_critic_2[k]['rnn_hidden'], outputs_critic_2[k]['rnn_cell'])
                                            for k in agent_list})
        else:
            outputs_critic_1 = {k: self.target_critic_1_representation[k](critic_rep_in) for k in agent_list}
            outputs_critic_2 = {k: self.target_critic_2_representation[k](critic_rep_in) for k in agent_list}

        bs = batch_size * self.n_agents if self.use_parameter_sharing else batch_size

        for key in agent_list:
            if self.use_parameter_sharing:
                joint_rep_out_1 = tf.repeat(tf.expand_dims(outputs_critic_1[key]['state'], 1), self.n_agents, 1)
                joint_rep_out_2 = tf.repeat(tf.expand_dims(outputs_critic_2[key]['state'], 1), self.n_agents, 1)
                if self.use_rnn:
                    joint_rep_out_1 = tf.reshape(joint_rep_out_1, [bs, seq_len, -1])
                    joint_rep_out_2 = tf.reshape(joint_rep_out_2, [bs, seq_len, -1])
                else:
                    joint_rep_out_1 = tf.reshape(joint_rep_out_1, [bs, -1])
                    joint_rep_out_2 = tf.reshape(joint_rep_out_2, [bs, -1])
                critic_1_in = tf.concat([joint_rep_out_1, agent_ids], axis=-1)
                critic_2_in = tf.concat([joint_rep_out_2, agent_ids], axis=-1)
            else:
                if self.use_rnn:
                    joint_rep_out_1 = tf.reshape(outputs_critic_1[key]['state'], [bs, seq_len, -1])
                    joint_rep_out_2 = tf.reshape(outputs_critic_2[key]['state'], [bs, seq_len, -1])
                else:
                    joint_rep_out_1 = tf.reshape(outputs_critic_1[key]['state'], [bs, -1])
                    joint_rep_out_2 = tf.reshape(outputs_critic_2[key]['state'], [bs, -1])
                critic_1_in = joint_rep_out_1
                critic_2_in = joint_rep_out_2
            q_1 = self.target_critic_1[key](critic_1_in)
            q_2 = self.target_critic_2[key](critic_2_in)
            target_q[key] = tf.math.minimum(q_1, q_2)
        return rnn_hidden_critic_new_1, rnn_hidden_critic_new_2, target_q
