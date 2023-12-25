Mixiers
=======================================================

In this module, we define components used in multi-agent reinforcement learning (MARL) algorithms for value function approximation and mixing in a cooperative setting. 

These classes are designed to be components within larger multi-agent reinforcement learning systems. 
The mixers, in particular, are responsible for combining the individual agent values into a joint value that represents the collective performance in a cooperative task. 
The use of hypernetworks and feedforward networks in the mixers adds flexibility for capturing complex interactions among agents. 
The QTRAN classes provide a structure for modeling joint action-observation values and joint observation values in a multi-agent setting.

.. raw:: html

    <br><hr>

PyTorch
------------------------------------------

.. py:class::
  xuance.torch.policies.mixers.VDN_mixer()

  This class implements the Value Decomposition Network (VDN) mixer, 
  a simple mixer that aggregates the values of all agents by summing them up.

.. py:function::
  xuance.torch.policies.mixers.VDN_mixer.forward(values_n, states)

  The forward method takes values_n as input, which represents the independent values for each agent,
  and it returns the sum of these values along the specified dimension.

  :param values_n: The independent values of n agents.
  :type values_n: torch.Tensor
  :param states: The global states.
  :type states: torch.Tensor
  :return: The sum of these values along the agent dimension.
  :rtype: torch.Tensor

.. py:class::
  xuance.torch.policies.mixers.QMIX_mixer(dim_state, dim_hidden, dim_hypernet_hidden, n_agents, device)

  - This class implements the QMIX mixer, which is a more sophisticated mixer for cooperative MARL. 
  - It uses hypernetworks to generate per-agent mixing weights, allowing the global value to be a non-linear combination of individual agent values.

  :param dim_state: The dimension of the global states.
  :type dim_state: int
  :param dim_hidden: The dimension of the hidden layers.
  :type dim_hidden: int
  :param dim_hypernet_hidden: The dimension of the hidden layer of the hypyer network.
  :type dim_hypernet_hidden: int
  :param n_agents: The number of agents.
  :type n_agents: int
  :param device: The calculating device.
  :type device: str

.. py:function::
  xuance.torch.policies.mixers.QMIX_mixer.forward(values_n, states)

  The feed forward method that calculates the total team values via mixing networks.

  :param values_n: The independent values of n agents.
  :type values_n: torch.Tensor
  :param states: The global states.
  :type states: torch.Tensor
  :return: The total team values via mixing networks.
  :rtype: torch.Tensor

.. py:class::
  xuance.torch.policies.mixers.QMIX_FF_mixer(dim_state, dim_hidden, n_agents, device)

  This class is another implementation of the QMIX mixer, but it uses a feedforward neural network for mixing instead of hypernetworks.

  :param dim_state: The dimension of the global states.
  :type dim_state: int
  :param dim_hidden: The dimension of the hidden layers.
  :type dim_hidden: int
  :param n_agents: The number of agents.
  :type n_agents: int
  :param device: The calculating device.
  :type device: str

.. py:function::
  xuance.torch.policies.mixers.QMIX_FF_mixer.forward(values_n, states)

  Calculates the total values via a feed forward mixer (QMIX_FF_mixer).

  :param values_n: The independent values of n agents.
  :type values_n: torch.Tensor
  :param states: The global states.
  :type states: torch.Tensor
  :return: The total values.
  :rtype: torch.Tensor

.. py:class::
  xuance.torch.policies.mixers.QTRAN_base(dim_state, dim_action, dim_hidden, n_agents, dim_utility_hidden)

  This is a base class for QTRAN (Quantal Response Transform) mixer. 
  It includes a common structure shared between QTRAN and its alternative version, such as the architecture for computing 
  :math:`Q_{jt}` (joint action-observation value) and :math:`V_{jt}` (joint observation value).

  :param dim_state: The dimension of the global state.
  :type dim_state: int
  :param dim_action: The dimension of the actions.
  :type dim_action: int
  :param dim_hidden: The dimension of the hidden layers.
  :type dim_hidden: int
  :param n_agents: The number of agents.
  :type n_agents: int
  :param dim_utility_hidden: The dimension of the utility hidden states.
  :type dim_utility_hidden: int

.. py:function::
  xuance.torch.policies.mixers.QTRAN_base.forward(hidden_states_n, actions_n)

  Calculates the total values with the QTRAN mixer.

  :param hidden_states_n: The independent hidden states of n agents.
  :type hidden_states_n: int
  :param actions_n: The independent actions of n agents.
  :type actions_n: torch.Tensor
  :return: The evaluated total values of the agents team.
  :rtype: torch.Tensor

.. py:class::
  xuance.torch.policies.mixers.QTRAN_alt(dim_state, dim_action, dim_hidden, n_agents, dim_utility_hidden)

  This class represents an alternative version of QTRAN. 
  It extends the QTRAN_base class and includes methods for computing counterfactual values for self-play scenarios.

  :param dim_state: The dimension of the global state.
  :type dim_state: int
  :param dim_action: The dimension of the action space.
  :type dim_action: int
  :param dim_hidden: The dimension of the hidden layers.
  :type dim_hidden: int
  :param n_agents: The number of agents.
  :type n_agents: int
  :param dim_utility_hidden: The dimension of the utility hidden state.
  :type dim_utility_hidden: int

.. py:function::
  xuance.torch.policies.mixers.QTRAN_alt.counterfactual_values(q_self_values, q_selected_values)

  Calculate the counterfactual Q values given self Q-values and the selected Q-values.

  :param q_self_values: The Q-values of self agents.
  :type q_self_values: torch.Tensor
  :param q_selected_values: The Q-values of selected agents.
  :type q_selected_values: torch.Tensor
  :return: the counterfactual Q values.
  :rtype: torch.Tensor

.. py:function::
  xuance.torch.policies.mixers.QTRAN_alt.counterfactual_values_hat(hidden_states_n, actions_n)

  Calculate the evaluated counterfactual Q values given self Q-values and the selected Q-values.

  :param hidden_states_n: The dimension of the hidden states of n agents.
  :type hidden_states_n: int
  :param actions_n: The independent actions of n agents.
  :type actions_n: torch.Tensor
  :return: The evaluated counterfactual Q values.
  :rtype: torch.Tensor

.. raw:: html

    <br><hr>

TensorFlow
------------------------------------------

.. py:class::
  xuance.tensorflow.policies.mixers.VDN_mixer()

  This class implements the Value Decomposition Network (VDN) mixer, 
  a simple mixer that aggregates the values of all agents by summing them up.

.. py:function::
  xuance.tensorflow.policies.mixers.VDN_mixer.call(values_n)

  The forward method takes values_n as input, which represents the independent values for each agent,
  and it returns the sum of these values along the specified dimension.

  :param values_n: The independent values of n agents.
  :type values_n: tf.Tensor
  :param states: The global states.
  :type states: tf.Tensor
  :return: The sum of these values along the agent dimension.
  :rtype: tf.Tensor

.. py:class::
  xuance.tensorflow.policies.mixers.QMIX_mixer(dim_state, dim_hidden, dim_hypernet_hidden, n_agents, device)

  - This class implements the QMIX mixer, which is a more sophisticated mixer for cooperative MARL. 
  - It uses hypernetworks to generate per-agent mixing weights, allowing the global value to be a non-linear combination of individual agent values.

  :param dim_state: The dimension of the global states.
  :type dim_state: int
  :param dim_hidden: The dimension of the hidden layers.
  :type dim_hidden: int
  :param dim_hypernet_hidden: The dimension of the hidden layer of the hypyer network.
  :type dim_hypernet_hidden: int
  :param n_agents: The number of agents.
  :type n_agents: int
  :param device: The calculating device.
  :type device: str

.. py:function::
  xuance.tensorflow.policies.mixers.QMIX_mixer.call(values_n, states)

  The feed forward method that calculates the total team values via mixing networks.

  :param values_n: The independent values of n agents.
  :type values_n: tf.Tensor
  :param states: The global states.
  :type states: tf.Tensor
  :return: The total team values via mixing networks.
  :rtype: tf.Tensor

.. py:class::
  xuance.tensorflow.policies.mixers.QMIX_FF_mixer(dim_state, dim_hidden, n_agents, device)

  This class is another implementation of the QMIX mixer, but it uses a feedforward neural network for mixing instead of hypernetworks.

  :param dim_state: The dimension of the global state.
  :type dim_state: int
  :param dim_hidden: The dimension of the hidden layers.
  :type dim_hidden: int
  :param n_agents: The number of agents.
  :type n_agents: int
  :param device: The calculating device.
  :type device: str

.. py:function::
  xuance.tensorflow.policies.mixers.QMIX_FF_mixer.call(values_n, states)

  Calculates the total values via a feed forward mixer (QMIX_FF_mixer).

  :param values_n: The independent values of n agents.
  :type values_n: tf.Tensor
  :param states: The global states.
  :type states: tf.Tensor
  :return: The total values.
  :rtype: tf.Tensor

.. py:class::
  xuance.tensorflow.policies.mixers.QTRAN_base(dim_state, dim_action, dim_hidden, n_agents, dim_utility_hidden)

  This is a base class for QTRAN (Quantal Response Transform) mixer. 
  It includes a common structure shared between QTRAN and its alternative version, such as the architecture for computing 
  :math:`Q_{jt}` (joint action-observation value) and :math:`V_{jt}` (joint observation value).

  :param dim_state: The dimension of the global state.
  :type dim_state: int
  :param dim_action: The dimension of the action space.
  :type dim_action: int
  :param dim_hidden: The dimension of the hidden layers.
  :type dim_hidden: int
  :param n_agents: The number of agents.
  :type n_agents: int
  :param dim_utility_hidden: The dimension of the utility hidden state.
  :type dim_utility_hidden: int

.. py:function::
  xuance.tensorflow.policies.mixers.QTRAN_base.call(hidden_states_n, actions_n)

  Calculates the total values with the QTRAN mixer.

  :param hidden_states_n: The independent hidden states of n agents.
  :type hidden_states_n: int
  :param actions_n: The independent actions of n agents.
  :type actions_n: tf.Tensor
  :return: The evaluated total values of the agents team.
  :rtype: tf.Tensor

.. py:class::
  xuance.tensorflow.policies.mixers.QTRAN_alt(dim_state, dim_action, dim_hidden, n_agents, dim_utility_hidden)

  This class represents an alternative version of QTRAN. 
  It extends the QTRAN_base class and includes methods for computing counterfactual values for self-play scenarios.

  :param dim_state: The dimension of the global state.
  :type dim_state: int
  :param dim_action: The dimension of the action space.
  :type dim_action: int
  :param dim_hidden: The dimension of the hidden layers.
  :type dim_hidden: int
  :param n_agents: The number of agents.
  :type n_agents: int
  :param dim_utility_hidden: The dimension of the utility hidden state.
  :type dim_utility_hidden: int

.. py:function::
  xuance.tensorflow.policies.mixers.QTRAN_alt.counterfactual_values(q_self_values, q_selected_values)

  Calculate the counterfactual Q values given self Q-values and the selected Q-values.

  :param q_self_values: The Q-values of self agents.
  :type q_self_values: tf.Tensor
  :param q_selected_values: The Q-values of selected agents.
  :type q_selected_values: tf.Tensor
  :return: the counterfactual Q values.
  :rtype: tf.Tensor

.. py:function::
  xuance.tensorflow.policies.mixers.QTRAN_alt.counterfactual_values_hat(hidden_states_n, actions_n)

  Calculate the evaluated counterfactual Q values given self Q-values and the selected Q-values.

  :param hidden_states_n: The dimension of the hidden states of n agents.
  :type hidden_states_n: int
  :param actions_n: The independent actions of n agents.
  :type actions_n: tf.Tensor
  :return: The evaluated counterfactual Q values.
  :rtype: tf.Tensor

.. raw:: html

    <br><hr>


MindSpore
------------------------------------------

.. py:class::
  xuance.mindspore.policies.mixers.VDN_mixer()

  This class implements the Value Decomposition Network (VDN) mixer, 
  a simple mixer that aggregates the values of all agents by summing them up.

.. py:function::
  xuance.mindspore.policies.mixers.VDN_mixer.construct(values_n, states)

  The forward method takes values_n as input, which represents the independent values for each agent,
  and it returns the sum of these values along the specified dimension.

  :param values_n: The independent values of n agents.
  :type values_n: torch.Tensor
  :param states: The global states.
  :type states: torch.Tensor
  :return: The sum of these values along the agent dimension.
  :rtype: torch.Tensor

.. py:class::
  xuance.mindspore.policies.mixers.QMIX_mixer(dim_state, dim_hidden, dim_hypernet_hidden, n_agents)

  - This class implements the QMIX mixer, which is a more sophisticated mixer for cooperative MARL. 
  - It uses hypernetworks to generate per-agent mixing weights, allowing the global value to be a non-linear combination of individual agent values.

  :param dim_state: The dimension of the global state.
  :type dim_state: int
  :param dim_hidden: The dimension of the hidden layers.
  :type dim_hidden: int
  :param dim_hypernet_hidden: The dimension of hidden states for hyper network.
  :type dim_hypernet_hidden: int
  :param n_agents: The number of agents.
  :type n_agents: int

.. py:function::
  xuance.mindspore.policies.mixers.QMIX_mixer.construct(values_n, states)

  The feed forward method that calculates the total team values via mixing networks.

  :param values_n: The independent values of n agents.
  :type values_n: torch.Tensor
  :param states: The global states.
  :type states: torch.Tensor
  :return: The total team values via mixing networks.
  :rtype: torch.Tensor

.. py:class::
  xuance.mindspore.policies.mixers.QMIX_FF_mixer(dim_state, dim_hidden, n_agents)

  This class is another implementation of the QMIX mixer, but it uses a feedforward neural network for mixing instead of hypernetworks.

  :param dim_state: The dimension of the global state.
  :type dim_state: int
  :param dim_hidden: The dimension of the hidden layers.
  :type dim_hidden: int
  :param n_agents: The number of agents.
  :type n_agents: int

.. py:function::
  xuance.mindspore.policies.mixers.QMIX_FF_mixer.construct(values_n, states)

  Calculates the total values via a feed forward mixer (QMIX_FF_mixer).

  :param values_n: The independent values of n agents.
  :type values_n: torch.Tensor
  :param states: The global states.
  :type states: torch.Tensor
  :return: The total values.
  :rtype: torch.Tensor

.. py:class::
  xuance.mindspore.policies.mixers.QTRAN_base(dim_state, dim_action, dim_hidden, n_agents, dim_utility_hidden)

  This is a base class for QTRAN (Quantal Response Transform) mixer. 
  It includes a common structure shared between QTRAN and its alternative version, such as the architecture for computing 
  :math:`Q_{jt}` (joint action-observation value) and :math:`V_{jt}` (joint observation value).

  :param dim_state: The dimension of the global state.
  :type dim_state: int
  :param dim_action: The dimension of the action space.
  :type dim_action: int
  :param dim_hidden: The dimension of the hidden layers.
  :type dim_hidden: int
  :param n_agents: The number of agents.
  :type n_agents: int
  :param dim_utility_hidden: The dimension of the utility hidden state.
  :type dim_utility_hidden: int

.. py:function::
  xuance.mindspore.policies.mixers.QTRAN_base.construct(hidden_states_n, actions_n)

  Calculates the total values with the QTRAN mixer.

  :param hidden_states_n: The independent hidden states of n agents.
  :type hidden_states_n: int
  :param actions_n: The independent actions of n agents.
  :type actions_n: torch.Tensor
  :return: The evaluated total values of the agents team.
  :rtype: torch.Tensor

.. py:class::
  xuance.mindspore.policies.mixers.QTRAN_alt(dim_state, dim_action, dim_hidden, n_agents, dim_utility_hidden)

  This class represents an alternative version of QTRAN. 
  It extends the QTRAN_base class and includes methods for computing counterfactual values for self-play scenarios.

  :param dim_state: The dimension of the global state.
  :type dim_state: int
  :param dim_action: The dimension of the action space.
  :type dim_action: int
  :param dim_hidden: The dimension of the hidden layers.
  :type dim_hidden: int
  :param n_agents: The number of agents.
  :type n_agents: int
  :param dim_utility_hidden: The dimension of the utility hidden state.
  :type dim_utility_hidden: int

.. py:function::
  xuance.mindspore.policies.mixers.QTRAN_alt.counterfactual_values(q_self_values, q_selected_values)

  Calculate the counterfactual Q values given self Q-values and the selected Q-values.

  :param q_self_values: The Q-values of self agents.
  :type q_self_values: torch.Tensor
  :param q_selected_values: The Q-values of selected agents.
  :type q_selected_values: torch.Tensor
  :return: the counterfactual Q values.
  :rtype: torch.Tensor
.. py:function::
  xuance.mindspore.policies.mixers.QTRAN_alt.counterfactual_values_hat(hidden_states_n, actions_n)

  Calculate the evaluated counterfactual Q values given self Q-values and the selected Q-values.

  :param hidden_states_n: The dimension of the hidden states of n agents.
  :type hidden_states_n: int
  :param actions_n: The independent actions of n agents.
  :type actions_n: torch.Tensor
  :return: The evaluated counterfactual Q values.
  :rtype: torch.Tensor

.. raw:: html

    <br><hr>



Source Code
-----------------

.. tabs::

  .. group-tab:: PyTorch

    .. code-block:: python

        import torch
        import torch.nn as nn
        import torch.nn.functional as F


        class VDN_mixer(nn.Module):
            def __init__(self):
                super(VDN_mixer, self).__init__()

            def forward(self, values_n, states=None):
                return values_n.sum(dim=1)


        class QMIX_mixer(nn.Module):
            def __init__(self, dim_state, dim_hidden, dim_hypernet_hidden, n_agents, device):
                super(QMIX_mixer, self).__init__()
                self.device = device
                self.dim_state = dim_state
                self.dim_hidden = dim_hidden
                self.dim_hypernet_hidden = dim_hypernet_hidden
                self.n_agents = n_agents
                # self.hyper_w_1 = nn.Linear(self.dim_state, self.dim_hidden * self.n_agents)
                # self.hyper_w_2 = nn.Linear(self.dim_state, self.dim_hidden)
                self.hyper_w_1 = nn.Sequential(nn.Linear(self.dim_state, self.dim_hypernet_hidden),
                                               nn.ReLU(),
                                               nn.Linear(self.dim_hypernet_hidden, self.dim_hidden * self.n_agents)).to(device)
                self.hyper_w_2 = nn.Sequential(nn.Linear(self.dim_state, self.dim_hypernet_hidden),
                                               nn.ReLU(),
                                               nn.Linear(self.dim_hypernet_hidden, self.dim_hidden)).to(device)

                self.hyper_b_1 = nn.Linear(self.dim_state, self.dim_hidden).to(device)
                self.hyper_b_2 = nn.Sequential(nn.Linear(self.dim_state, self.dim_hypernet_hidden),
                                               nn.ReLU(),
                                               nn.Linear(self.dim_hypernet_hidden, 1)).to(device)

            def forward(self, values_n, states):
                states = torch.as_tensor(states, dtype=torch.float32, device=self.device)
                states = states.reshape(-1, self.dim_state)
                agent_qs = values_n.reshape(-1, 1, self.n_agents)
                # First layer
                w_1 = torch.abs(self.hyper_w_1(states))
                w_1 = w_1.view(-1, self.n_agents, self.dim_hidden)
                b_1 = self.hyper_b_1(states)
                b_1 = b_1.view(-1, 1, self.dim_hidden)
                hidden = F.elu(torch.bmm(agent_qs, w_1) + b_1)
                # Second layer
                w_2 = torch.abs(self.hyper_w_2(states))
                w_2 = w_2.view(-1, self.dim_hidden, 1)
                b_2 = self.hyper_b_2(states)
                b_2 = b_2.view(-1, 1, 1)
                # Compute final output
                y = torch.bmm(hidden, w_2) + b_2
                # Reshape and return
                q_tot = y.view(-1, 1)
                return q_tot


        class QMIX_FF_mixer(nn.Module):
            def __init__(self, dim_state, dim_hidden, n_agents, device):
                super(QMIX_FF_mixer, self).__init__()
                self.device = device
                self.dim_state = dim_state
                self.dim_hidden = dim_hidden
                self.n_agents = n_agents
                self.dim_input = self.n_agents + self.dim_state
                self.ff_net = nn.Sequential(nn.Linear(self.dim_input, self.dim_hidden),
                                            nn.ReLU(),
                                            nn.Linear(self.dim_hidden, self.dim_hidden),
                                            nn.ReLU(),
                                            nn.Linear(self.dim_hidden, self.dim_hidden),
                                            nn.ReLU(),
                                            nn.Linear(self.dim_hidden, 1)).to(self.device)
                self.ff_net_bias = nn.Sequential(nn.Linear(self.dim_state, self.dim_hidden),
                                                 nn.ReLU(),
                                                 nn.Linear(self.dim_hidden, 1)).to(self.device)

            def forward(self, values_n, states):
                states = states.reshape(-1, self.dim_state)
                agent_qs = values_n.view([-1, self.n_agents])
                inputs = torch.cat([agent_qs, states], dim=-1).to(self.device)
                out_put = self.ff_net(inputs)
                bias = self.ff_net_bias(states)
                y = out_put + bias
                q_tot = y.view([-1, 1])
                return q_tot


        class QTRAN_base(nn.Module):
            def __init__(self, dim_state, dim_action, dim_hidden, n_agents, dim_utility_hidden):
                super(QTRAN_base, self).__init__()
                self.dim_state = dim_state
                self.dim_action = dim_action
                self.dim_hidden = dim_hidden
                self.n_agents = n_agents
                self.dim_q_input = (dim_utility_hidden + self.dim_action) * self.n_agents
                self.dim_v_input = dim_utility_hidden * self.n_agents

                self.Q_jt = nn.Sequential(nn.Linear(self.dim_q_input, self.dim_hidden),
                                          nn.ReLU(),
                                          nn.Linear(self.dim_hidden, self.dim_hidden),
                                          nn.ReLU(),
                                          nn.Linear(self.dim_hidden, 1))
                self.V_jt = nn.Sequential(nn.Linear(self.dim_v_input, self.dim_hidden),
                                          nn.ReLU(),
                                          nn.Linear(self.dim_hidden, self.dim_hidden),
                                          nn.ReLU(),
                                          nn.Linear(self.dim_hidden, 1))

            def forward(self, hidden_states_n, actions_n):
                input_q = torch.cat([hidden_states_n, actions_n], dim=-1).view([-1, self.dim_q_input])
                input_v = hidden_states_n.view([-1, self.dim_v_input])
                q_jt = self.Q_jt(input_q)
                v_jt = self.V_jt(input_v)
                return q_jt, v_jt


        class QTRAN_alt(QTRAN_base):
            def __init__(self, dim_state, dim_action, dim_hidden, n_agents, dim_utility_hidden):
                super(QTRAN_alt, self).__init__(dim_state, dim_action, dim_hidden, n_agents, dim_utility_hidden)

            def counterfactual_values(self, q_self_values, q_selected_values):
                q_repeat = q_selected_values.unsqueeze(dim=1).repeat(1, self.n_agents, 1, self.dim_action)
                counterfactual_values_n = q_repeat
                for agent in range(self.n_agents):
                    counterfactual_values_n[:, agent, agent] = q_self_values[:, agent, :]
                return counterfactual_values_n.sum(dim=2)

            def counterfactual_values_hat(self, hidden_states_n, actions_n):
                action_repeat = actions_n.unsqueeze(dim=2).repeat(1, 1, self.dim_action, 1)
                action_self_all = torch.eye(self.dim_action).unsqueeze(0)
                action_counterfactual_n = action_repeat.unsqueeze(dim=2).repeat(1, 1, self.n_agents, 1, 1)  # batch * N * N * dim_a * dim_a
                q_n = []
                for agent in range(self.n_agents):
                    action_counterfactual_n[:, agent, agent, :, :] = action_self_all
                    q_actions = []
                    for a in range(self.dim_action):
                        input_a = action_counterfactual_n[:, :, agent, a, :]
                        q, _ = self.forward(hidden_states_n, input_a)
                        q_actions.append(q)
                    q_n.append(torch.cat(q_actions, dim=-1).unsqueeze(dim=1))
                return torch.cat(q_n, dim=1)


  .. group-tab:: TensorFlow

    .. code-block:: python

        import tensorflow as tf
        import tensorflow.keras as tk


        class VDN_mixer(tk.Model):
            def __init__(self):
                super(VDN_mixer, self).__init__()

            def call(self, values_n, states=None, **kwargs):
                return tf.reduce_sum(values_n, axis=1)


        class QMIX_mixer(tk.Model):
            def __init__(self, dim_state, dim_hidden, dim_hypernet_hidden, n_agents, device):
                super(QMIX_mixer, self).__init__()
                self.device = device
                self.dim_state = dim_state
                self.dim_hidden = dim_hidden
                self.dim_hypernet_hidden = dim_hypernet_hidden
                self.n_agents = n_agents
                # self.hyper_w_1 = nn.Linear(self.dim_state, self.dim_hidden * self.n_agents)
                # self.hyper_w_2 = nn.Linear(self.dim_state, self.dim_hidden)
                linear_w_1 = [tk.layers.Dense(units=self.dim_hypernet_hidden,
                                              activation=tk.layers.Activation('relu'),
                                              input_shape=(self.dim_state,)),
                              tk.layers.Dense(units=self.dim_hidden * self.n_agents, input_shape=(self.dim_hypernet_hidden,))]
                self.hyper_w_1 = tk.Sequential(linear_w_1)
                linear_w_2 = [tk.layers.Dense(units=self.dim_hypernet_hidden,
                                              activation=tk.layers.Activation('relu'),
                                              input_shape=(self.dim_state,)),
                              tk.layers.Dense(units=self.dim_hidden, input_shape=(self.dim_hypernet_hidden,))]
                self.hyper_w_2 = tk.Sequential(linear_w_2)

                self.hyper_b_1 = tk.layers.Dense(units=self.dim_hidden, input_shape=(self.dim_state,))
                self.hyper_b_2 = tk.Sequential([tk.layers.Dense(units=self.dim_hypernet_hidden,
                                                                activation=tk.layers.Activation('relu'),
                                                                input_shape=(self.dim_state,)),
                                                tk.layers.Dense(units=1, input_shape=(self.dim_hypernet_hidden,))])

            def call(self, values_n, states=None, **kwargs):
                states = tf.reshape(states, [-1, self.dim_state])
                agent_qs = tf.reshape(values_n, [-1, 1, self.n_agents])
                # First layer
                w_1 = tf.abs(self.hyper_w_1(states))
                w_1 = tf.reshape(w_1, [-1, self.n_agents, self.dim_hidden])
                b_1 = self.hyper_b_1(states)
                b_1 = tf.reshape(b_1, [-1, 1, self.dim_hidden])
                hidden = tf.nn.elu(tf.linalg.matmul(agent_qs, w_1) + b_1)
                # Second layer
                w_2 = tf.abs(self.hyper_w_2(states))
                w_2 = tf.reshape(w_2, [-1, self.dim_hidden, 1])
                b_2 = self.hyper_b_2(states)
                b_2 = tf.reshape(b_2, [-1, 1, 1])
                # Compute final output
                y = tf.linalg.matmul(hidden, w_2) + b_2
                # Reshape and return
                q_tot = tf.reshape(y, [-1, 1])
                return q_tot


        class QMIX_FF_mixer(tk.Model):
            def __init__(self, dim_state, dim_hidden, n_agents):
                super(QMIX_FF_mixer, self).__init__()
                self.dim_state = dim_state
                self.dim_hidden = dim_hidden
                self.n_agents = n_agents
                self.dim_input = self.n_agents + self.dim_state
                tk.layers.Dense(input_shape=(self.dim_input,), units=self.dim_hidden, activation=tk.layers.Activation('relu'))
                layers_ff_net = [tk.layers.Dense(input_shape=(self.dim_input,), units=self.dim_hidden,
                                                activation=tk.layers.Activation('relu')),
                                tk.layers.Dense(input_shape=(self.dim_hidden,), units=self.dim_hidden,
                                                activation=tk.layers.Activation('relu')),
                                tk.layers.Dense(input_shape=(self.dim_hidden,), units=self.dim_hidden,
                                                activation=tk.layers.Activation('relu')),
                                tk.layers.Dense(input_shape=(self.dim_hidden,), units=1)]
                self.ff_net = tk.Sequential(layers_ff_net)
                layers_ff_net_bias = [tk.layers.Dense(input_shape=(self.dim_state,), units=self.dim_hidden,
                                                      activation=tk.layers.Activation('relu')),
                                      tk.layers.Dense(input_shape=(self.dim_hidden,), units=1)]
                self.ff_net_bias = tk.Sequential(layers_ff_net_bias)

            def call(self, values_n, states=None, **kwargs):
                states = tf.reshape(states, [-1, self.dim_state])
                agent_qs = tf.reshape(values_n, [-1, self.n_agents])
                inputs = tf.concat([agent_qs, states], axis=-1)
                out_put = self.ff_net(inputs)
                bias = self.ff_net_bias(states)
                y = out_put + bias
                q_tot = tf.reshape(y, [-1, 1])
                return q_tot


        class QTRAN_base(tk.Model):
            def __init__(self, dim_state, dim_action, dim_hidden, n_agents, dim_utility_hidden):
                super(QTRAN_base, self).__init__()
                self.dim_state = dim_state
                self.dim_action = dim_action
                self.dim_hidden = dim_hidden
                self.n_agents = n_agents
                self.dim_q_input = (dim_utility_hidden + self.dim_action) * self.n_agents
                self.dim_v_input = dim_utility_hidden * self.n_agents

                linear_Q_jt = [tk.layers.Dense(input_shape=(self.dim_q_input,), units=self.dim_hidden,
                                              activation=tk.layers.Activation('relu')),
                              tk.layers.Dense(input_shape=(self.dim_hidden,), units=self.dim_hidden,
                                              activation=tk.layers.Activation('relu')),
                              tk.layers.Dense(input_shape=(self.dim_hidden,), units=1)]
                self.Q_jt = tk.Sequential(linear_Q_jt)
                linear_V_jt = [tk.layers.Dense(input_shape=(self.dim_v_input,), units=self.dim_hidden,
                                              activation=tk.layers.Activation('relu')),
                              tk.layers.Dense(input_shape=(self.dim_hidden,), units=self.dim_hidden,
                                              activation=tk.layers.Activation('relu')),
                              tk.layers.Dense(input_shape=(self.dim_hidden,), units=1)]
                self.V_jt = tk.Sequential(linear_V_jt)

            def call(self, hidden_states_n, actions_n=None, **kwargs):
                input_q = tf.reshape(tf.concat([hidden_states_n, actions_n], axis=-1), [-1, self.dim_q_input])
                input_v = tf.reshape(hidden_states_n, [-1, self.dim_v_input])
                q_jt = self.Q_jt(input_q)
                v_jt = self.V_jt(input_v)
                return q_jt, v_jt


        class QTRAN_alt(QTRAN_base):
            def __init__(self, dim_state, dim_action, dim_hidden, n_agents, dim_utility_hidden):
                super(QTRAN_alt, self).__init__(dim_state, dim_action, dim_hidden, n_agents, dim_utility_hidden)

            def counterfactual_values(self, q_self_values, q_selected_values):
                q_repeat = tf.tile(tf.expand_dims(q_selected_values, axis=1), multiples=(1, self.n_agents, 1, self.dim_action))
                counterfactual_values_n = q_repeat.numpy()
                for agent in range(self.n_agents):
                    counterfactual_values_n[:, agent, agent] = q_self_values[:, agent, :].numpy()
                counterfactual_values_n = tf.convert_to_tensor(counterfactual_values_n)
                return tf.reduce_sum(counterfactual_values_n, axis=2)

            def counterfactual_values_hat(self, hidden_states_n, actions_n):
                action_repeat = tf.tile(tf.expand_dims(actions_n, axis=2), multiples=(1, 1, self.dim_action, 1))
                action_self_all = tf.expand_dims(tf.eye(self.dim_action), axis=0).numpy()
                action_counterfactual_n = tf.tile(tf.expand_dims(action_repeat, axis=2), multiples=(
                    1, 1, self.n_agents, 1, 1)).numpy()  # batch * N * N * dim_a * dim_a
                q_n = []
                for agent in range(self.n_agents):
                    action_counterfactual_n[:, agent, agent, :, :] = action_self_all
                    q_actions = []
                    for a in range(self.dim_action):
                        input_a = tf.convert_to_tensor(action_counterfactual_n[:, :, agent, a, :])
                        q, _ = self.call(hidden_states_n, input_a)
                        q_actions.append(q)
                    q_n.append(tf.expand_dims(tf.concat(q_actions, axis=-1), axis=1))
                return tf.concat(q_n, axis=1)



  .. group-tab:: MindSpore

    .. code-block:: python

        import mindspore as ms
        import mindspore.nn as nn


        class VDN_mixer(nn.Cell):
            def __init__(self):
                super(VDN_mixer, self).__init__()
                self._sum = ms.ops.ReduceSum(keep_dims=False)

            def construct(self, values_n, states=None):
                return self._sum(values_n, 1)


        class QMIX_mixer(nn.Cell):
            def __init__(self, dim_state, dim_hidden, dim_hypernet_hidden, n_agents):
                super(QMIX_mixer, self).__init__()
                self.dim_state = dim_state
                self.dim_hidden = dim_hidden
                self.dim_hypernet_hidden = dim_hypernet_hidden
                self.n_agents = n_agents
                # self.hyper_w_1 = nn.Linear(self.dim_state, self.dim_hidden * self.n_agents)
                # self.hyper_w_2 = nn.Linear(self.dim_state, self.dim_hidden)
                self.hyper_w_1 = nn.SequentialCell(nn.Dense(self.dim_state, self.dim_hypernet_hidden),
                                                  nn.ReLU(),
                                                  nn.Dense(self.dim_hypernet_hidden, self.dim_hidden * self.n_agents))
                self.hyper_w_2 = nn.SequentialCell(nn.Dense(self.dim_state, self.dim_hypernet_hidden),
                                                  nn.ReLU(),
                                                  nn.Dense(self.dim_hypernet_hidden, self.dim_hidden))

                self.hyper_b_1 = nn.Dense(self.dim_state, self.dim_hidden)
                self.hyper_b_2 = nn.SequentialCell(nn.Dense(self.dim_state, self.dim_hypernet_hidden),
                                                  nn.ReLU(),
                                                  nn.Dense(self.dim_hypernet_hidden, 1))
                self._abs = ms.ops.Abs()
                self._elu = ms.ops.Elu()

            def construct(self, values_n, states):
                states = states.reshape(-1, self.dim_state)
                agent_qs = values_n.view(-1, 1, self.n_agents)
                # First layer
                w_1 = self._abs(self.hyper_w_1(states))
                w_1 = w_1.view(-1, self.n_agents, self.dim_hidden)
                b_1 = self.hyper_b_1(states)
                b_1 = b_1.view(-1, 1, self.dim_hidden)
                hidden = self._elu(ms.ops.matmul(agent_qs, w_1) + b_1)
                # Second layer
                w_2 = self._abs(self.hyper_w_2(states))
                w_2 = w_2.view(-1, self.dim_hidden, 1)
                b_2 = self.hyper_b_2(states)
                b_2 = b_2.view(-1, 1, 1)
                # Compute final output
                y = ms.ops.matmul(hidden, w_2) + b_2
                # Reshape and return
                q_tot = y.view(-1, 1)
                return q_tot


        class QMIX_FF_mixer(nn.Cell):
            def __init__(self, dim_state, dim_hidden, n_agents):
                super(QMIX_FF_mixer, self).__init__()
                self.dim_state = dim_state
                self.dim_hidden = dim_hidden
                self.n_agents = n_agents
                self.dim_input = self.n_agents + self.dim_state
                self.ff_net = nn.SequentialCell(nn.Dense(self.dim_input, self.dim_hidden),
                                                nn.ReLU(),
                                                nn.Dense(self.dim_hidden, self.dim_hidden),
                                                nn.ReLU(),
                                                nn.Dense(self.dim_hidden, self.dim_hidden),
                                                nn.ReLU(),
                                                nn.Dense(self.dim_hidden, 1))
                self.ff_net_bias = nn.SequentialCell(nn.Dense(self.dim_state, self.dim_hidden),
                                                    nn.ReLU(),
                                                    nn.Dense(self.dim_hidden, 1))
                self._concat = ms.ops.Concat(axis=-1)

            def construct(self, values_n, states):
                states = states.reshape(-1, self.dim_state)
                agent_qs = values_n.view(-1, self.n_agents)
                inputs = self._concat([agent_qs, states])
                out_put = self.ff_net(inputs)
                bias = self.ff_net_bias(states)
                y = out_put + bias
                q_tot = y.view(-1, 1)
                return q_tot


        class QTRAN_base(nn.Cell):
            def __init__(self, dim_state, dim_action, dim_hidden, n_agents, dim_utility_hidden):
                super(QTRAN_base, self).__init__()
                self.dim_state = dim_state
                self.dim_action = dim_action
                self.dim_hidden = dim_hidden
                self.n_agents = n_agents
                self.dim_q_input = (dim_utility_hidden + self.dim_action) * self.n_agents
                self.dim_v_input = dim_utility_hidden * self.n_agents

                self.Q_jt = nn.SequentialCell(nn.Dense(self.dim_q_input, self.dim_hidden),
                                              nn.ReLU(),
                                              nn.Dense(self.dim_hidden, self.dim_hidden),
                                              nn.ReLU(),
                                              nn.Dense(self.dim_hidden, 1))
                self.V_jt = nn.SequentialCell(nn.Dense(self.dim_v_input, self.dim_hidden),
                                              nn.ReLU(),
                                              nn.Dense(self.dim_hidden, self.dim_hidden),
                                              nn.ReLU(),
                                              nn.Dense(self.dim_hidden, 1))
                self._concat = ms.ops.Concat(axis=-1)

            def construct(self, hidden_states_n, actions_n):
                input_q = self._concat([hidden_states_n, actions_n]).view(-1, self.dim_q_input)
                input_v = hidden_states_n.view(-1, self.dim_v_input)
                q_jt = self.Q_jt(input_q)
                v_jt = self.V_jt(input_v)
                return q_jt, v_jt


        class QTRAN_alt(QTRAN_base):
            def __init__(self, dim_state, dim_action, dim_hidden, n_agents, dim_utility_hidden):
                super(QTRAN_alt, self).__init__(dim_state, dim_action, dim_hidden, n_agents, dim_utility_hidden)

            def counterfactual_values(self, q_self_values, q_selected_values):
                q_repeat = ms.ops.broadcast_to(ms.ops.expand_dims(q_selected_values, axis=1),
                                              (-1, self.n_agents, -1, self.dim_action))
                counterfactual_values_n = q_repeat
                for agent in range(self.n_agents):
                    counterfactual_values_n[:, agent, agent] = q_self_values[:, agent, :]
                return counterfactual_values_n.sum(axis=2)

            def counterfactual_values_hat(self, hidden_states_n, actions_n):
                action_repeat = ms.ops.broadcast_to(ms.ops.expand_dims(actions_n, axis=2), (-1, -1, self.dim_action, -1))
                action_self_all = ms.ops.expand_dims(ms.ops.eye(self.dim_action, self.dim_action, ms.float32), axis=0)
                action_counterfactual_n = ms.ops.broadcast_to(ms.ops.expand_dims(action_repeat, axis=2),
                                                              (-1, -1, self.n_agents, -1, -1))  # batch * N * N * dim_a * dim_a

                q_n = []
                for agent in range(self.n_agents):
                    action_counterfactual_n[:, agent, agent, :, :] = action_self_all
                    q_actions = []
                    for a in range(self.dim_action):
                        input_a = action_counterfactual_n[:, :, agent, a, :]
                        q, _ = self.construct(hidden_states_n, input_a)
                        q_actions.append(q)
                    q_n.append(ms.ops.expand_dims(self._concat(q_actions), axis=1))
                return ms.ops.concat(q_n, axis=1)

