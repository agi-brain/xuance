Mixiers
=======================================================

xxxxxx.

.. raw:: html

    <br><hr>

PyTorch
------------------------------------------

.. py:class::
  xuance.torch.policies.mixers.VDN_mixer()

.. py:function::
  xuance.torch.policies.mixers.VDN_mixer.forward(values_n, states)

  xxxxxx.

  :param values_n: The joint values of n agents.
  :type values_n: Tensor
  :param states: xxxxxx.
  :type states: xxxxxx
  :return: xxxxxx.
  :rtype: xxxxxx

.. py:class::
  xuance.torch.policies.mixers.QMIX_mixer(dim_state, dim_hidden, dim_hypernet_hidden, n_agents, device)

  :param dim_state: xxxxxx.
  :type dim_state: xxxxxx
  :param dim_hidden: xxxxxx.
  :type dim_hidden: xxxxxx
  :param dim_hypernet_hidden: xxxxxx.
  :type dim_hypernet_hidden: xxxxxx
  :param n_agents: The number of agents.
  :type n_agents: int
  :param device: The calculating device.
  :type device: str

.. py:function::
  xuance.torch.policies.mixers.QMIX_mixer.forward(values_n, states)

  xxxxxx.

  :param values_n: The joint values of n agents.
  :type values_n: Tensor
  :param states: xxxxxx.
  :type states: xxxxxx
  :return: xxxxxx.
  :rtype: xxxxxx

.. py:class::
  xuance.torch.policies.mixers.QMIX_FF_mixer(dim_state, dim_hidden, n_agents, device)

  :param dim_state: xxxxxx.
  :type dim_state: xxxxxx
  :param dim_hidden: xxxxxx.
  :type dim_hidden: xxxxxx
  :param n_agents: The number of agents.
  :type n_agents: int
  :param device: The calculating device.
  :type device: str

.. py:function::
  xuance.torch.policies.mixers.QMIX_FF_mixer.forward(values_n, states)

  xxxxxx.

  :param values_n: The joint values of n agents.
  :type values_n: Tensor
  :param states: xxxxxx.
  :type states: xxxxxx
  :return: xxxxxx.
  :rtype: xxxxxx

.. py:class::
  xuance.torch.policies.mixers.QTRAN_base(dim_state, dim_action, dim_hidden, n_agents, dim_utility_hidden)

  :param dim_state: xxxxxx.
  :type dim_state: xxxxxx
  :param dim_action: xxxxxx.
  :type dim_action: xxxxxx
  :param dim_hidden: xxxxxx.
  :type dim_hidden: xxxxxx
  :param n_agents: The number of agents.
  :type n_agents: int
  :param dim_utility_hidden: xxxxxx.
  :type dim_utility_hidden: xxxxxx

.. py:function::
  xuance.torch.policies.mixers.QTRAN_base.forward(hidden_states_n, actions_n)

  xxxxxx.

  :param hidden_states_n: xxxxxx.
  :type hidden_states_n: xxxxxx
  :param actions_n: xxxxxx.
  :type actions_n: xxxxxx
  :return: xxxxxx.
  :rtype: xxxxxx

.. py:class::
  xuance.torch.policies.mixers.QTRAN_alt(dim_state, dim_action, dim_hidden, n_agents, dim_utility_hidden)

  :param dim_state: xxxxxx.
  :type dim_state: xxxxxx
  :param dim_action: xxxxxx.
  :type dim_action: xxxxxx
  :param dim_hidden: xxxxxx.
  :type dim_hidden: xxxxxx
  :param n_agents: The number of agents.
  :type n_agents: int
  :param dim_utility_hidden: xxxxxx.
  :type dim_utility_hidden: xxxxxx

.. py:function::
  xuance.torch.policies.mixers.QTRAN_alt.counterfactual_values(q_self_values, q_selected_values)

  xxxxxx.

  :param q_self_values: xxxxxx.
  :type q_self_values: xxxxxx
  :param q_selected_values: xxxxxx.
  :type q_selected_values: xxxxxx
  :return: xxxxxx.
  :rtype: xxxxxx

.. py:function::
  xuance.torch.policies.mixers.QTRAN_alt.counterfactual_values_hat(hidden_states_n, actions_n)

  xxxxxx.

  :param hidden_states_n: xxxxxx.
  :type hidden_states_n: xxxxxx
  :param actions_n: xxxxxx.
  :type actions_n: xxxxxx
  :return: xxxxxx.
  :rtype: xxxxxx

.. raw:: html

    <br><hr>

TensorFlow
------------------------------------------

.. py:class::
  xuance.tensorflow.policies.mixers.VDN_mixer()

.. py:function::
  xuance.tensorflow.policies.mixers.VDN_mixer.call(values_n)

  xxxxxx.

  :param values_n: The joint values of n agents.
  :type values_n: Tensor
  :return: xxxxxx.
  :rtype: xxxxxx

.. py:class::
  xuance.tensorflow.policies.mixers.QMIX_mixer(dim_state, dim_hidden, dim_hypernet_hidden, n_agents, device)

  :param dim_state: xxxxxx.
  :type dim_state: xxxxxx
  :param dim_hidden: xxxxxx.
  :type dim_hidden: xxxxxx
  :param dim_hypernet_hidden: xxxxxx.
  :type dim_hypernet_hidden: xxxxxx
  :param n_agents: The number of agents.
  :type n_agents: int
  :param device: The calculating device.
  :type device: str

.. py:function::
  xuance.tensorflow.policies.mixers.QMIX_mixer.call(values_n, states)

  xxxxxx.

  :param values_n: The joint values of n agents.
  :type values_n: Tensor
  :param states: xxxxxx.
  :type states: xxxxxx
  :return: xxxxxx.
  :rtype: xxxxxx

.. py:class::
  xuance.tensorflow.policies.mixers.QMIX_FF_mixer(dim_state, dim_hidden, n_agents, device)

  :param dim_state: xxxxxx.
  :type dim_state: xxxxxx
  :param dim_hidden: xxxxxx.
  :type dim_hidden: xxxxxx
  :param n_agents: The number of agents.
  :type n_agents: int
  :param device: The calculating device.
  :type device: str

.. py:function::
  xuance.tensorflow.policies.mixers.QMIX_FF_mixer.call(values_n, states)

  xxxxxx.

  :param values_n: The joint values of n agents.
  :type values_n: Tensor
  :param states: xxxxxx.
  :type states: xxxxxx
  :return: xxxxxx.
  :rtype: xxxxxx

.. py:class::
  xuance.tensorflow.policies.mixers.QTRAN_base(dim_state, dim_action, dim_hidden, n_agents, dim_utility_hidden)

  :param dim_state: xxxxxx.
  :type dim_state: xxxxxx
  :param dim_action: xxxxxx.
  :type dim_action: xxxxxx
  :param dim_hidden: xxxxxx.
  :type dim_hidden: xxxxxx
  :param n_agents: The number of agents.
  :type n_agents: int
  :param dim_utility_hidden: xxxxxx.
  :type dim_utility_hidden: xxxxxx

.. py:function::
  xuance.tensorflow.policies.mixers.QTRAN_base.call(hidden_states_n, actions_n)

  xxxxxx.

  :param hidden_states_n: xxxxxx.
  :type hidden_states_n: xxxxxx
  :param actions_n: xxxxxx.
  :type actions_n: xxxxxx
  :return: xxxxxx.
  :rtype: xxxxxx

.. py:class::
  xuance.tensorflow.policies.mixers.QTRAN_alt(dim_state, dim_action, dim_hidden, n_agents, dim_utility_hidden)

  :param dim_state: xxxxxx.
  :type dim_state: xxxxxx
  :param dim_action: xxxxxx.
  :type dim_action: xxxxxx
  :param dim_hidden: xxxxxx.
  :type dim_hidden: xxxxxx
  :param n_agents: The number of agents.
  :type n_agents: int
  :param dim_utility_hidden: xxxxxx.
  :type dim_utility_hidden: xxxxxx

.. py:function::
  xuance.tensorflow.policies.mixers.QTRAN_alt.counterfactual_values(q_self_values, q_selected_values)

  xxxxxx.

  :param q_self_values: xxxxxx.
  :type q_self_values: xxxxxx
  :param q_selected_values: xxxxxx.
  :type q_selected_values: xxxxxx
  :return: xxxxxx.
  :rtype: xxxxxx

.. py:function::
  xuance.tensorflow.policies.mixers.QTRAN_alt.counterfactual_values_hat(hidden_states_n, actions_n)

  xxxxxx.

  :param hidden_states_n: xxxxxx.
  :type hidden_states_n: xxxxxx
  :param actions_n: xxxxxx.
  :type actions_n: xxxxxx
  :return: xxxxxx.
  :rtype: xxxxxx

.. py:class::
  xuance.tensorflow.policies.mixers.DCG_utility(dim_input, dim_hidden, dim_output)

  :param dim_input: xxxxxx.
  :type dim_input: xxxxxx
  :param dim_hidden: xxxxxx.
  :type dim_hidden: xxxxxx
  :param dim_output: xxxxxx.
  :type dim_output: xxxxxx

.. py:function::
  xuance.tensorflow.policies.mixers.DCG_utility.call(hidden_states_n, **kwargs)

  xxxxxx.

  :param hidden_states_n: xxxxxx.
  :type hidden_states_n: xxxxxx
  :param kwargs: xxxxxx.
  :type kwargs: xxxxxx
  :return: xxxxxx.
  :rtype: xxxxxx

.. py:class::
  xuance.tensorflow.policies.mixers.DCG_payoff(dim_input, dim_hidden, dim_act, args)

  :param dim_input: xxxxxx.
  :type dim_input: xxxxxx
  :param dim_hidden: xxxxxx.
  :type dim_hidden: xxxxxx
  :param dim_act: xxxxxx.
  :type dim_act: xxxxxx
  :param args: the arguments.
  :type args: Namespace

.. py:function::
  xuance.tensorflow.policies.mixers.DCG_payoff.call(hidden_from_to, hidden_to_from=None, **kwargs)

  xxxxxx.

  :param hidden_from_to: xxxxxx.
  :type hidden_from_to: xxxxxx
  :param hidden_to_from: xxxxxx.
  :type hidden_to_from: xxxxxx
  :param kwargs: xxxxxx.
  :type kwargs: xxxxxx
  :return: xxxxxx.
  :rtype: xxxxxx

.. py:function::
  xuance.tensorflow.policies.mixers.DCG_payoff.mean_payoffs(payoffs)

  xxxxxx.

  :param payoffs: xxxxxx.
  :type payoffs: xxxxxx
  :return: xxxxxx.
  :rtype: xxxxxx

.. py:class::
  xuance.tensorflow.policies.mixers.Coordination_Graph(n_vertexes, graph_type)

  :param n_vertexes: xxxxxx.
  :type n_vertexes: xxxxxx
  :param graph_type: xxxxxx.
  :type graph_type: xxxxxx

.. py:function::
  xuance.tensorflow.policies.mixers.Coordination_Graph.set_coordination_graph(device)

  xxxxxx.

  :param device: The calculating device.
  :type device: str
  :return: xxxxxx.
  :rtype: xxxxxx

.. raw:: html

    <br><hr>

MindSpore
------------------------------------------

.. py:class::
  xuance.mindspore.policies.mixers.VDN_mixer()

.. py:function::
  xuance.mindspore.policies.mixers.VDN_mixer.construct(values_n, states)

  xxxxxx.

  :param values_n: The joint values of n agents.
  :type values_n: Tensor
  :param states: xxxxxx.
  :type states: xxxxxx
  :return: xxxxxx.
  :rtype: xxxxxx

.. py:class::
  xuance.mindspore.policies.mixers.QMIX_mixer(dim_state, dim_hidden, dim_hypernet_hidden, n_agents)

  :param dim_state: xxxxxx.
  :type dim_state: xxxxxx
  :param dim_hidden: xxxxxx.
  :type dim_hidden: xxxxxx
  :param dim_hypernet_hidden: xxxxxx.
  :type dim_hypernet_hidden: xxxxxx
  :param n_agents: The number of agents.
  :type n_agents: int

.. py:function::
  xuance.mindspore.policies.mixers.QMIX_mixer.construct(values_n, states)

  xxxxxx.

  :param values_n: The joint values of n agents.
  :type values_n: Tensor
  :param states: xxxxxx.
  :type states: xxxxxx
  :return: xxxxxx.
  :rtype: xxxxxx

.. py:class::
  xuance.mindspore.policies.mixers.QMIX_FF_mixer(dim_state, dim_hidden, n_agents)

  :param dim_state: xxxxxx.
  :type dim_state: xxxxxx
  :param dim_hidden: xxxxxx.
  :type dim_hidden: xxxxxx
  :param n_agents: The number of agents.
  :type n_agents: int

.. py:function::
  xuance.mindspore.policies.mixers.QMIX_FF_mixer.construct(values_n, states)

  xxxxxx.

  :param values_n: The joint values of n agents.
  :type values_n: Tensor
  :param states: xxxxxx.
  :type states: xxxxxx
  :return: xxxxxx.
  :rtype: xxxxxx

.. py:class::
  xuance.mindspore.policies.mixers.QTRAN_base(dim_state, dim_action, dim_hidden, n_agents, dim_utility_hidden)

  :param dim_state: xxxxxx.
  :type dim_state: xxxxxx
  :param dim_action: xxxxxx.
  :type dim_action: xxxxxx
  :param dim_hidden: xxxxxx.
  :type dim_hidden: xxxxxx
  :param n_agents: The number of agents.
  :type n_agents: int
  :param dim_utility_hidden: xxxxxx.
  :type dim_utility_hidden: xxxxxx

.. py:function::
  xuance.mindspore.policies.mixers.QTRAN_base.construct(hidden_states_n, actions_n)

  xxxxxx.

  :param hidden_states_n: xxxxxx.
  :type hidden_states_n: xxxxxx
  :param actions_n: xxxxxx.
  :type actions_n: xxxxxx
  :return: xxxxxx.
  :rtype: xxxxxx

.. py:class::
  xuance.mindspore.policies.mixers.QTRAN_alt(dim_state, dim_action, dim_hidden, n_agents, dim_utility_hidden)

  :param dim_state: xxxxxx.
  :type dim_state: xxxxxx
  :param dim_action: xxxxxx.
  :type dim_action: xxxxxx
  :param dim_hidden: xxxxxx.
  :type dim_hidden: xxxxxx
  :param n_agents: The number of agents.
  :type n_agents: int
  :param dim_utility_hidden: xxxxxx.
  :type dim_utility_hidden: xxxxxx

.. py:function::
  xuance.mindspore.policies.mixers.QTRAN_alt.counterfactual_values(q_self_values, q_selected_values)

  xxxxxx.

  :param q_self_values: xxxxxx.
  :type q_self_values: xxxxxx
  :param q_selected_values: xxxxxx.
  :type q_selected_values: xxxxxx
  :return: xxxxxx.
  :rtype: xxxxxx

.. py:function::
  xuance.mindspore.policies.mixers.QTRAN_alt.counterfactual_values_hat(hidden_states_n, actions_n)

  xxxxxx.

  :param hidden_states_n: xxxxxx.
  :type hidden_states_n: xxxxxx
  :param actions_n: xxxxxx.
  :type actions_n: xxxxxx
  :return: xxxxxx.
  :rtype: xxxxxx

.. py:class::
  xuance.mindspore.policies.mixers.DCG_utility(dim_input, dim_hidden, dim_output)

  :param dim_input: xxxxxx.
  :type dim_input: xxxxxx
  :param dim_hidden: xxxxxx.
  :type dim_hidden: xxxxxx
  :param dim_output: xxxxxx.
  :type dim_output: xxxxxx

.. py:function::
  xuance.mindspore.policies.mixers.DCG_utility.construct(hidden_states_n)

  xxxxxx.

  :param hidden_states_n: xxxxxx.
  :type hidden_states_n: xxxxxx
  :return: xxxxxx.
  :rtype: xxxxxx

.. py:class::
  xuance.mindspore.policies.mixers.DCG_payoff(dim_input, dim_hidden, dim_act, args)

  :param dim_input: xxxxxx.
  :type dim_input: xxxxxx
  :param dim_hidden: xxxxxx.
  :type dim_hidden: xxxxxx
  :param dim_act: xxxxxx.
  :type dim_act: xxxxxx
  :param args: the arguments.
  :type args: Namespace

.. py:function::
  xuance.mindspore.policies.mixers.DCG_payoff.construct(hidden_states_n, edges_from, edges_to)

  xxxxxx.

  :param hidden_states_n: xxxxxx.
  :type hidden_states_n: xxxxxx
  :param edges_from: xxxxxx.
  :type edges_from: xxxxxx
  :param edges_to: xxxxxx.
  :type edges_to: xxxxxx
  :return: xxxxxx.
  :rtype: xxxxxx

.. py:class::
  xuance.mindspore.policies.mixers.Coordination_Graph(n_vertexes, graph_type)

  :param n_vertexes: xxxxxx.
  :type n_vertexes: xxxxxx
  :param graph_type: xxxxxx.
  :type graph_type: xxxxxx

.. py:function::
  xuance.mindspore.policies.mixers.Coordination_Graph.set_coordination_graph()

  xxxxxx.

  :return: xxxxxx.
  :rtype: xxxxxx

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
        import numpy as np
        import torch
        import torch_scatter


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


        class DCG_utility(tk.Model):
            def __init__(self, dim_input, dim_hidden, dim_output):
                super(DCG_utility, self).__init__()
                self.dim_input = dim_input
                self.dim_hidden = dim_hidden
                self.dim_output = dim_output
                linears_layers = [tk.layers.Dense(input_shape=(self.dim_input,), units=self.dim_hidden,
                                                  activation=tk.layers.Activation('relu')),
                                  tk.layers.Dense(input_shape=(self.dim_input,), units=self.dim_output)]

                self.model = tk.Sequential(linears_layers)
                # self.output = tk.Sequential(nn.Linear(self.dim_input, self.dim_output))

            def call(self, hidden_states_n, **kwargs):
                return self.model(hidden_states_n)


        class DCG_payoff(DCG_utility):
            def __init__(self, dim_input, dim_hidden, dim_act, args):
                self.dim_act = dim_act
                self.low_rank_payoff = args.low_rank_payoff
                self.payoff_rank = args.payoff_rank
                dim_payoff_out = 2 * self.payoff_rank * self.dim_act if self.low_rank_payoff else self.dim_act ** 2
                super(DCG_payoff, self).__init__(dim_input, dim_hidden, dim_payoff_out)
                self.input_payoff_shape = None

            def call(self, hidden_from_to, hidden_to_from=None, **kwargs):
                # input_payoff = tf.stack([tf.convert_to_tensor(hidden_from_to), tf.convert_to_tensor(hidden_to_from)], axis=0)
                input_payoff = tf.stack([hidden_from_to, hidden_to_from], axis=0)

                self.input_payoff_shape = input_payoff.shape
                return self.model(tf.reshape(input_payoff, [-1, self.input_payoff_shape[-1]]))

            def mean_payoffs(self, payoffs):
                payoffs = tf.reshape(payoffs, self.input_payoff_shape[0:-1] + (self.dim_output,))
                dim = payoffs.shape[0:-1]
                if self.low_rank_payoff:
                    payoffs = tf.reshape(payoffs, [np.prod(dim) * self.payoff_rank, 2, self.dim_act])
                    payoffs_0 = tf.convert_to_tensor(payoffs.numpy()[:, 0, :])
                    payoffs_1 = tf.convert_to_tensor(payoffs.numpy()[:, 1, :])
                    payoffs = tf.matmul(tf.expand_dims(payoffs_0, axis=-1), tf.expand_dims(payoffs_1,
                                                                                           axis=-2))  # (dim_act * 1) * (1 * dim_act) -> (dim_act * dim_act)
                    payoffs = tf.reshape(payoffs, list(dim) + [self.payoff_rank, self.dim_act, self.dim_act])
                    payoffs = tf.reduce_sum(payoffs, axis=-3)
                else:
                    payoffs = tf.reshape(payoffs, list(dim) + [self.dim_act, self.dim_act])

                payoffs = payoffs.numpy()
                dim_num = len(payoffs.shape) - 1
                dim_trans = list(np.arange(dim_num - 2)) + [dim_num - 1, dim_num - 2]
                payoffs[1] = np.transpose(payoffs[1], dim_trans).copy()
                return payoffs.mean(axis=0)  # f^E_{ij} = (f_ij(a_i, a_j) + f_ji(a_j, a_i)) / 2


        class Coordination_Graph(object):
            def __init__(self, n_vertexes, graph_type):
                self.n_vertexes = n_vertexes
                self.edges = []
                if graph_type == "CYCLE":
                    self.edges = [(i, i + 1) for i in range(self.n_vertexes - 1)] + [(self.n_vertexes - 1, 0)]
                elif graph_type == "LINE":
                    self.edges = [(i, i + 1) for i in range(self.n_vertexes - 1)]
                elif graph_type == "STAR":
                    self.edges = [(0, i + 1) for i in range(self.n_vertexes - 1)]
                elif graph_type == "VDN":
                    pass
                elif graph_type == "FULL":
                    self.edges = [[(j, i + j + 1) for i in range(self.n_vertexes - j - 1)] for j in range(self.n_vertexes - 1)]
                    self.edges = [e for l in self.edges for e in l]
                else:
                    raise AttributeError("There is no graph type named {}!".format(graph_type))
                self.n_edges = len(self.edges)
                self.edges_from = None
                self.edges_to = None

            def set_coordination_graph(self, device):
                self.edges_from = torch.zeros(self.n_edges).long()
                self.edges_to = torch.zeros(self.n_edges).long()
                for i, edge in enumerate(self.edges):
                    self.edges_from[i] = edge[0]
                    self.edges_to[i] = edge[1]
                self.edges_n_in = torch_scatter.scatter_add(src=self.edges_to.new_ones(len(self.edges_to)),
                                                            index=self.edges_to, dim=0, dim_size=self.n_vertexes) \
                                  + torch_scatter.scatter_add(src=self.edges_to.new_ones(len(self.edges_to)),
                                                              index=self.edges_from, dim=0, dim_size=self.n_vertexes)
                self.edges_n_in = self.edges_n_in.float().numpy()
                self.edges_from = self.edges_from.numpy()
                self.edges_to = self.edges_to.numpy()
                return


  .. group-tab:: MindSpore

    .. code-block:: python

        import mindspore as ms
        import mindspore.nn as nn
        import torch_scatter
        import torch
        import numpy as np


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


        class DCG_utility(nn.Cell):
            def __init__(self, dim_input, dim_hidden, dim_output):
                super(DCG_utility, self).__init__()
                self.dim_input = dim_input
                self.dim_hidden = dim_hidden
                self.dim_output = dim_output
                self.output = nn.SequentialCell(nn.Dense(int(self.dim_input), int(self.dim_hidden)),
                                                nn.ReLU(),
                                                nn.Dense(int(self.dim_hidden), int(self.dim_output)))
                # self.output = nn.Sequential(nn.Linear(self.dim_input, self.dim_output))

            def construct(self, hidden_states_n):
                return self.output(hidden_states_n)


        class DCG_payoff(DCG_utility):
            def __init__(self, dim_input, dim_hidden, dim_act, args):
                self.dim_act = dim_act
                self.low_rank_payoff = args.low_rank_payoff
                self.payoff_rank = args.payoff_rank
                dim_payoff_out = 2 * self.payoff_rank * self.dim_act if self.low_rank_payoff else self.dim_act ** 2
                super(DCG_payoff, self).__init__(dim_input, dim_hidden, dim_payoff_out)
                self._concat = ms.ops.Concat(axis=-1)
                self.stack = ms.ops.Stack(axis=0)
                self.expand_dims = ms.ops.ExpandDims()
                self.transpose = ms.ops.Transpose()

            def construct(self, hidden_states_n, edges_from=None, edges_to=None):
                input_payoff = self.stack([self._concat([hidden_states_n[:, edges_from], hidden_states_n[:, edges_to]]),
                                           self._concat([hidden_states_n[:, edges_to], hidden_states_n[:, edges_from]])])
                payoffs = self.output(input_payoff)
                dim = payoffs.shape[0:-1]
                if self.low_rank_payoff:
                    payoffs = payoffs.view(np.prod(dim) * self.payoff_rank, 2, self.dim_act)
                    self.expand_dim(payoffs[:, 1, :], -2)
                    payoffs = ms.ops.matmul(self.expand_dim(payoffs[:, 0, :], -1), self.expand_dim(payoffs[:, 1, :], -2))  # (dim_act * 1) * (1 * dim_act) -> (dim_act * dim_act)
                    payoffs = payoffs.view(tuple(list(dim) + [self.payoff_rank, self.dim_act, self.dim_act])).sum(axis=-3)
                else:
                    payoffs = payoffs.view(tuple(list(dim) + [self.dim_act, self.dim_act]))
                payoffs[1] = self.transpose(payoffs[1], (0, 1, 3, 2))  # f_ij(a_i, a_j) <-> f_ji(a_j, a_i)
                return payoffs.mean(axis=0)  # f^E_{ij} = (f_ij(a_i, a_j) + f_ji(a_j, a_i)) / 2


        class Coordination_Graph(nn.Cell):
            def __init__(self, n_vertexes, graph_type):
                super(Coordination_Graph, self).__init__()
                self.n_vertexes = n_vertexes
                self.edges = []
                if graph_type == "CYCLE":
                    self.edges = [(i, i + 1) for i in range(self.n_vertexes - 1)] + [(self.n_vertexes - 1, 0)]
                elif graph_type == "LINE":
                    self.edges = [(i, i + 1) for i in range(self.n_vertexes - 1)]
                elif graph_type == "STAR":
                    self.edges = [(0, i + 1) for i in range(self.n_vertexes - 1)]
                elif graph_type == "VDN":
                    pass
                elif graph_type == "FULL":
                    self.edges = [[(j, i + j + 1) for i in range(self.n_vertexes - j - 1)] for j in range(self.n_vertexes - 1)]
                    self.edges = [e for l in self.edges for e in l]
                else:
                    raise AttributeError("There is no graph type named {}!".format(graph_type))
                self.n_edges = len(self.edges)
                self.edges_from = None
                self.edges_to = None

            def set_coordination_graph(self):
                self.edges_from = torch.zeros(self.n_edges).long()
                self.edges_to = torch.zeros(self.n_edges).long()
                for i, edge in enumerate(self.edges):
                    self.edges_from[i] = edge[0]
                    self.edges_to[i] = edge[1]
                self.edges_n_in = torch_scatter.scatter_add(src=self.edges_to.new_ones(len(self.edges_to)),
                                                            index=self.edges_to, dim=0, dim_size=self.n_vertexes) \
                                  + torch_scatter.scatter_add(src=self.edges_to.new_ones(len(self.edges_to)),
                                                              index=self.edges_from, dim=0, dim_size=self.n_vertexes)
                self.edges_n_in = self.edges_n_in.float()
                # convert to mindspore tensor
                self.edges_from = ms.Tensor(self.edges_from.numpy())
                self.edges_to = ms.Tensor(self.edges_to.numpy())
                self.edges_n_in = ms.Tensor(self.edges_n_in.numpy())
                return
