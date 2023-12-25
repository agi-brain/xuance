DCG_Learner
=====================================

A DCG (Deep Coordination Graphs) learner for multi-agent reinforcement learning.

.. raw:: html

    <br><hr>

PyTorch
------------------------------------------

.. py:class::
  xuance.torch.learners.multi_agent_rl.dcg_learner.DCG_Learner(config, policy, optimizer, scheduler, device, model_dir, gamma, sync_frequency)

  :param config: Provides hyper parameters.
  :type config: Namespace
  :param policy: The policy that provides actions and values.
  :type policy: nn.Module
  :param optimizer: The optimizer that update the parameters of the model.
  :type optimizer: Optimizer
  :param scheduler: The tool for learning rate decay.
  :type scheduler: lr_scheduler
  :param device: The calculating device.
  :type device: str
  :param model_dir: The directory for saving or loading the model parameters.
  :type model_dir: str
  :param gamma: The discount factor.
  :type gamma: float
  :param sync_frequency: The frequency to synchronize the target networks.
  :type sync_frequency: int

.. py:function::
  xuance.torch.learners.multi_agent_rl.dcg_learner.DCG_Learner.get_hidden_states(obs_n, *rnn_hidden, use_target_net)

  Retrieves hidden states from the policy network.

  :param obs_n: The joint observations of n agents.
  :type obs_n: np.ndarray
  :param rnn_hidden: The last final hidden states of the sequence.
  :type rnn_hidden: Tensor
  :param use_target_net: Flag indicating whether to use the target network.
  :type use_target_net: bool
  :return: hidden states.
  :rtype: Tensor

.. py:function::
  xuance.torch.learners.multi_agent_rl.dcg_learner.DCG_Learner.get_graph_values(hidden_states, use_target_net)

  Retrieves graph values from the policy network.

  :param hidden_states: Hidden states obtained from the policy network.
  :type hidden_states: Tensor
  :param use_target_net: Flag indicating whether to use the target network.
  :type use_target_net: bool
  :return: The graph values.
  :rtype: np.ndarray, Tensor

.. py:function::
  xuance.torch.learners.multi_agent_rl.dcg_learner.DCG_Learner.act(hidden_states, avail_actions)

  Selects joint actions based on the provided hidden states and optional availability mask.

  :param hidden_states: Hidden states obtained from the policy network.
  :type hidden_states: Tensor
  :param avail_actions: The mask varibales for availabel actions.
  :type avail_actions: Tensor
  :return: Joint actions.
  :rtype: np.ndarray, Tensor

.. py:function::
  xuance.torch.learners.multi_agent_rl.dcg_learner.DCG_Learner.q_dcg(hidden_states, actions, states, use_target_net)

  Calculates the Q-values based on the provided hidden states and actions.

  :param hidden_states: Hidden states obtained from the policy network.
  :type hidden_states: tensor
  :param actions: The actions input.
  :type actions: Tensor
  :param states: Optional states input.
  :type states: Tensor
  :param use_target_net: Indicates whether to use the target network for calculation.
  :type use_target_net: bool
  :return: Q-values.
  :rtype: tensor

.. py:function::
  xuance.torch.learners.multi_agent_rl.dcg_learner.DCG_Learner.update(sample)

  Update the DCG learner using the provided sample.

  :param sample: A dictionary containing the states, observations, actions, next states, next observations, rewards,
                    terminal flags indicating the end of episodes, binary agent mask.
  :type sample: dict
  :return: The information of the training.
  :rtype: dict

.. py:function::
  xuance.torch.learners.multi_agent_rl.dcg_learner.DCG_Learner.update_recurrent(sample)

  Updates the DCG learner using recurrent data.

  :param sample: recurrent training data.
  :type sample: dict
  :return: The information of the training.
  :rtype: dict

.. raw:: html

    <br><hr>

TensorFlow
------------------------------------------

.. py:class::
  xuance.tensorflow.learners.multi_agent_rl.dcg_learner.DCG_Learner(config, policy, optimizer, device, model_dir, gamma, sync_frequency)

  :param config: Provides hyper parameters.
  :type config: Namespace
  :param policy: The policy that provides actions and values.
  :type policy: nn.Module
  :param optimizer: The optimizer that update the parameters of the model.
  :type optimizer: Optimizer
  :param device: The calculating device.
  :type device: str
  :param model_dir: The directory for saving or loading the model parameters.
  :type model_dir: str
  :param gamma: The discount factor.
  :type gamma: float
  :param sync_frequency: The frequency to synchronize the target networks.
  :type sync_frequency: int

.. py:function::
  xuance.tensorflow.learners.multi_agent_rl.dcg_learner.DCG_Learner.get_hidden_states(obs_n, *rnn_hidden, use_target_net)

  Retrieves hidden states from the policy network.

  :param obs_n: The joint observations of n agents.
  :type obs_n: np.ndarray
  :param rnn_hidden: The last final hidden states of the sequence.
  :type rnn_hidden: Tensor
  :param use_target_net: Indicates whether to use the target network for calculation.
  :type use_target_net: bool
  :return: The hidden states.
  :rtype: np.ndarray, tf.Tensor

.. py:function::
  xuance.tensorflow.learners.multi_agent_rl.dcg_learner.DCG_Learner.get_graph_values(hidden_states, use_target_net)

  Retrieves graph values from the policy network.

  :param hidden_states: Hidden states obtained from the policy network.
  :type hidden_states: Tensor
  :param use_target_net: Flag indicating whether to use the target network.
  :type use_target_net: bool
  :return: The graph values.
  :rtype: np.ndarray, tf.Tensor

.. py:function::
  xuance.tensorflow.learners.multi_agent_rl.dcg_learner.DCG_Learner.act(hidden_states, avail_actions)

  Selects joint actions based on the provided hidden states and optional availability mask.

  :param hidden_states: Hidden states obtained from the policy network.
  :type hidden_states: Tensor
  :param avail_actions: The mask varibales for availabel actions.
  :type avail_actions: Tensor
  :return: The joint actions.
  :rtype: np.ndarray, tf.Tensor

.. py:function::
  xuance.tensorflow.learners.multi_agent_rl.dcg_learner.DCG_Learner.q_dcg(hidden_states, actions, states, use_target_net)

  Compute Q-values for the DCG (Deep Coordination Graphs) learner.

  :param hidden_states: Hidden states of the agents.
  :type hidden_states: np.ndarray, tf.Tensor
  :param actions: The actions input.
  :type actions: Tensor
  :param states: States information (optional).
  :type states: Tensor
  :param use_target_net: Flag indicating whether to use the target network.
  :type use_target_net: bool
  :return: The Q values of the graph.
  :rtype: np.ndarray, tf.Tensor

.. py:function::
  xuance.tensorflow.learners.multi_agent_rl.dcg_learner.DCG_Learner.update(sample)

  Update the learner based on a training sample.

  :param sample: Training sample.
  :type sample: dict
  :return: The information of the training.
  :rtype: dict

.. raw:: html

    <br><hr>

MindSpore
------------------------------------------

.. py:class::
  xuance.mindspore.learners.multi_agent_rl.dcg_learner.DCG_Learner(config, policy, optimizer, scheduler, model_dir, gamma, sync_frequency)

  :param config: Provides hyper parameters.
  :type config: Namespace
  :param policy: The policy that provides actions and values.
  :type policy: nn.Module
  :param optimizer: The optimizer that update the parameters of the model.
  :type optimizer: Optimizer
  :param scheduler: The tool for learning rate decay.
  :type scheduler: lr_scheduler
  :param model_dir: The directory for saving or loading the model parameters.
  :type model_dir: str
  :param gamma: The discount factor.
  :type gamma: float
  :param sync_frequency: The frequency to synchronize the target networks.
  :type sync_frequency: int

.. py:function::
  xuance.mindspore.learners.multi_agent_rl.dcg_learner.DCG_Learner.get_hidden_states(obs_n, *rnn_hidden, use_target_net)

  Retrieves hidden states from the policy network.

  :param obs_n: The joint observations of n agents.
  :type obs_n: np.ndarray
  :param rnn_hidden: The last final hidden states of the sequence.
  :type rnn_hidden: Tensor
  :param use_target_net: Flag indicating whether to use the target network.
  :type use_target_net: bool
  :return: The hidden states.
  :rtype: np.ndarray, ms.Tensor

.. py:function::
  xuance.mindspore.learners.multi_agent_rl.dcg_learner.DCG_Learner.get_graph_values(hidden_states, use_target_net)

  Retrieves graph values from the policy network.

  :param hidden_states: Hidden states obtained from the policy network.
  :type hidden_states: Tensor
  :param use_target_net: Flag indicating whether to use the target network.
  :type use_target_net: bool
  :return: The graph values.
  :rtype: np.ndarray, ms.Tensor

.. py:function::
  xuance.mindspore.learners.multi_agent_rl.dcg_learner.DCG_Learner.act(hidden_states, avail_actions)

  Selects joint actions based on the provided hidden states and optional availability mask.

  :param hidden_states: Hidden states obtained from the policy network.
  :type hidden_states: Tensor
  :param avail_actions: The mask varibales for availabel actions.
  :type avail_actions: Tensor
  :return: The joint actions.
  :rtype: np.ndarray, ms.Tensor

.. py:function::
  xuance.mindspore.learners.multi_agent_rl.dcg_learner.DCG_Learner.q_dcg(hidden_states, actions, states, use_target_net)

  Calculate the Q values for the Deep Coordination Graph (DCG) learner.

  :param hidden_states: The hidden states of the learner.
  :type hidden_states: Tensor
  :param actions: The actions input.
  :type actions: Tensor
  :param states: The states.
  :type states: Tensor
  :param use_target_net: A flag indicating whether to use the target network.
  :type use_target_net: bool
  :return: The Q values of the graph.
  :rtype: np.ndarray, ms.Tensor

.. py:function::
  xuance.mindspore.learners.multi_agent_rl.dcg_learner.DCG_Learner.update(sample)

  Update the DCG learner based on the given sample.

  :param sample: The training sample.
  :type sample: dict
  :return: The information of the training.
  :rtype: dict

.. raw:: html

    <br><hr>

Source Code
-----------------

.. tabs::

  .. group-tab:: PyTorch

    .. code-block:: python

        """
        DCG: Deep coordination graphs
        Paper link: http://proceedings.mlr.press/v119/boehmer20a/boehmer20a.pdf
        Implementation: Pytorch
        """
        from xuance.torch.learners import *
        import torch_scatter


        class DCG_Learner(LearnerMAS):
            def __init__(self,
                         config: Namespace,
                         policy: nn.Module,
                         optimizer: torch.optim.Optimizer,
                         scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
                         device: Optional[Union[int, str, torch.device]] = None,
                         model_dir: str = "./",
                         gamma: float = 0.99,
                         sync_frequency: int = 100
                         ):
                self.gamma = gamma
                self.use_recurrent = config.use_recurrent
                self.sync_frequency = sync_frequency
                self.dim_hidden_state = policy.representation.output_shapes['state'][0]
                self.mse_loss = nn.MSELoss()
                super(DCG_Learner, self).__init__(config, policy, optimizer, scheduler, device, model_dir)

            def get_hidden_states(self, obs_n, *rnn_hidden, use_target_net=False):
                if self.use_recurrent:
                    if use_target_net:
                        outputs = self.policy.target_representation(obs_n, *rnn_hidden)
                    else:
                        outputs = self.policy.representation(obs_n, *rnn_hidden)
                    hidden_states = outputs['state']
                    rnn_hidden = (outputs['rnn_hidden'], outputs['rnn_cell'])
                else:
                    if use_target_net:
                        hidden_states = self.policy.target_representation(obs_n)['state']
                    else:
                        hidden_states = self.policy.representation(obs_n)['state']
                    rnn_hidden = None
                return rnn_hidden, hidden_states

            def get_graph_values(self, hidden_states, use_target_net=False):
                if use_target_net:
                    utilities = self.policy.target_utility(hidden_states)
                    payoff = self.policy.target_payoffs(hidden_states, self.policy.graph.edges_from, self.policy.graph.edges_to)
                else:
                    utilities = self.policy.utility(hidden_states)
                    payoff = self.policy.payoffs(hidden_states, self.policy.graph.edges_from, self.policy.graph.edges_to)
                return utilities, payoff

            def act(self, hidden_states, avail_actions=None):
                with torch.no_grad():
                    f_i, f_ij = self.get_graph_values(hidden_states)
                n_edges = self.policy.graph.n_edges
                n_vertexes = self.policy.graph.n_vertexes
                f_i_mean = f_i.double() / n_vertexes
                f_ij_mean = f_ij.double() / n_edges
                f_ji_mean = f_ij_mean.transpose(dim0=-1, dim1=-2).clone()
                batch_size = f_i.shape[0]

                msg_ij = torch.zeros(batch_size, n_edges, self.dim_act).to(self.device)  # i -> j (send)
                msg_ji = torch.zeros(batch_size, n_edges, self.dim_act).to(self.device)  # j -> i (receive)
                #
                msg_forward = torch_scatter.scatter_add(src=msg_ij, index=self.policy.graph.edges_to, dim=1,
                                                        dim_size=n_vertexes)
                msg_backward = torch_scatter.scatter_add(src=msg_ji, index=self.policy.graph.edges_from, dim=1,
                                                         dim_size=n_vertexes)
                utility = f_i_mean + msg_forward + msg_backward
                if len(self.policy.graph.edges) != 0:
                    for i in range(self.args.n_msg_iterations):
                        joint_forward = (utility[:, self.policy.graph.edges_from, :] - msg_ji).unsqueeze(dim=-1) + f_ij_mean
                        joint_backward = (utility[:, self.policy.graph.edges_to, :] - msg_ij).unsqueeze(dim=-1) + f_ji_mean
                        msg_ij = joint_forward.max(dim=-2).values
                        msg_ji = joint_backward.max(dim=-2).values
                        if self.args.msg_normalized:
                            msg_ij -= msg_ij.mean(dim=-1, keepdim=True)
                            msg_ji -= msg_ji.mean(dim=-1, keepdim=True)

                        msg_forward = torch_scatter.scatter_add(src=msg_ij, index=self.policy.graph.edges_to, dim=1,
                                                                dim_size=n_vertexes)
                        msg_backward = torch_scatter.scatter_add(src=msg_ji, index=self.policy.graph.edges_from, dim=1,
                                                                 dim_size=n_vertexes)
                        utility = f_i_mean + msg_forward + msg_backward
                if avail_actions is not None:
                    avail_actions = torch.Tensor(avail_actions)
                    utility_detach = utility.clone().detach()
                    utility_detach[avail_actions == 0] = -9999999
                    actions_greedy = utility_detach.argmax(dim=-1)
                else:
                    actions_greedy = utility.argmax(dim=-1)
                return actions_greedy

            def q_dcg(self, hidden_states, actions, states=None, use_target_net=False):
                f_i, f_ij = self.get_graph_values(hidden_states, use_target_net=use_target_net)
                f_i_mean = f_i.double() / self.policy.graph.n_vertexes
                f_ij_mean = f_ij.double() / self.policy.graph.n_edges
                utilities = f_i_mean.gather(-1, actions.unsqueeze(dim=-1).long()).sum(dim=1)
                if len(self.policy.graph.edges) == 0 or self.args.n_msg_iterations == 0:
                    return utilities
                actions_ij = (actions[:, self.policy.graph.edges_from] * self.dim_act + actions[:,
                                                                                        self.policy.graph.edges_to]).unsqueeze(
                    -1)
                payoffs = f_ij_mean.reshape(list(f_ij_mean.shape[0:-2]) + [-1]).gather(-1, actions_ij.long()).sum(dim=1)
                if self.args.agent == "DCG_S":
                    state_value = self.policy.bias(states)
                    return utilities + payoffs + state_value
                else:
                    return utilities + payoffs

            def update(self, sample):
                self.iterations += 1
                state = torch.Tensor(sample['state']).to(self.device)
                obs = torch.Tensor(sample['obs']).to(self.device)
                actions = torch.Tensor(sample['actions']).to(self.device)
                state_next = torch.Tensor(sample['state_next']).to(self.device)
                obs_next = torch.Tensor(sample['obs_next']).to(self.device)
                rewards = torch.Tensor(sample['rewards']).mean(dim=1).to(self.device)
                terminals = torch.Tensor(sample['terminals']).all(dim=1, keepdims=True).float().to(self.device)
                agent_mask = torch.Tensor(sample['agent_mask']).float().reshape(-1, self.n_agents, 1).to(self.device)
                IDs = torch.eye(self.n_agents).unsqueeze(0).expand(self.args.batch_size, -1, -1).to(self.device)

                _, hidden_states = self.get_hidden_states(obs, use_target_net=False)
                q_eval_a = self.q_dcg(hidden_states, actions, states=state, use_target_net=False)
                with torch.no_grad():
                    _, hidden_states_next = self.get_hidden_states(obs_next)
                    action_next_greedy = torch.Tensor(self.act(hidden_states_next)).to(self.device)
                    _, hidden_states_target = self.get_hidden_states(obs_next, use_target_net=True)
                    q_next_a = self.q_dcg(hidden_states_target, action_next_greedy, states=state_next, use_target_net=True)

                q_target = rewards + (1 - terminals) * self.args.gamma * q_next_a

                # calculate the loss function
                loss = self.mse_loss(q_eval_a, q_target.detach())
                self.optimizer.zero_grad()
                loss.backward()
                if self.args.use_grad_clip:
                    torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.args.grad_clip_norm)
                self.optimizer.step()
                if self.scheduler is not None:
                    self.scheduler.step()

                if self.iterations % self.sync_frequency == 0:
                    self.policy.copy_target()
                lr = self.optimizer.state_dict()['param_groups'][0]['lr']

                info = {
                    "learning_rate": lr,
                    "loss_Q": loss.item(),
                    "predictQ": q_eval_a.mean().item()
                }

                return info

            def update_recurrent(self, sample):
                self.iterations += 1
                state = torch.Tensor(sample['state']).to(self.device)
                obs = torch.Tensor(sample['obs']).to(self.device)
                actions = torch.Tensor(sample['actions']).to(self.device)
                rewards = torch.Tensor(sample['rewards']).mean(dim=1, keepdims=False).to(self.device)
                terminals = torch.Tensor(sample['terminals']).float().to(self.device)
                avail_actions = torch.Tensor(sample['avail_actions']).float().to(self.device)
                filled = torch.Tensor(sample['filled']).float().to(self.device)
                batch_size = actions.shape[0]
                episode_length = actions.shape[2]
                IDs = torch.eye(self.n_agents).unsqueeze(1).unsqueeze(0).expand(batch_size, -1, episode_length + 1, -1).to(
                    self.device)

                rnn_hidden = self.policy.representation.init_hidden(batch_size * self.n_agents)
                _, hidden_states = self.get_hidden_states(obs.reshape(-1, episode_length + 1, self.dim_obs),
                                                          *rnn_hidden, use_target_net=False)
                hidden_states = hidden_states.reshape(batch_size, self.n_agents, episode_length + 1, -1).transpose(1, 2)
                batch_transitions = batch_size * episode_length
                actions = actions.transpose(1, 2).reshape(batch_transitions, self.n_agents)
                q_eval_a = self.q_dcg(hidden_states[:, :-1].reshape(batch_transitions, self.n_agents, self.dim_hidden_state),
                                      actions, states=state[:, :-1].reshape(batch_transitions, -1),
                                      use_target_net=False)
                with torch.no_grad():
                    avail_a_next = avail_actions.transpose(1, 2)[:, 1:].reshape(batch_transitions, self.n_agents, self.dim_act)
                    hidden_states_next = hidden_states[:, 1:].reshape(batch_transitions, self.n_agents, self.dim_hidden_state)
                    action_next_greedy = torch.Tensor(self.act(hidden_states_next, avail_actions=avail_a_next)).to(self.device)
                    rnn_hidden_target = self.policy.target_representation.init_hidden(batch_size * self.n_agents)
                    _, hidden_states_tar = self.get_hidden_states(obs[:, :, 1:].reshape(-1, episode_length, self.dim_obs),
                                                                  *rnn_hidden_target, use_target_net=True)
                    hidden_states_tar = hidden_states_tar.reshape(batch_size, self.n_agents, episode_length, -1).transpose(1, 2)
                    q_next_a = self.q_dcg(hidden_states_tar.reshape(batch_transitions, self.n_agents, self.dim_hidden_state),
                                          action_next_greedy,
                                          states=state[:, 1:].reshape(batch_transitions, -1),
                                          use_target_net=True)
                rewards = rewards.reshape(-1, 1)
                terminals = terminals.reshape(-1, 1)
                filled = filled.reshape(-1, 1)
                q_target = rewards + (1 - terminals) * self.args.gamma * q_next_a
                td_error = (q_eval_a - q_target.detach()) * filled

                # calculate the loss function
                loss = (td_error ** 2).sum() / filled.sum()
                self.optimizer.zero_grad()
                loss.backward()
                if self.args.use_grad_clip:
                    torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.args.grad_clip_norm)
                self.optimizer.step()
                if self.scheduler is not None:
                    self.scheduler.step()

                if self.iterations % self.sync_frequency == 0:
                    self.policy.copy_target()
                lr = self.optimizer.state_dict()['param_groups'][0]['lr']

                info = {
                    "learning_rate": lr,
                    "loss_Q": loss.item(),
                    "predictQ": q_eval_a.mean().item()
                }

                return info





  .. group-tab:: TensorFlow

    .. code-block:: python

        """
        DCG: Deep coordination graphs
        Paper link: http://proceedings.mlr.press/v119/boehmer20a/boehmer20a.pdf
        Implementation: TensorFlow 2.X
        """
        import torch

        from xuance.tensorflow.learners import *
        import torch_scatter


        class DCG_Learner(LearnerMAS):
            def __init__(self,
                         config: Namespace,
                         policy: tk.Model,
                         optimizer: tk.optimizers.Optimizer,
                         device: str = "cpu:0",
                         model_dir: str = "./",
                         gamma: float = 0.99,
                         sync_frequency: int = 100
                         ):
                self.gamma = gamma
                self.use_recurrent = config.use_recurrent
                self.sync_frequency = sync_frequency
                self.dim_hidden_state = policy.representation.output_shapes['state'][0]
                self.sync_frequency = sync_frequency
                super(DCG_Learner, self).__init__(config, policy, optimizer, device, model_dir)

            def get_hidden_states(self, obs_n, *rnn_hidden, use_target_net=False):
                if self.use_recurrent:
                    if use_target_net:
                        outputs = self.policy.target_representation(obs_n, *rnn_hidden)
                    else:
                        outputs = self.policy.representation(obs_n, *rnn_hidden)
                    hidden_states = outputs['state']
                    rnn_hidden = (outputs['rnn_hidden'], outputs['rnn_cell'])
                else:
                    shape_obs_n = obs_n.shape
                    rep_in = tf.reshape(obs_n, [-1, shape_obs_n[-1]])
                    if use_target_net:
                        hidden_states = self.policy.target_representation(rep_in)['state']
                    else:
                        hidden_states = self.policy.representation(rep_in)['state']
                    hidden_states_out = tf.reshape(hidden_states, shape_obs_n[:-1] + (self.dim_hidden_state, ))
                    rnn_hidden = None
                return rnn_hidden, hidden_states_out

            def get_graph_values(self, hidden_states, use_target_net=False):
                if use_target_net:
                    utilities = self.policy.target_utility(hidden_states)
                    payoff = self.policy.target_payoffs(hidden_states, self.policy.graph.edges_from, self.policy.graph.edges_to)
                else:
                    utilities = self.policy.utility(hidden_states)
                    payoff = self.policy.payoffs(hidden_states, self.policy.graph.edges_from.numpy(), self.policy.graph.edges_to.numpy())
                return utilities, payoff

            def act(self, hidden_states, avail_actions=None):
                with torch.no_grad():
                    f_i, f_ij = self.get_graph_values(hidden_states)
                n_edges = self.policy.graph.n_edges
                n_vertexes = self.policy.graph.n_vertexes
                f_i_mean = tf.cast(f_i, dtype=tf.double) / n_vertexes
                f_ij_mean = tf.cast(f_ij, dtype=tf.double) / n_edges
                f_ji_mean = copy.deepcopy(tf.transpose(f_ij_mean, perm=(0, 1, 3, 2)))
                batch_size = f_i.shape[0]

                msg_ij = torch.zeros(batch_size, n_edges, self.dim_act)  # i -> j (send)
                msg_ji = torch.zeros(batch_size, n_edges, self.dim_act)  # j -> i (receive)
                #
                msg_forward = torch_scatter.scatter_add(src=msg_ij, index=self.policy.graph.edges_to, dim=1,
                                                        dim_size=n_vertexes)
                msg_backward = torch_scatter.scatter_add(src=msg_ji, index=self.policy.graph.edges_from, dim=1,
                                                         dim_size=n_vertexes)

                f_i_mean = torch.tensor(f_i_mean.numpy())
                f_ij_mean = torch.tensor(f_ij_mean.numpy())
                f_ji_mean = torch.tensor(f_ji_mean.numpy())
                utility = f_i_mean + msg_forward + msg_backward
                if len(self.policy.graph.edges) != 0:
                    for i in range(self.args.n_msg_iterations):
                        joint_forward = (utility[:, self.policy.graph.edges_from, :] - msg_ji).unsqueeze(dim=-1) + f_ij_mean
                        joint_backward = (utility[:, self.policy.graph.edges_to, :] - msg_ij).unsqueeze(dim=-1) + f_ji_mean
                        msg_ij = joint_forward.max(dim=-2).values
                        msg_ji = joint_backward.max(dim=-2).values
                        if self.args.msg_normalized:
                            msg_ij -= msg_ij.mean(dim=-1, keepdim=True)
                            msg_ji -= msg_ji.mean(dim=-1, keepdim=True)

                        msg_forward = torch_scatter.scatter_add(src=msg_ij, index=self.policy.graph.edges_to, dim=1,
                                                                dim_size=n_vertexes)
                        msg_backward = torch_scatter.scatter_add(src=msg_ji, index=self.policy.graph.edges_from, dim=1,
                                                                 dim_size=n_vertexes)
                        utility = f_i_mean + msg_forward + msg_backward
                if avail_actions is not None:
                    avail_actions = torch.Tensor(avail_actions)
                    utility_detach = utility.clone().detach()
                    utility_detach[avail_actions == 0] = -9999999
                    actions_greedy = utility_detach.argmax(dim=-1)
                else:
                    actions_greedy = utility.argmax(dim=-1)
                return actions_greedy

            def q_dcg(self, hidden_states, actions, states=None, use_target_net=False):
                f_i, f_ij = self.get_graph_values(hidden_states, use_target_net=use_target_net)
                f_i_mean = tf.cast(f_i, tf.double) / self.policy.graph.n_vertexes
                f_ij_mean = tf.cast(f_ij, tf.double) / self.policy.graph.n_edges
                utilities = tf.reduce_sum(tf.gather(f_i_mean, tf.expand_dims(actions, -1), axis=-1, batch_dims=-1), axis=1)
                if len(self.policy.graph.edges) == 0 or self.args.n_msg_iterations == 0:
                    return utilities
                edges_from = self.policy.graph.edges_from.numpy()
                edges_to = self.policy.graph.edges_to.numpy()
                actions_ij = tf.expand_dims(tf.gather(actions, edges_from, axis=1) * self.dim_act + tf.gather(actions, edges_to, axis=1), -1)
                payoffs = tf.reduce_sum(tf.gather(tf.reshape(f_ij_mean, list(f_ij_mean.shape[0:-2]) + [-1]), actions_ij, axis=-1, batch_dims=-1), axis=1)
                if self.args.agent == "DCG_S":
                    state_value = self.policy.bias(states)
                    return utilities + payoffs + state_value
                else:
                    return utilities + payoffs

            def update(self, sample):
                self.iterations += 1
                with tf.device(self.device):
                    state = tf.convert_to_tensor(sample['state'])
                    state_next = tf.convert_to_tensor(sample['state_next'])
                    obs = tf.convert_to_tensor(sample['obs'])
                    actions = tf.convert_to_tensor(sample['actions'], dtype=tf.int64)
                    obs_next = tf.convert_to_tensor(sample['obs_next'])
                    rewards = tf.reduce_mean(tf.convert_to_tensor(sample['rewards']), axis=1)
                    terminals = tf.reshape(tf.convert_to_tensor(sample['terminals'].all(axis=-1, keepdims=True), dtype=tf.float32), [-1, 1])
                    agent_mask = tf.reshape(tf.convert_to_tensor(sample['agent_mask'], dtype=tf.float32),
                                            [-1, self.n_agents, 1])
                    IDs = tf.tile(tf.expand_dims(tf.eye(self.n_agents), axis=0), multiples=(self.args.batch_size, 1, 1))
                    batch_size = obs.shape[0]

                    with tf.GradientTape() as tape:
                        _, hidden_states = self.get_hidden_states(obs, use_target_net=False)
                        q_eval_a = self.q_dcg(hidden_states, actions, states=state, use_target_net=False)

                        _, hidden_states_next = self.get_hidden_states(obs_next)
                        action_next_greedy = tf.convert_to_tensor(self.act(hidden_states_next))
                        _, hidden_states_target = self.get_hidden_states(obs_next, use_target_net=True)
                        q_next_a = self.q_dcg(hidden_states_target, action_next_greedy, states=state_next, use_target_net=True)
                        q_next_a = tf.cast(q_next_a, dtype=tf.float32)
                        q_target = rewards + (1 - terminals) * self.args.gamma * q_next_a

                        # calculate the loss function
                        y_true = tf.stop_gradient(tf.reshape(q_target, [-1]))
                        y_pred = tf.reshape(q_eval_a, [-1])
                        loss = tk.losses.mean_squared_error(y_true, y_pred)
                        gradients = tape.gradient(loss, self.policy.trainable_variables)
                        self.optimizer.apply_gradients([
                            (grad, var)
                            for (grad, var) in zip(gradients, self.policy.trainable_variables)
                            if grad is not None
                        ])

                    if self.iterations % self.sync_frequency == 0:
                        self.policy.copy_target()

                    lr = self.optimizer._decayed_lr(tf.float32)

                    info = {
                        "learning_rate": lr.numpy(),
                        "loss_Q": loss.numpy(),
                        "predictQ": tf.math.reduce_mean(q_eval_a).numpy()
                    }

                    return info


  .. group-tab:: MindSpore

    .. code-block:: python

        """
        DCG: Deep coordination graphs
        Paper link: http://proceedings.mlr.press/v119/boehmer20a/boehmer20a.pdf
        Implementation: MindSpore
        """
        from xuance.mindspore.learners import *
        import torch_scatter
        import torch
        import copy


        class DCG_Learner(LearnerMAS):
            class PolicyNetWithLossCell(nn.Cell):
                def __init__(self, backbone, n_msg_iterations, dim_act, agent, use_recurrent):
                    super(DCG_Learner.PolicyNetWithLossCell, self).__init__(auto_prefix=False)
                    self._backbone = backbone
                    self.n_msg_iterations = n_msg_iterations
                    self.expand_dims = ops.ExpandDims()
                    self.dim_act = dim_act
                    self.agent = agent
                    self.use_recurrent = use_recurrent

                def construct(self, s, o, a, label, *rnn_hidden):
                    # get hidden states
                    if self.use_recurrent:
                        outputs = self._backbone.representation(o, *rnn_hidden)
                        hidden_states = outputs['state']
                    else:
                        hidden_states = self._backbone.representation(o)['state']

                    # get evaluate Q values
                    f_i = self._backbone.utility(hidden_states)
                    f_ij = self._backbone.payoffs(hidden_states, self._backbone.graph.edges_from, self._backbone.graph.edges_to)
                    f_i_mean = f_i.astype(ms.double) / self._backbone.graph.n_vertexes
                    f_ij_mean = f_ij.astype(ms.double) / self._backbone.graph.n_edges
                    utilities = GatherD()(f_i_mean, -1, self.expand_dims(a, -1).astype(ms.int32)).sum(axis=1)
                    if len(self._backbone.graph.edges) == 0 or self.n_msg_iterations == 0:
                        q_eval_a = utilities
                    else:
                        actions_ij = self.expand_dims(
                            (a[:, self._backbone.graph.edges_from] * self.dim_act + a[:, self._backbone.graph.edges_to]), -1)
                        payoffs = GatherD()(f_ij_mean.view(tuple(list(f_ij_mean.shape[0:-2]) + [-1])), -1, actions_ij).sum(axis=1)
                        if self.agent == "DCG_S":
                            state_value = self._backbone.bias(s)
                            q_eval_a = utilities + payoffs + state_value
                        else:
                            q_eval_a = utilities + payoffs

                    td_error = q_eval_a - label
                    loss = (td_error ** 2).mean()
                    return loss

            def __init__(self,
                         config: Namespace,
                         policy: nn.Cell,
                         optimizer: nn.Optimizer,
                         scheduler: Optional[nn.exponential_decay_lr] = None,
                         model_dir: str = "./",
                         gamma: float = 0.99,
                         sync_frequency: int = 100
                         ):
                self.gamma = gamma
                self.use_recurrent = config.use_recurrent
                self.sync_frequency = sync_frequency
                self.mse_loss = nn.MSELoss()
                super(DCG_Learner, self).__init__(config, policy, optimizer, scheduler, model_dir)
                # build train net
                self.zeros = ms.ops.Zeros()
                self._mean = ops.ReduceMean(keep_dims=False)
                self.transpose = ops.Transpose()
                self.loss_net = self.PolicyNetWithLossCell(policy, config.n_msg_iterations,
                                                           self.dim_act, config.agent, self.use_recurrent)
                self.policy_train = nn.TrainOneStepCell(self.loss_net, optimizer)
                self.policy_train.set_train()

            def get_hidden_states(self, obs_n, *rnn_hidden, use_target_net=False):
                if self.use_recurrent:
                    if use_target_net:
                        outputs = self.policy.target_representation(obs_n, *rnn_hidden)
                    else:
                        outputs = self.policy.representation(obs_n, *rnn_hidden)
                    hidden_states = outputs['state']
                    rnn_hidden = (outputs['rnn_hidden'], outputs['rnn_cell'])
                else:
                    if use_target_net:
                        hidden_states = self.policy.target_representation(obs_n)['state']
                    else:
                        hidden_states = self.policy.representation(obs_n)['state']
                    rnn_hidden = None
                return rnn_hidden, hidden_states

            def get_graph_values(self, hidden_states, use_target_net=False):
                if use_target_net:
                    utilities = self.policy.target_utility(hidden_states)
                    payoff = self.policy.target_payoffs(hidden_states, self.policy.graph.edges_from, self.policy.graph.edges_to)
                else:
                    utilities = self.policy.utility(hidden_states)
                    payoff = self.policy.payoffs(hidden_states, self.policy.graph.edges_from, self.policy.graph.edges_to)
                return utilities, payoff

            def act(self, hidden_states, avail_actions=None):
                f_i, f_ij = self.get_graph_values(hidden_states)
                n_edges = self.policy.graph.n_edges
                n_vertexes = self.policy.graph.n_vertexes
                f_i_mean = f_i.astype(ms.double) / n_vertexes
                f_ij_mean = f_ij.astype(ms.double) / n_edges
                f_ji_mean = copy.deepcopy(self.transpose(f_ij_mean, (0, 1, 3, 2)))
                batch_size = f_i.shape[0]

                msg_ij = torch.zeros(batch_size, n_edges, self.dim_act)  # i -> j (send)
                msg_ji = torch.zeros(batch_size, n_edges, self.dim_act)  # j -> i (receive)
                #
                msg_forward = torch_scatter.scatter_add(src=msg_ij, index=torch.tensor(self.policy.graph.edges_to.asnumpy()),
                                                        dim=1, dim_size=n_vertexes)
                msg_backward = torch_scatter.scatter_add(src=msg_ji, index=torch.tensor(self.policy.graph.edges_from.asnumpy()),
                                                         dim=1, dim_size=n_vertexes)
                utility = f_i_mean + Tensor(msg_forward.numpy()) + Tensor(msg_backward.numpy())
                if len(self.policy.graph.edges) != 0:
                    utility = torch.tensor(utility.asnumpy())
                    f_i_mean = torch.tensor(f_i_mean.asnumpy())
                    f_ij_mean = torch.tensor(f_ij_mean.asnumpy())
                    f_ji_mean = torch.tensor(f_ji_mean.asnumpy())
                    edges_from = torch.tensor(self.policy.graph.edges_from.asnumpy())
                    edges_to = torch.tensor(self.policy.graph.edges_to.asnumpy())
                    for i in range(self.args.n_msg_iterations):
                        joint_forward = (utility[:, edges_from, :] - msg_ji).unsqueeze(dim=-1) + f_ij_mean
                        joint_backward = (utility[:, edges_to, :] - msg_ij).unsqueeze(dim=-1) + f_ji_mean
                        msg_ij = joint_forward.max(dim=-2).values
                        msg_ji = joint_backward.max(dim=-2).values
                        if self.args.msg_normalized:
                            msg_ij -= msg_ij.mean(dim=-1, keepdim=True)
                            msg_ji -= msg_ji.mean(dim=-1, keepdim=True)

                        msg_forward = torch_scatter.scatter_add(src=msg_ij, index=edges_to, dim=1,
                                                                dim_size=n_vertexes)
                        msg_backward = torch_scatter.scatter_add(src=msg_ji, index=edges_from, dim=1,
                                                                 dim_size=n_vertexes)
                        utility = f_i_mean + msg_forward + msg_backward
                utility = Tensor(utility.numpy())
                if avail_actions is not None:
                    utility_detach = copy.deepcopy(utility)
                    utility_detach[avail_actions == 0] = -9999999
                    actions_greedy = utility_detach.argmax(axis=-1)
                else:
                    actions_greedy = utility.argmax(axis=-1)
                return actions_greedy

            def q_dcg(self, hidden_states, actions, states=None, use_target_net=False):
                f_i, f_ij = self.get_graph_values(hidden_states, use_target_net=use_target_net)
                f_i_mean = f_i.astype(ms.double) / self.policy.graph.n_vertexes
                f_ij_mean = f_ij.astype(ms.double) / self.policy.graph.n_edges
                utilities = GatherD()(f_i_mean, -1, self.expand_dims(actions, -1).astype(ms.int32)).sum(axis=1)
                if len(self.policy.graph.edges) == 0 or self.args.n_msg_iterations == 0:
                    return utilities
                actions_ij = self.expand_dims((actions[:, self.policy.graph.edges_from] * self.dim_act + actions[:, self.policy.graph.edges_to]), -1)
                payoffs = GatherD()(f_ij_mean.view(tuple(list(f_ij_mean.shape[0:-2]) + [-1])), -1, actions_ij).sum(axis=1)
                if self.args.agent == "DCG_S":
                    state_value = self.policy.bias(states)
                    return utilities + payoffs + state_value
                else:
                    return utilities + payoffs

            def update(self, sample):
                self.iterations += 1
                state = Tensor(sample['state'])
                obs = Tensor(sample['obs'])
                actions = Tensor(sample['actions']).astype(ms.int32)
                state_next = Tensor(sample['state_next'])
                obs_next = Tensor(sample['obs_next'])
                rewards = self._mean(Tensor(sample['rewards']), 1)
                terminals = Tensor(sample['terminals']).view(-1, self.n_agents, 1).all(axis=1, keep_dims=False)
                agent_mask = Tensor(sample['agent_mask']).view(-1, self.n_agents, 1)
                batch_size = obs.shape[0]
                IDs = ops.broadcast_to(self.expand_dims(self.eye(self.n_agents, self.n_agents, ms.float32), 0),
                                       (batch_size, -1, -1))

                _, hidden_states_next = self.get_hidden_states(obs_next)
                action_next_greedy = Tensor(self.act(hidden_states_next))
                _, hidden_states_target = self.get_hidden_states(obs_next, use_target_net=True)
                q_next_a = self.q_dcg(hidden_states_target, action_next_greedy, states=state_next, use_target_net=True)
                q_target = rewards + (1 - terminals) * self.args.gamma * q_next_a

                # calculate the loss and train
                loss = self.policy_train(state, obs, actions, q_target)
                if self.iterations % self.sync_frequency == 0:
                    self.policy.copy_target()

                lr = self.scheduler(self.iterations).asnumpy()

                info = {
                    "learning_rate": lr,
                    "loss_Q": loss.asnumpy()
                }

                return info

