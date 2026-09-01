"""
DCG: Deep coordination graphs
Paper link: http://proceedings.mlr.press/v119/boehmer20a/boehmer20a.pdf
Implementation: TensorFlow 2.X
"""
from argparse import Namespace
from xuance.common import AgentGrouping

from xuance.tensorflow import tf, Module
from xuance.tensorflow.learners import LearnerMAS

try:
    import torch_scatter
except ImportError:
    print("The module torch_scatter is not installed.")


class DCG_Learner(LearnerMAS):
    def __init__(self,
                 config: Namespace,
                 agent_grouping: AgentGrouping,
                 model: Module,
                 callback):
        super(DCG_Learner, self).__init__(config, agent_grouping, model, callback)
        self.dim_hidden_state = model.representation[self.group_keys[0]].output_shapes['state'][0]
        self.dim_act = max([self.model.action_space[key].n for key in self.agent_keys])
        self.sync_frequency = config.sync_frequency

    def get_graph_values(self, hidden_states, use_target_net=False):
        if use_target_net:
            utilities = self.model.target_utility(hidden_states)
            payoff = self.model.target_payoffs(hidden_states, self.model.graph.edges_from, self.model.graph.edges_to)
        else:
            utilities = self.model.utility(hidden_states)
            payoff = self.model.payoffs(hidden_states, self.model.graph.edges_from, self.model.graph.edges_to)
        return utilities, payoff

    def act(self, hidden_states, avail_actions: Tensor | None = None):
        """
        Calculate the actions via belief propagation.

        Args:
            hidden_states (torch.Tensor): The hidden states for the representation of all agents.
            avail_actions (torch.Tensor): The avail actions for the agents, default is None.

        Returns: The actions.
        """
        with torch.no_grad():
            f_i, f_ij = self.get_graph_values(hidden_states)
        n_edges = self.model.graph.n_edges
        n_vertexes = self.model.graph.n_vertexes
        f_i_mean = f_i.double() / n_vertexes
        f_ij_mean = f_ij.double() / n_edges
        f_ji_mean = f_ij_mean.transpose(dim0=-1, dim1=-2).clone()
        batch_size = f_i.shape[0]

        msg_ij = torch.zeros(batch_size, n_edges, self.dim_act).to(self.device)  # i -> j (send)
        msg_ji = torch.zeros(batch_size, n_edges, self.dim_act).to(self.device)  # j -> i (receive)
        #
        msg_forward = torch_scatter.scatter_add(src=msg_ij, index=self.model.graph.edges_to, dim=1,
                                                dim_size=n_vertexes)
        msg_backward = torch_scatter.scatter_add(src=msg_ji, index=self.model.graph.edges_from, dim=1,
                                                 dim_size=n_vertexes)
        utility = f_i_mean + msg_forward + msg_backward
        if len(self.model.graph.edges) != 0:
            for i in range(self.config.n_msg_iterations):
                joint_forward = (utility[:, self.model.graph.edges_from, :] - msg_ji).unsqueeze(dim=-1) + f_ij_mean
                joint_backward = (utility[:, self.model.graph.edges_to, :] - msg_ij).unsqueeze(dim=-1) + f_ji_mean
                msg_ij = joint_forward.max(dim=-2).values
                msg_ji = joint_backward.max(dim=-2).values
                if self.config.msg_normalized:
                    msg_ij -= msg_ij.mean(dim=-1, keepdim=True)
                    msg_ji -= msg_ji.mean(dim=-1, keepdim=True)

                msg_forward = torch_scatter.scatter_add(src=msg_ij, index=self.model.graph.edges_to, dim=1,
                                                        dim_size=n_vertexes)
                msg_backward = torch_scatter.scatter_add(src=msg_ji, index=self.model.graph.edges_from, dim=1,
                                                         dim_size=n_vertexes)
                utility = f_i_mean + msg_forward + msg_backward
        if avail_actions is not None:
            avail_actions = torch.as_tensor(avail_actions, device=self.device)
            utility_detach = utility.clone().detach()
            utility_detach[avail_actions == 0] = -1e10
            actions_greedy = utility_detach.argmax(dim=-1)
        else:
            actions_greedy = utility.argmax(dim=-1)
        return actions_greedy

    def q_dcg(self, hidden_states, actions, states=None, use_target_net=False):
        f_i, f_ij = self.get_graph_values(hidden_states, use_target_net=use_target_net)
        f_i_mean = tf.cast(f_i, tf.double) / self.model.graph.n_vertexes
        f_ij_mean = tf.cast(f_ij, tf.double) / self.model.graph.n_edges
        utilities = tf.reduce_sum(tf.gather(f_i_mean, tf.expand_dims(actions, -1), axis=-1, batch_dims=-1), axis=1)
        if len(self.model.graph.edges) == 0 or self.args.n_msg_iterations == 0:
            return utilities
        edges_from = self.model.graph.edges_from.numpy()
        edges_to = self.model.graph.edges_to.numpy()
        actions_ij = tf.expand_dims(tf.gather(actions, edges_from, axis=1) * self.dim_act + tf.gather(actions, edges_to, axis=1), -1)
        payoffs = tf.reduce_sum(tf.gather(tf.reshape(f_ij_mean, list(f_ij_mean.shape[0:-2]) + [-1]), actions_ij, axis=-1, batch_dims=-1), axis=1)
        if self.config.agent == "DCG_S":
            state_value = self.model.bias(states)
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
                loss = self.mse_loss(y_true, y_pred)
                gradients = tape.gradient(loss, self.model.trainable_variables)
                self.optimizer.apply_gradients([
                    (grad, var)
                    for (grad, var) in zip(gradients, self.model.trainable_variables)
                    if grad is not None
                ])

            if self.iterations % self.sync_frequency == 0:
                self.model.copy_target()

            lr = self.optimizer._decayed_lr(tf.float32)

            info = {
                "learning_rate": lr.numpy(),
                "loss_Q": loss.numpy(),
                "predictQ": tf.math.reduce_mean(q_eval_a).numpy()
            }

            return info
