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
