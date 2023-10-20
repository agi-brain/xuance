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
                 modeldir: str = "./",
                 gamma: float = 0.99,
                 sync_frequency: int = 100
                 ):
        self.gamma = gamma
        self.sync_frequency = sync_frequency
        super(DCG_Learner, self).__init__(config, policy, optimizer, device, modeldir)

    def save_model(self):
        pass
        model_path = self.modeldir + "model-%s-%s" % (time.asctime(), str(self.iterations))
        self.policy.representation.save(model_path + "/representations")
        self.policy.utility.save(model_path + "/utility")
        self.policy.target_utility.save(model_path + "/target_utility")
        self.policy.payoffs.save(model_path + "/payoffs")
        self.policy.target_payoffs.save(model_path + "/target_payoffs")
        if self.policy.dcg_s:
            self.policy.bias.save(model_path + "/bias")
            self.policy.target_bias.save(model_path + "/target_bias")

    def load_model(self, path):
        model_names = os.listdir(path)
        # try:
        model_names.sort()
        model_path = path + model_names[-1]
        print(model_path)
        # self.policy = tk.models.load_model(model_path, compile=False)
        self.policy.representation.load_weights(model_path + "/representations")
        self.policy.utility.load_weights(model_path + "/utility")
        self.policy.target_utility.load_weights(model_path + "/target_utility")
        self.policy.payoffs.load_weights(model_path + "/payoffs")
        self.policy.target_payoffs.load_weights(model_path + "/target_payoffs")
        if self.policy.dcg_s:
            self.policy.bias.load_weights(model_path + "/bias")
            self.policy.target_bias.load_weights(model_path + "/target_bias")
        # except:
        #     raise "Failed to load model! Please train and save the model first."

    def get_graph_values(self, obs_n, batch=None, use_target_net=False):
        if use_target_net:
            hidden_states = self.policy.representation(obs_n)['state']
            utilities = self.policy.target_utility(hidden_states)
            utilities = tf.reshape(utilities, [batch, self.n_agents, -1])
            hidden_states = tf.reshape(hidden_states, [batch, self.n_agents, -1]).numpy()
            hidden_from_to = np.concatenate([hidden_states[:, self.policy.graph.edges_from],
                                             hidden_states[:, self.policy.graph.edges_to]], axis=-1)
            hidden_to_from = np.concatenate([hidden_states[:, self.policy.graph.edges_to],
                                             hidden_states[:, self.policy.graph.edges_from]], axis=-1)
            payoff_ = self.policy.target_payoffs(tf.convert_to_tensor(hidden_from_to), tf.convert_to_tensor(hidden_to_from))
            payoff = self.policy.target_payoffs.mean_payoffs(payoff_)

        else:
            hidden_states = self.policy.representation(obs_n)['state']
            utilities = self.policy.utility(hidden_states)
            utilities = tf.reshape(utilities, [batch, self.n_agents, -1])
            hidden_states = tf.reshape(hidden_states, [batch, self.n_agents, -1]).numpy()
            hidden_from_to = np.concatenate([hidden_states[:, self.policy.graph.edges_from],
                                             hidden_states[:, self.policy.graph.edges_to]], axis=-1)
            hidden_to_from = np.concatenate([hidden_states[:, self.policy.graph.edges_to],
                                             hidden_states[:, self.policy.graph.edges_from]], axis=-1)
            payoff_ = self.policy.payoffs(tf.convert_to_tensor(hidden_from_to),
                                          tf.convert_to_tensor(hidden_to_from))
            payoff = self.policy.payoffs.mean_payoffs(payoff_)
        return utilities, payoff

    def q_dcg(self, obs_n, actions, batch=None, states=None, use_target_net=False):
        f_i, f_ij = self.get_graph_values(obs_n, batch, use_target_net)
        f_i_mean = tf.cast(f_i, dtype=tf.float32) / self.policy.graph.n_vertexes
        f_ij_mean = tf.cast(f_ij, dtype=tf.float32) / self.policy.graph.n_edges
        utilities = tf.reduce_sum(tf.gather(f_i_mean, indices=tf.expand_dims(actions, axis=-1), axis=-1, batch_dims=-1), axis=1)
        if len(self.policy.graph.edges) == 0 or self.args.n_msg_iterations == 0:
            return utilities
        actions = torch.Tensor(actions.numpy())
        actions_ij = (actions[:, self.policy.graph.edges_from] * self.dim_act + actions[:, self.policy.graph.edges_to]).unsqueeze(-1)
        actions_ij = tf.convert_to_tensor(actions_ij.numpy(), dtype=tf.int64)
        payoffs = tf.reduce_sum(tf.gather(params=tf.reshape(f_ij_mean, list(f_ij_mean.shape[0:-2]) + [-1]),
                                indices=actions_ij, axis=-1, batch_dims=-1), axis=1)
        if self.args.agent == "DCG_S":
            state_value = self.policy.bias(states)
            return utilities + payoffs + state_value
        else:
            return utilities + payoffs

    def act(self, obs_n, episode=None, test_mode=True, noise=False):
        f_i, f_ij = self.get_graph_values(obs_n.reshape([-1, self.dim_obs[0]]), batch=obs_n.shape[0])
        f_i, f_ij = tf.stop_gradient(f_i).numpy(), tf.stop_gradient(f_ij).numpy()
        n_edges = self.policy.graph.n_edges
        n_vertexes = self.policy.graph.n_vertexes
        f_i_mean = f_i / n_vertexes
        f_ij_mean = f_ij / n_edges
        dim_num = len(f_ij_mean.shape)
        dim_trans = list(np.arange(dim_num-2)) + [dim_num-1, dim_num-2]
        f_ji_mean = np.transpose(f_ij_mean, dim_trans).copy()
        batch_size = f_i.shape[0]

        msg_ij = torch.zeros(batch_size, n_edges, self.dim_act)  # i -> j (send)
        msg_ji = torch.zeros(batch_size, n_edges, self.dim_act)  # j -> i (receive)
        #
        msg_forward = torch_scatter.scatter_add(src=msg_ij, index=torch.tensor(self.policy.graph.edges_to), dim=1, dim_size=n_vertexes)
        msg_backward = torch_scatter.scatter_add(src=msg_ji, index=torch.tensor(self.policy.graph.edges_from), dim=1, dim_size=n_vertexes)
        utility = torch.tensor(f_i_mean) + msg_forward + msg_backward
        if len(self.policy.graph.edges) == 0:
            return np.argmax(utility, axis=-1)
        else:
            for i in range(self.args.n_msg_iterations):
                joint_forward = (utility[:, self.policy.graph.edges_from, :] - msg_ji).unsqueeze(dim=-1) + f_ij_mean
                joint_backward = (utility[:, self.policy.graph.edges_to, :] - msg_ij).unsqueeze(dim=-1) + f_ji_mean
                msg_ij = tf.reduce_max(joint_forward, axis=-2)
                msg_ji = tf.reduce_max(joint_backward, axis=-2)
                if self.args.msg_normalized:
                    msg_ij -= tf.reduce_mean(msg_ij, axis=-1, keepdims=True)
                    msg_ji -= tf.reduce_mean(msg_ji, axis=-1, keepdims=True)

                msg_forward = torch_scatter.scatter_add(src=torch.tensor(msg_ij.numpy()),
                                                        index=torch.tensor(self.policy.graph.edges_to), dim=1,
                                                        dim_size=n_vertexes)
                msg_backward = torch_scatter.scatter_add(src=torch.tensor(msg_ji.numpy()),
                                                         index=torch.tensor(self.policy.graph.edges_from), dim=1,
                                                         dim_size=n_vertexes)
                utility = f_i_mean + msg_forward.numpy() + msg_backward.numpy()
            return np.argmax(utility, axis=-1)

    def update(self, sample):
        self.iterations += 1
        with tf.device(self.device):
            state = tf.convert_to_tensor(sample['state'])
            state_next = tf.convert_to_tensor(sample['state_next'])
            obs = tf.convert_to_tensor(sample['obs'])
            actions = tf.convert_to_tensor(sample['actions'], dtype=tf.int64)
            obs_next = tf.convert_to_tensor(sample['obs_next'])
            rewards = tf.reduce_mean(tf.convert_to_tensor(sample['rewards']), axis=1)
            terminals = tf.reshape(tf.convert_to_tensor(sample['terminals'], dtype=tf.float32), [-1, self.n_agents, 1])
            agent_mask = tf.reshape(tf.convert_to_tensor(sample['agent_mask'], dtype=tf.float32),
                                    [-1, self.n_agents, 1])
            IDs = tf.tile(tf.expand_dims(tf.eye(self.n_agents), axis=0), multiples=(self.args.batch_size, 1, 1))
            batch_size = obs.shape[0]

            with tf.GradientTape() as tape:
                q_eval_a = self.q_dcg(tf.reshape(obs, [-1, self.dim_obs[0]]), actions, batch_size, states=state, use_target_net=False)
                action_next_greedy = tf.convert_to_tensor(self.act(obs_next.numpy()))
                q_next_a = self.q_dcg(tf.reshape(obs_next, [-1, self.dim_obs[0]]), action_next_greedy, batch_size, states=state_next, use_target_net=True)
                q_next_a = tf.stop_gradient(q_next_a)

                q_target = rewards + (1-terminals) * self.args.gamma * q_next_a

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
