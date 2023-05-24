"""
DCG: Deep coordination graphs
Paper link: http://proceedings.mlr.press/v119/boehmer20a/boehmer20a.pdf
Implementation: MindSpore
"""
from xuanpolicy.mindspore.learners import *
import torch_scatter
import torch
import copy


class DCG_Learner(LearnerMAS):
    class PolicyNetWithLossCell(nn.Cell):
        def __init__(self, backbone, n_msg_iterations, dim_act, agent):
            super(DCG_Learner.PolicyNetWithLossCell, self).__init__(auto_prefix=False)
            self._backbone = backbone
            self.n_msg_iterations = n_msg_iterations
            self.expand_dims = ops.ExpandDims()
            self.dim_act = dim_act
            self.agent = agent

        def construct(self, s, o, a, label):
            # the q_dcg network
            # self.get_graph_values
            hidden_states = self._backbone.representation(o)['state']
            f_i = self._backbone.utility(hidden_states)
            f_ij = self._backbone.payoffs(hidden_states, self._backbone.graph.edges_from, self._backbone.graph.edges_to)

            f_i_mean = f_i.astype(ms.double) / self._backbone.graph.n_vertexes
            f_ij_mean = f_ij.astype(ms.double) / self._backbone.graph.n_edges
            utilities = GatherD()(f_i_mean, -1, self.expand_dims(a, -1).astype(ms.int32)).sum(axis=1)
            if len(self._backbone.graph.edges) == 0 or self.n_msg_iterations == 0:
                q_eval_a = utilities
            else:
                actions_ij = self.expand_dims((a[:, self._backbone.graph.edges_from] * self.dim_act + a[:, self._backbone.graph.edges_to]), -1)
                f_ij_mean_shape = f_ij_mean.shape
                payoffs = GatherD()(f_ij_mean.view(tuple(list(f_ij_mean_shape[0:-2]) + [-1])), -1, actions_ij.astype(ms.int32)).sum(axis=1)
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
                 summary_writer: Optional[SummaryWriter] = None,
                 modeldir: str = "./",
                 gamma: float = 0.99,
                 sync_frequency: int = 100
                 ):
        self.gamma = gamma
        self.sync_frequency = sync_frequency
        self.mse_loss = nn.MSELoss()
        super(DCG_Learner, self).__init__(config, policy, optimizer, scheduler, summary_writer, modeldir)
        # build train net
        self.zeros = ms.ops.Zeros()
        self._mean = ops.ReduceMean(keep_dims=False)
        self.transpose = ops.Transpose()
        self.loss_net = self.PolicyNetWithLossCell(policy, config.n_msg_iterations, self.dim_act, config.agent)
        self.policy_train = nn.TrainOneStepCell(self.loss_net, optimizer)
        self.policy_train.set_train()

    def get_graph_values(self, obs_n, use_target_net=False):
        if use_target_net:
            hidden_states = self.policy.representation(obs_n)[0]
            utilities = self.policy.target_utility(hidden_states)
            payoff = self.policy.target_payoffs(hidden_states, self.policy.graph.edges_from, self.policy.graph.edges_to)
        else:
            hidden_states = self.policy.representation(obs_n)[0]
            utilities = self.policy.utility(hidden_states)
            payoff = self.policy.payoffs(hidden_states, self.policy.graph.edges_from, self.policy.graph.edges_to)
        return utilities, payoff

    def q_dcg(self, obs_n, actions, states=None, use_target_net=False):
        f_i, f_ij = self.get_graph_values(obs_n, use_target_net)
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

    def act(self, obs_n, episode=None, test_mode=True, noise=False):
        obs_n = Tensor(obs_n)
        f_i, f_ij = self.get_graph_values(obs_n)
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
        if len(self.policy.graph.edges) == 0:
            return utility.argmax(dim=-1).asnumpy()
        else:
            for i in range(self.args.n_msg_iterations):
                joint_forward = self.expand_dims(utility[:, self.policy.graph.edges_from, :] - Tensor(msg_ji.numpy()), -1) + f_ij_mean
                joint_backward = self.expand_dims(utility[:, self.policy.graph.edges_to, :] - Tensor(msg_ij.numpy()), -1) + f_ji_mean
                msg_ij = joint_forward.max(axis=-2)
                msg_ji = joint_backward.max(axis=-2)
                if self.args.msg_normalized:
                    msg_ij -= msg_ij.mean(axis=-1, keep_dims=True)
                    msg_ji -= msg_ji.mean(axis=-1, keep_dims=True)

                msg_forward = torch_scatter.scatter_add(src=torch.tensor(msg_ij.asnumpy()),
                                                        index=torch.tensor(self.policy.graph.edges_to.asnumpy()), dim=1,
                                                        dim_size=n_vertexes)
                msg_backward = torch_scatter.scatter_add(src=torch.tensor(msg_ji.asnumpy()),
                                                         index=torch.tensor(self.policy.graph.edges_from.asnumpy()), dim=1,
                                                         dim_size=n_vertexes)
                utility = f_i_mean + Tensor(msg_forward.numpy()) + Tensor(msg_backward.numpy())
            return utility.argmax(axis=-1).asnumpy()

    def update(self, sample):
        self.iterations += 1
        state = Tensor(sample['state'])
        obs = Tensor(sample['obs'])
        actions = Tensor(sample['actions'])
        state_next = Tensor(sample['state_next'])
        obs_next = Tensor(sample['obs_next'])
        rewards = self._mean(Tensor(sample['rewards']), 1)
        terminals = Tensor(sample['terminals']).view(-1, self.n_agents, 1)
        agent_mask = Tensor(sample['agent_mask']).view(-1, self.n_agents, 1)
        batch_size = obs.shape[0]
        IDs = ops.broadcast_to(self.expand_dims(self.eye(self.n_agents, self.n_agents, ms.float32), 0),
                               (batch_size, -1, -1))

        action_next_greedy = ms.ops.stop_gradient(Tensor(self.act(obs_next)))
        q_next_a = ms.ops.stop_gradient(self.q_dcg(obs_next, action_next_greedy, states=state_next, use_target_net=True))

        if self.args.consider_terminal_states:
            q_target = rewards + (1-terminals) * self.args.gamma * q_next_a
        else:
            q_target = rewards + self.args.gamma * q_next_a

        # calculate the loss and train
        loss = self.policy_train(state, obs, actions, q_target)
        if self.iterations % self.sync_frequency == 0:
            self.policy.copy_target()

        lr = self.scheduler(self.iterations).asnumpy()
        self.writer.add_scalar("learning_rate", lr, self.iterations)
        self.writer.add_scalar("loss_Q", loss.asnumpy(), self.iterations)
