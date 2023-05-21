"""
DCG: Deep coordination graphs
Paper link: http://proceedings.mlr.press/v119/boehmer20a/boehmer20a.pdf
Implementation: Pytorch
"""
from xuanpolicy.xuanpolicy_torch.learners import *
import torch_scatter


class DCG_Learner(LearnerMAS):
    def __init__(self,
                 config: Namespace,
                 policy: nn.Module,
                 optimizer: torch.optim.Optimizer,
                 scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
                 summary_writer: Optional[SummaryWriter] = None,
                 device: Optional[Union[int, str, torch.device]] = None,
                 modeldir: str = "./",
                 gamma: float = 0.99,
                 sync_frequency: int = 100
                 ):
        self.gamma = gamma
        self.sync_frequency = sync_frequency
        self.mse_loss = nn.MSELoss()
        super(DCG_Learner, self).__init__(config, policy, optimizer, scheduler, summary_writer, device, modeldir)

    def get_graph_values(self, obs_n, use_target_net=False):
        if use_target_net:
            hidden_states = self.policy.representation(obs_n)['state']
            utilities = self.policy.target_utility(hidden_states)
            payoff = self.policy.target_payoffs(hidden_states, self.policy.graph.edges_from, self.policy.graph.edges_to)
        else:
            hidden_states = self.policy.representation(obs_n)['state']
            utilities = self.policy.utility(hidden_states)
            payoff = self.policy.payoffs(hidden_states, self.policy.graph.edges_from, self.policy.graph.edges_to)
        return utilities, payoff

    def q_dcg(self, obs_n, actions, states=None, use_target_net=False):
        f_i, f_ij = self.get_graph_values(obs_n, use_target_net)
        f_i_mean = f_i.double() / self.policy.graph.n_vertexes
        f_ij_mean = f_ij.double() / self.policy.graph.n_edges
        utilities = f_i_mean.gather(-1, actions.unsqueeze(dim=-1).long()).sum(dim=1)
        if len(self.policy.graph.edges) == 0 or self.args.n_msg_iterations == 0:
            return utilities
        actions_ij = (actions[:, self.policy.graph.edges_from] * self.dim_act + actions[:, self.policy.graph.edges_to]).unsqueeze(-1)
        payoffs = f_ij_mean.view(list(f_ij_mean.shape[0:-2]) + [-1]).gather(-1, actions_ij.long()).sum(dim=1)
        if self.args.agent == "DCG_S":
            state_value = self.policy.bias(states)
            return utilities + payoffs + state_value
        else:
            return utilities + payoffs

    def act(self, obs_n, episode=None, test_mode=True, noise=False):
        obs_n = torch.Tensor(obs_n).to(self.device)
        with torch.no_grad():
            f_i, f_ij = self.get_graph_values(obs_n)
        n_edges = self.policy.graph.n_edges
        n_vertexes = self.policy.graph.n_vertexes
        f_i_mean = f_i.double() / n_vertexes
        f_ij_mean = f_ij.double() / n_edges
        f_ji_mean = f_ij_mean.transpose(dim0=-1, dim1=-2).clone()
        batch_size = f_i.shape[0]

        msg_ij = torch.zeros(batch_size, n_edges, self.dim_act).to(self.device)  # i -> j (send)
        msg_ji = torch.zeros(batch_size, n_edges, self.dim_act).to(self.device)  # j -> i (receive)
        #
        msg_forward = torch_scatter.scatter_add(src=msg_ij, index=self.policy.graph.edges_to, dim=1, dim_size=n_vertexes)
        msg_backward = torch_scatter.scatter_add(src=msg_ji, index=self.policy.graph.edges_from, dim=1, dim_size=n_vertexes)
        utility = f_i_mean + msg_forward + msg_backward
        if len(self.policy.graph.edges) == 0:
            return utility.argmax(dim=-1).cpu().numpy()
        else:
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
            return utility.argmax(dim=-1).cpu().numpy()

    def update(self, sample):
        self.iterations += 1
        state = torch.Tensor(sample['state']).to(self.device)
        obs = torch.Tensor(sample['obs']).to(self.device)
        actions = torch.Tensor(sample['actions']).to(self.device)
        state_next = torch.Tensor(sample['state_next']).to(self.device)
        obs_next = torch.Tensor(sample['obs_next']).to(self.device)
        rewards = torch.Tensor(sample['rewards']).mean(dim=1).to(self.device)
        terminals = torch.Tensor(sample['terminals']).float().view(-1, self.n_agents, 1).to(self.device)
        agent_mask = torch.Tensor(sample['agent_mask']).float().view(-1, self.n_agents, 1).to(self.device)
        IDs = torch.eye(self.n_agents).unsqueeze(0).expand(self.args.batch_size, -1, -1).to(self.device)

        q_eval_a = self.q_dcg(obs, actions, states=state, use_target_net=False)
        with torch.no_grad():
            action_next_greedy = torch.Tensor(self.act(obs_next.cpu())).to(self.device)
            q_next_a = self.q_dcg(obs_next, action_next_greedy, states=state_next, use_target_net=True)

        if self.args.consider_terminal_states:
            q_target = rewards + (1-terminals) * self.args.gamma * q_next_a
        else:
            q_target = rewards + self.args.gamma * q_next_a

        # calculate the loss function
        loss = self.mse_loss(q_eval_a, q_target.detach())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        if self.scheduler is not None:
            self.scheduler.step()

        if self.iterations % self.sync_frequency == 0:
            self.policy.copy_target()
        lr = self.optimizer.state_dict()['param_groups'][0]['lr']
        self.writer.add_scalar("learning_rate", lr, self.iterations)
        self.writer.add_scalar("loss_Q", loss.item(), self.iterations)
        self.writer.add_scalar("predictQ", q_eval_a.mean().item(), self.iterations)
