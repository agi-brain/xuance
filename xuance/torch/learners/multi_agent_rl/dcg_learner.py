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
        self.use_rnn = config.use_rnn
        self.sync_frequency = sync_frequency
        self.dim_hidden_state = policy.representation.output_shapes['state'][0]
        self.mse_loss = nn.MSELoss()
        super(DCG_Learner, self).__init__(config, policy, optimizer, scheduler, device, model_dir)

    def get_hidden_states(self, obs_n, *rnn_hidden, use_target_net=False):
        if self.use_rnn:
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
