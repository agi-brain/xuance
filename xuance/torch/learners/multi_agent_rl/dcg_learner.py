"""
DCG: Deep coordination graphs
Paper link: http://proceedings.mlr.press/v119/boehmer20a/boehmer20a.pdf
Implementation: Pytorch
"""
from operator import itemgetter

import torch
from torch import nn
from xuance.torch.learners import LearnerMAS
from xuance.common import List
from argparse import Namespace
import torch_scatter


class DCG_Learner(LearnerMAS):
    def __init__(self,
                 config: Namespace,
                 model_keys: List[str],
                 agent_keys: List[str],
                 policy: nn.Module):
        self.gamma = config.gamma
        self.sync_frequency = config.sync_frequency
        self.mse_loss = nn.MSELoss()
        super(DCG_Learner, self).__init__(config, model_keys, agent_keys, policy)
        self.optimizer = torch.optim.Adam(self.policy.parameters_model, config.learning_rate, eps=1e-5)
        self.scheduler = torch.optim.lr_scheduler.LinearLR(self.optimizer, start_factor=1.0, end_factor=0.5,
                                                           total_iters=self.config.running_steps)
        self.dim_hidden_state = policy.representation[self.model_keys[0]].output_shapes['state'][0]
        self.dim_act = max([self.policy.action_space[key].n for key in agent_keys])

    def get_graph_values(self, hidden_states, use_target_net=False):
        if use_target_net:
            utilities = self.policy.target_utility(hidden_states)
            payoff = self.policy.target_payoffs(hidden_states, self.policy.graph.edges_from, self.policy.graph.edges_to)
        else:
            utilities = self.policy.utility(hidden_states)
            payoff = self.policy.payoffs(hidden_states, self.policy.graph.edges_from, self.policy.graph.edges_to)
        return utilities, payoff

    def act(self, hidden_states, avail_actions=None):
        """
        Calculate the actions via belief propagation.

        Args:
            hidden_states (torch.Tensor): The hidden states for the representation of all agents.
            avail_actions (torch.Tensor): The avail actions for the agents, default is None.

        Returns: The actions.
        """
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
            for i in range(self.config.n_msg_iterations):
                joint_forward = (utility[:, self.policy.graph.edges_from, :] - msg_ji).unsqueeze(dim=-1) + f_ij_mean
                joint_backward = (utility[:, self.policy.graph.edges_to, :] - msg_ij).unsqueeze(dim=-1) + f_ji_mean
                msg_ij = joint_forward.max(dim=-2).values
                msg_ji = joint_backward.max(dim=-2).values
                if self.config.msg_normalized:
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
        if len(self.policy.graph.edges) == 0 or self.config.n_msg_iterations == 0:
            return utilities
        actions_ij = (actions[:, self.policy.graph.edges_from] * self.dim_act + \
                      actions[:, self.policy.graph.edges_to]).unsqueeze(-1)
        payoffs = f_ij_mean.reshape(list(f_ij_mean.shape[0:-2]) + [-1]).gather(-1, actions_ij.long()).sum(dim=1)
        if self.config.agent == "DCG_S":
            state_value = self.policy.bias(states)
            return utilities + payoffs + state_value
        else:
            return utilities + payoffs

    def update(self, sample):
        self.iterations += 1
        info = {}

        # prepare training data
        sample_Tensor = self.build_training_data(sample=sample,
                                                 use_parameter_sharing=self.use_parameter_sharing,
                                                 use_actions_mask=self.use_actions_mask,
                                                 use_global_state=True if self.config.agent=="DCG_S" else False)
        batch_size = sample_Tensor['batch_size']
        state = sample_Tensor['state']
        state_next = sample_Tensor['state_next']
        obs = sample_Tensor['obs']
        actions = sample_Tensor['actions']
        obs_next = sample_Tensor['obs_next']
        rewards = sample_Tensor['rewards']
        terminals = sample_Tensor['terminals']
        agent_mask = sample_Tensor['agent_mask']
        avail_actions = sample_Tensor['avail_actions']
        avail_actions_next = sample_Tensor['avail_actions_next']
        IDs = sample_Tensor['agent_ids']

        if self.use_parameter_sharing:
            key = self.model_keys[0]
            bs = batch_size * self.n_agents
            rewards_tot = rewards[key].mean(dim=1).reshape(batch_size, 1)
            terminals_tot = terminals[key].all(dim=1, keepdim=False).float().reshape(batch_size, 1)
            actions = actions[key].reshape(batch_size, self.n_agents)
        else:
            bs = batch_size
            rewards_tot = torch.stack(itemgetter(*self.agent_keys)(rewards), dim=1).mean(dim=-1, keepdim=True)
            terminals_tot = torch.stack(itemgetter(*self.agent_keys)(terminals), dim=1).all(dim=1, keepdim=True).float()
            actions = itemgetter(*self.agent_keys)(actions).reshape(batch_size, self.n_agents)

        _, hidden_states = self.policy.get_hidden_states(batch_size, obs, use_target_net=False)
        q_tot_eval = self.q_dcg(hidden_states, actions, states=state, use_target_net=False)

        _, hidden_states_next = self.policy.get_hidden_states(batch_size, obs_next, use_target_net=False)
        action_next_greedy = torch.Tensor(self.act(hidden_states_next)).to(self.device)
        _, hidden_states_target = self.policy.get_hidden_states(batch_size, obs_next, use_target_net=True)
        q_tot_next = self.q_dcg(hidden_states_target, action_next_greedy, states=state_next, use_target_net=True)

        q_tot_target = rewards_tot + (1 - terminals_tot) * self.gamma * q_tot_next

        # calculate the loss function
        loss = self.mse_loss(q_tot_eval, q_tot_target.detach())
        self.optimizer.zero_grad()
        loss.backward()
        if self.use_grad_clip:
            torch.nn.utils.clip_grad_norm_(self.policy.parameters_model, self.grad_clip_norm)
        self.optimizer.step()
        if self.scheduler is not None:
            self.scheduler.step()

        lr = self.optimizer.state_dict()['param_groups'][0]['lr']

        info = {
            "learning_rate": lr,
            "loss_Q": loss.item(),
            "predictQ": q_tot_eval.mean().item()
        }

        if self.iterations % self.sync_frequency == 0:
            self.policy.copy_target()
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
        _, hidden_states = self.policy.get_hidden_states(batch_size, obs.reshape(-1, episode_length + 1, self.dim_obs),
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
            _, hidden_states_tar = self.policy.get_hidden_states(batch_size, obs[:, :, 1:].reshape(-1, episode_length, self.dim_obs),
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
