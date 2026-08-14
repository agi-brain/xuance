"""
DCG: Deep coordination graphs
Paper link: http://proceedings.mlr.press/v119/boehmer20a/boehmer20a.pdf
Implementation: Pytorch
"""
import torch
from torch import nn, Tensor
from argparse import Namespace
from xuance.common import Optional, AgentGrouping
from xuance.torch.learners import OffPolicyMultiAgentLearner
from xuance.torch.rl_models.modules import OffPolicyMARLBatch

try:
    import torch_scatter
except ImportError:
    print("The module torch_scatter is not installed.")


class DCG_Learner(OffPolicyMultiAgentLearner):
    def __init__(self,
                 config: Namespace,
                 agent_grouping: AgentGrouping,
                 model: nn.Module,
                 callback):
        super(DCG_Learner, self).__init__(config, agent_grouping, model, callback)
        self.dim_hidden_state = model.representation[self.group_keys[0]].output_shapes['state'][0]
        self.dim_act = max([self.model.action_space[key].n for key in self.agent_keys])
        self.sync_frequency = config.sync_frequency

    def build_training_data(
            self,
            sample: Optional[dict],
            use_parameter_sharing: Optional[bool] = False,
            use_actions_mask: Optional[bool] = False,
            use_global_state: Optional[bool] = False,
            use_shared_rewards: Optional[bool] = False,
    ) -> OffPolicyMARLBatch:
        """
        Prepare the training data.

        Parameters:
            sample (dict): The raw sampled data.
            use_parameter_sharing (bool): Whether to use parameter sharing for individual agent models.
            use_actions_mask (bool): Whether to use actions mask for unavailable actions.
            use_global_state (bool): Whether to use global state.
            use_shared_rewards (bool)： Whether to use shared rewards for each agent.

        Returns:
            OffPolicyMARLBatch: The formatted sampled data.
        """
        batch_size = sample['batch_size']
        seq_length = sample['sequence_length'] if self.use_rnn else 1
        state, state_next, filled = None, None, None
        obs, actions, rewards, terminals, agent_mask = {}, {}, {}, {}, {}
        avail_actions = {} if use_actions_mask else None
        agent_indices = {}
        if self.use_rnn:
            obs_next, avail_actions_next = None, None
        else:
            obs_next = {}
            avail_actions_next = {} if use_actions_mask else None

        for agent in self.agent_keys:
            obs[agent] = torch.as_tensor(sample['obs'][agent], device=self.device)
            if not self.use_rnn:
                obs_next[agent] = torch.as_tensor(sample['obs_next'][agent], device=self.device)
            actions[agent] = torch.as_tensor(sample['actions'][agent], device=self.device)
            rewards[agent] = torch.as_tensor(sample['rewards'][agent], device=self.device)
            terminals[agent] = torch.as_tensor(sample['terminals'][agent], device=self.device, dtype=torch.float32)
            agent_mask[agent] = torch.as_tensor(sample['agent_mask'][agent], device=self.device, dtype=torch.float32)
            if use_actions_mask:
                avail_actions[agent] = torch.as_tensor(sample['avail_actions'][agent],
                                                       device=self.device, dtype=torch.float32)
                if not self.use_rnn:
                    avail_actions_next[agent] = torch.as_tensor(sample['avail_actions_next'][agent],
                                                                device=self.device, dtype=torch.float32)

        if use_global_state:
            state = torch.as_tensor(sample['state'], device=self.device)
            if not self.use_rnn:
                state_next = torch.as_tensor(sample['state_next'], device=self.device)

        if self.use_rnn:
            filled = torch.as_tensor(sample['filled'], device=self.device, dtype=torch.float32)

        for key, n_agents in self.n_group_agents.items():
            bs = batch_size * n_agents

            if self.use_rnn:
                agents_id = torch.as_tensor(self.agent_grouping.agent_indices(key), dtype=torch.int64).repeat(
                    batch_size, 1).reshape(bs, 1, 1).expand(-1, seq_length + 1, -1).to(self.device)
            else:
                agents_id = torch.as_tensor(self.agent_indices[key], dtype=torch.int64).repeat(
                    batch_size, 1).reshape([bs, 1]).to(self.device)

            agent_indices[key] = agents_id

        if not self.agent_grouping.full_independent:
            obs = self.packed_tensor(obs)
            obs_next = self.packed_tensor(obs_next)
            if not use_shared_rewards:
                rewards = self.packed_tensor(rewards)
                terminals = self.packed_tensor(terminals)
            agent_mask = self.packed_tensor(agent_mask)

        return OffPolicyMARLBatch(
            batch_size=batch_size,
            global_states=state,
            next_global_states=state_next,
            observations=obs,
            actions=actions,
            next_observations=obs_next,
            rewards=rewards,
            terminals=terminals,
            agent_masks=agent_mask,
            avail_actions=avail_actions,
            next_avail_actions=avail_actions_next,
            agent_indices=agent_indices,
            filled_masks=filled,
            seq_length=seq_length,
        )

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
        f_i_mean = f_i.double() / self.model.graph.n_vertexes
        f_ij_mean = f_ij.double() / self.model.graph.n_edges
        utilities = f_i_mean.gather(-1, actions.unsqueeze(dim=-1).long()).sum(dim=1)
        if len(self.model.graph.edges) == 0 or self.config.n_msg_iterations == 0:
            return utilities
        actions_ij = (actions[:, self.model.graph.edges_from] * self.dim_act + \
                      actions[:, self.model.graph.edges_to]).unsqueeze(-1)
        payoffs = f_ij_mean.reshape(list(f_ij_mean.shape[0:-2]) + [-1]).gather(-1, actions_ij.long()).sum(dim=1)
        if self.config.agent == "DCG_S":
            state_value = self.model.bias(states).unsqueeze(-1)
            return utilities + payoffs + state_value
        else:
            return utilities + payoffs

    def _forward_transitions(self, batch: OffPolicyMARLBatch):
        batch_size = batch.batch_size
        actions = torch.stack([batch.actions[k] for k in self.agent_keys], dim=-1)

        rnn_states = self.model.init_rnn_states(batch_size)

        _, hidden_states = self.model.get_hidden_states(observations=batch.observations,
                                                        agent_indices=batch.agent_indices,
                                                        rnn_states=rnn_states,
                                                        use_target_net=False)
        if self.use_rnn:
            seq_len = batch.seq_length
            if self.config.agent == "DCG_S":
                state_current = batch.global_states[:, :-1].reshape(batch_size * seq_len, -1)
                state_next = batch.global_states[:, 1:].reshape(batch_size * seq_len, -1)
            else:
                state_current, state_next = None, None
            q_tot_eval = self.q_dcg(hidden_states[:, :-1].reshape(batch_size * seq_len, self.n_agents, -1),
                                    actions.reshape(batch_size * seq_len, self.n_agents),
                                    states=state_current, use_target_net=False)

            if self.use_actions_mask:
                avail_actions = torch.stack([batch.avail_actions[k] for k in self.agent_keys], dim=-2)
                avail_a_next = avail_actions[:, 1:].reshape(batch_size * seq_len, self.n_agents, -1)
            else:
                avail_a_next = None
            hidden_states_next = hidden_states[:, 1:].reshape(batch_size * seq_len, self.n_agents, -1)
            action_next_greedy = torch.Tensor(self.act(hidden_states_next, avail_actions=avail_a_next)).to(self.device)
            _, hidden_states_tar = self.model.get_hidden_states(observations=batch.observations,
                                                                agent_indices=batch.agent_indices,
                                                                rnn_states=rnn_states,
                                                                use_target_net=True)
            q_tot_next = self.q_dcg(hidden_states_tar[:, 1:].reshape(batch_size * seq_len, self.n_agents, -1),
                                    action_next_greedy, states=state_next, use_target_net=True)
        else:
            if self.use_actions_mask:
                avail_actions_next = torch.stack([batch.next_avail_actions[k] for k in self.agent_keys], dim=-2)
            else:
                avail_actions_next = None

            q_tot_eval = self.q_dcg(hidden_states, actions, states=batch.global_states, use_target_net=False)

            _, hidden_states_next = self.model.get_hidden_states(observations=batch.next_observations,
                                                                 agent_indices=batch.agent_indices,
                                                                 use_target_net=False)
            action_next_greedy = torch.Tensor(self.act(hidden_states_next, avail_actions_next)).to(self.device)
            _, hidden_states_target = self.model.get_hidden_states(observations=batch.next_observations,
                                                                   agent_indices=batch.agent_indices,
                                                                   use_target_net=True)

            q_tot_next = self.q_dcg(hidden_states_target, action_next_greedy,
                                    states=batch.next_global_states, use_target_net=True)

        return q_tot_eval, q_tot_next

    def update(self, sample):
        self.iterations += 1

        # prepare training data
        batch = self.build_training_data(
            sample=sample,
            use_parameter_sharing=self.use_parameter_sharing,
            use_actions_mask=self.use_actions_mask,
            use_global_state=True if self.config.agent == "DCG_S" else False,
            use_shared_rewards=True
        )

        rewards_tot = torch.stack([r for r in batch.rewards.values()], dim=1).mean(dim=1)
        terminals_tot = torch.stack([d for d in batch.terminals.values()], dim=1).all(dim=1).float()

        info = self.callback.on_update_start(self.iterations, model=self.model, batch=batch,
                                             rewards_tot=rewards_tot, terminals_tot=terminals_tot)

        # feedforward
        q_tot_eval, q_tot_next = self._forward_transitions(batch)

        # calculate target value
        q_tot_eval = q_tot_eval.reshape(-1)
        q_tot_next = q_tot_next.reshape(-1)
        rewards_tot = rewards_tot.reshape(-1)
        terminals_tot = terminals_tot.reshape(-1)
        q_tot_target = rewards_tot + (1 - terminals_tot) * self.gamma * q_tot_next

        # calculate the loss
        if self.use_rnn:
            filled = batch.filled_masks.reshape(-1)
            td_error = (q_tot_eval - q_tot_target.detach()) * filled
            loss = (td_error ** 2).sum() / filled.sum()
        else:
            loss = self.mse_loss(q_tot_eval, q_tot_target.detach())

        # update the networks
        self.optimizer.zero_grad()
        loss.backward()
        if self.use_grad_clip:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)
        self.optimizer.step()
        if self.scheduler is not None:
            self.scheduler.step()

        lr = self.optimizer.state_dict()['param_groups'][0]['lr']

        info.update({
            "learning_rate": lr,
            "loss_Q": loss.item(),
            "predictQ": q_tot_eval.mean().item()
        })

        if self.iterations % self.sync_frequency == 0:
            self.model.copy_target()

        info.update(self.callback.on_update_end(self.iterations, model=self.model, info=info,
                                                q_tot_eval=q_tot_eval, q_tot_next=q_tot_next, q_tot_target=q_tot_target,
                                                loss=loss))

        return info
