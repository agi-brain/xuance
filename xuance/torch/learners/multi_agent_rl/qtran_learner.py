"""
QTRAN: Learning to Factorize with Transformation for Cooperative Multi-Agent Reinforcement Learning
Paper link:
http://proceedings.mlr.press/v97/son19a/son19a.pdf
Implementation: Pytorch
"""
import torch
from torch import nn
from argparse import Namespace
from operator import itemgetter
from xuance.common import AgentGrouping
from xuance.torch.learners import OffPolicyMultiAgentLearner


class QTRAN_Learner(OffPolicyMultiAgentLearner):
    def __init__(self,
                 config: Namespace,
                 agent_grouping: AgentGrouping,
                 model: nn.Module,
                 callback):
        super(QTRAN_Learner, self).__init__(config, agent_grouping, model, callback)
        self.sync_frequency = config.sync_frequency
        self.n_actions = {k: self.model.individual_q_networks[k].action_space.n for k in self.group_keys}

    def update(self, sample):
        if self.use_rnn:
            return self.update_rnn(sample)

        self.iterations += 1

        # prepare training data
        batch = self.build_training_data(sample=sample,
                                         use_parameter_sharing=self.use_parameter_sharing,
                                         use_actions_mask=self.use_actions_mask,
                                         use_global_state=True,
                                         use_shared_rewards=True)
        batch_size = batch.batch_size
        state = batch.global_states
        state_next = batch.next_global_states
        obs = batch.observations
        actions = batch.actions
        obs_next = batch.next_observations
        rewards = batch.rewards
        terminals = batch.terminals
        agent_mask = batch.agent_masks
        avail_actions = batch.avail_actions
        avail_actions_next = batch.next_avail_actions
        agent_indices = batch.agent_indices

        if not self.agent_grouping.full_independent:
            obs = self.packed_tensor(obs)
            obs_next = self.packed_tensor(obs_next)

        rewards_tot = torch.stack(itemgetter(*self.agent_keys)(rewards), dim=1).mean(dim=-1, keepdim=True)
        terminals_tot = torch.stack(itemgetter(*self.agent_keys)(terminals), dim=1).all(dim=1, keepdim=True).float()

        info = self.callback.on_update_start(self.iterations, method="update",
                                             model=self.model, batch=batch,
                                             rewards_tot=rewards_tot, terminals_tot=terminals_tot)

        model_output = self.model(observations=obs,
                                  agent_indices=agent_indices,
                                  avail_actions=avail_actions)
        actions_greedy, q_eval = model_output.actions, model_output.values
        hidden_state = {k: v.embeddings for k, v in model_output.rep_out.items()}

        target_model_output = self.model.Qtarget(observations=obs_next,
                                                 agent_indices=agent_indices)
        q_next = target_model_output.values
        hidden_state_next = {k: v.embeddings for k, v in target_model_output.rep_out.items()}

        if self.config.double_q:
            a_next_greedy = self.model(observations=obs_next,
                                       agent_indices=agent_indices,
                                       avail_actions=avail_actions_next).actions
        else:
            a_next_greedy = {}

        q_eval_a, q_eval_greedy_a, q_next_a = {}, {}, {}
        for key in self.agent_keys:
            mask_values = agent_mask[key]
            q_eval_a[key] = q_eval[key].gather(-1, actions[key].long().unsqueeze(-1)).reshape(batch_size)
            q_eval_greedy_a[key] = q_eval[key].gather(-1, actions_greedy[key].long().unsqueeze(-1)).reshape(batch_size)

            if self.use_actions_mask:
                q_next[key][avail_actions_next[key] == 0] = -1e10

            if self.config.double_q:
                q_next_a[key] = q_next[key].gather(-1, a_next_greedy[key].long().unsqueeze(-1)).reshape(batch_size)
            else:
                a_next_greedy[key] = q_next[key].argmax(dim=-1, keepdim=False)
                q_next_a[key] = q_next[key].max(dim=-1, keepdim=True).values.reshape(batch_size)

            q_eval_a[key] *= mask_values
            q_eval_greedy_a[key] *= mask_values
            q_next_a[key] *= mask_values

            info.update(self.callback.on_update_agent_wise(self.iterations, key, info=info, method="update",
                                                           mask_values=mask_values, q_eval_a=q_eval_a,
                                                           q_eval_greedy_a=q_eval_greedy_a))

        if self.config.agent == "QTRAN_base":
            # -- TD Loss --
            q_joint, v_joint = self.model.Q_tran(state, hidden_state, actions, agent_mask)
            q_joint_next, _ = self.model.Q_tran_target(state_next, hidden_state_next, a_next_greedy, agent_mask)

            y_dqn = rewards_tot + (1 - terminals_tot) * self.gamma * q_joint_next
            loss_td = self.mse_loss(q_joint, y_dqn.detach())  # TD loss

            # -- Opt Loss --
            # Argmax across the current agents' actions
            q_tot_greedy = self.model.Q_tot(q_eval_greedy_a)
            q_joint_greedy_hat, _ = self.model.Q_tran(state, hidden_state, actions_greedy, agent_mask)
            error_opt = q_tot_greedy - q_joint_greedy_hat.detach() + v_joint
            loss_opt = torch.mean(error_opt ** 2)  # Opt loss

            # -- Nopt Loss --
            q_tot = self.model.Q_tot(q_eval_a)
            q_joint_hat = q_joint
            error_nopt = q_tot - q_joint_hat.detach() + v_joint
            error_nopt = error_nopt.clamp(max=0)
            loss_nopt = torch.mean(error_nopt ** 2)  # NOPT loss

            info["Q_joint"] = q_joint.mean().item()

        elif self.config.agent == "QTRAN_alt":
            # -- TD Loss -- (Computed for all agents)
            q_count, v_joint = self.model.Q_tran(state, hidden_state, actions, agent_mask)
            actions_choosen = torch.stack([actions[k] for k in self.agent_keys], dim=1)
            actions_choosen = actions_choosen.reshape(-1, self.n_agents, 1)
            q_joint_choosen = q_count.gather(-1, actions_choosen.long()).reshape(-1, self.n_agents)
            q_next_count, _ = self.model.Q_tran_target(state_next, hidden_state_next, a_next_greedy, agent_mask)
            actions_next_choosen = torch.stack([a_next_greedy[k] for k in self.agent_keys], dim=1)
            actions_next_choosen = actions_next_choosen.reshape(-1, self.n_agents, 1)
            q_joint_next_choosen = q_next_count.gather(-1, actions_next_choosen.long()).reshape(-1, self.n_agents)

            y_dqn = rewards_tot + (1 - terminals_tot) * self.gamma * q_joint_next_choosen
            loss_td = self.mse_loss(q_joint_choosen, y_dqn.detach())  # TD loss

            # -- Opt Loss -- (Computed for all agents)
            q_tot_greedy = self.model.Q_tot(q_eval_greedy_a)
            q_joint_greedy_hat, _ = self.model.Q_tran(state, hidden_state, actions_greedy, agent_mask)
            actions_greedy_current = torch.stack([actions_greedy[k] for k in self.agent_keys], dim=1)
            actions_greedy_current = actions_greedy_current.reshape(-1, self.n_agents, 1)
            q_joint_greedy_hat_all = q_joint_greedy_hat.gather(
                -1, actions_greedy_current.long()).reshape(-1, self.n_agents)
            error_opt = q_tot_greedy - q_joint_greedy_hat_all.detach() + v_joint
            loss_opt = torch.mean(error_opt ** 2)  # Opt loss

            # -- Nopt Loss --
            q_eval_count = torch.stack([q_eval[k] for k in self.agent_keys], dim=1)
            q_eval_count = q_eval_count.reshape(batch_size * self.n_agents, -1)
            q_sums = torch.stack([q_eval_a[k] for k in self.agent_keys], dim=1).reshape(-1, self.n_agents)
            q_sums_repeat = q_sums.unsqueeze(dim=1).repeat(1, self.n_agents, 1)
            agent_mask_diag = (1 - torch.eye(self.n_agents, dtype=torch.float32,
                                             device=self.device)).unsqueeze(0).repeat(batch_size, 1, 1)
            q_sum_mask = (q_sums_repeat * agent_mask_diag).sum(dim=-1)
            q_count_for_nopt = q_count.view(batch_size * self.n_agents, -1)
            v_joint_repeated = v_joint.repeat(1, self.n_agents).view(-1, 1)
            error_nopt = q_eval_count + q_sum_mask.view(-1, 1) - q_count_for_nopt.detach() + v_joint_repeated
            error_nopt_min = torch.min(error_nopt, dim=-1).values
            loss_nopt = torch.mean(error_nopt_min ** 2)  # NOPT loss

            info["Q_joint"] = q_joint_choosen.mean().item()

        else:
            raise ValueError("Mixer {} not recognised.".format(self.config.agent))

        # calculate the loss function
        loss = loss_td + self.config.lambda_opt * loss_opt + self.config.lambda_nopt * loss_nopt
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        if self.scheduler is not None:
            self.scheduler.step()

        if self.iterations % self.sync_frequency == 0:
            self.model.copy_target()
        lr = self.optimizer.state_dict()['param_groups'][0]['lr']

        info.update({
            "learning_rate": lr,
            "loss_td": loss_td.item(),
            "loss_opt": loss_opt.item(),
            "loss_nopt": loss_nopt.item(),
            "loss": loss.item()
        })

        info.update(self.callback.on_update_end(self.iterations, method="update", model=self.model, info=info,
                                                v_joint=v_joint, y_dqn=y_dqn, q_tot_greedy=q_tot_greedy,
                                                q_joint_greedy_hat=q_joint_greedy_hat, error_opt=error_opt,
                                                error_nopt=error_nopt))

        return info

    def update_rnn(self, sample):
        self.iterations += 1

        # prepare training data
        batch = self.build_training_data(sample=sample,
                                         use_parameter_sharing=self.use_parameter_sharing,
                                         use_actions_mask=self.use_actions_mask,
                                         use_global_state=True,
                                         use_shared_rewards=True)
        batch_size = batch.batch_size
        seq_len = batch.seq_length
        state = batch.global_states
        obs = batch.observations
        actions = batch.actions
        rewards = batch.rewards
        terminals = batch.terminals
        agent_mask = batch.agent_masks
        avail_actions = batch.avail_actions
        filled = batch.filled_masks.reshape([-1, 1])
        filled_n = filled.repeat(1, self.n_agents)
        agent_indices = batch.agent_indices

        if not self.agent_grouping.full_independent:
            obs = self.packed_tensor(obs)

        rewards_tot = torch.stack(itemgetter(*self.agent_keys)(rewards), dim=1).mean(dim=1).reshape(-1, 1)
        terminals_tot = torch.stack(itemgetter(*self.agent_keys)(terminals), dim=1).all(1).reshape([-1, 1]).float()

        info = self.callback.on_update_start(self.iterations, method="update_rnn",
                                             model=self.model, batch=batch,
                                             rewards_tot=rewards_tot, terminals_tot=terminals_tot)

        rnn_states = self.model.init_rnn_states(batch_size)
        model_output = self.model(observations=obs, agent_indices=agent_indices, avail_actions=avail_actions,
                                  rnn_states=rnn_states)
        actions_greedy, q_eval = model_output.actions, model_output.values
        hidden_state = {k: v.embeddings for k, v in model_output.rep_out.items()}

        target_model_output = self.model.Qtarget(observations=obs, agent_indices=agent_indices, rnn_states=rnn_states)
        q_next_seq = target_model_output.values
        hidden_state_next = {k: v.embeddings for k, v in target_model_output.rep_out.items()}

        q_eval_a, q_eval_greedy_a, q_next, q_next_a = {}, {}, {}, {}
        actions_greedy_eval, actions_next_greedy = {}, {}
        for key in self.agent_keys:
            mask_values = agent_mask[key]
            hidden_state[key] = hidden_state[key][:, :-1]
            hidden_state_next[key] = hidden_state_next[key][:, :-1]
            actions_greedy_eval[key] = actions_greedy[key][:, :-1]
            q_eval_a[key] = q_eval[key][:, :-1].gather(-1, actions[key].long().unsqueeze(-1)).reshape(
                batch_size, seq_len)
            q_eval_greedy_a[key] = q_eval[key][:, :-1].gather(
                -1, actions_greedy[key][:, :-1].long().unsqueeze(-1)).reshape(batch_size, seq_len)
            q_next[key] = q_next_seq[key][:, 1:]

            if self.use_actions_mask:
                q_next[key][avail_actions[key][:, 1:] == 0] = -1e10

            if self.config.double_q:
                act_next = actions_greedy[key][:, 1:]
                q_next_a[key] = q_next[key].gather(-1, act_next.long().unsqueeze(-1)).reshape(batch_size, seq_len)
                actions_next_greedy[key] = act_next
            else:
                actions_next_greedy[key] = q_next[key].argmax(dim=-1, keepdim=False)
                q_next_a[key] = q_next[key].max(dim=-1, keepdim=True).values.reshape(batch_size, seq_len)

            q_eval_a[key] *= mask_values
            q_eval_greedy_a[key] *= mask_values
            q_next_a[key] *= mask_values

            info.update(self.callback.on_update_agent_wise(self.iterations, key, info=info, method="update_rnn",
                                                           mask_values=mask_values, q_eval_a=q_eval_a,
                                                           q_eval_greedy_a=q_eval_greedy_a, q_next=q_next,
                                                           q_next_a=q_next_a))

        if self.config.agent == "QTRAN_base":
            # -- TD Loss --
            q_joint, v_joint = self.model.Q_tran(state[:, :-1], hidden_state, actions, agent_mask)
            q_joint_next, _ = self.model.Q_tran_target(state[:, 1:], hidden_state_next,
                                                       actions_next_greedy, agent_mask)
            y_dqn = rewards_tot + (1 - terminals_tot) * self.gamma * q_joint_next
            td_error = (q_joint - y_dqn.detach()) * filled
            loss_td = (td_error ** 2).sum() / filled.sum()  # TD loss

            # -- Opt Loss --
            # Argmax across the current agents' actions
            q_tot_greedy = self.model.Q_tot(q_eval_greedy_a)
            q_joint_greedy_hat, _ = self.model.Q_tran(state[:, :-1], hidden_state, actions_greedy_eval, agent_mask)
            error_opt = (q_tot_greedy - q_joint_greedy_hat.detach() + v_joint) * filled
            loss_opt = (error_opt ** 2).sum() / filled.sum()  # Opt loss

            # -- Nopt Loss --
            q_tot = self.model.Q_tot(q_eval_a)
            q_joint_hat = q_joint
            error_nopt = q_tot - q_joint_hat.detach() + v_joint
            error_nopt = error_nopt.clamp(max=0) * filled
            loss_nopt = (error_nopt ** 2).sum() / filled.sum()  # NOPT loss

            info["Q_joint"] = q_joint.mean().item()

        elif self.config.agent == "QTRAN_alt":
            # -- TD Loss -- (Computed for all agents)
            q_count, v_joint = self.model.Q_tran(state[:, :-1], hidden_state, actions, agent_mask)
            actions_choosen = torch.stack([actions[k] for k in self.agent_keys], dim=2)
            actions_choosen = actions_choosen.reshape(-1, self.n_agents, 1)
            q_joint_choosen = q_count.gather(-1, actions_choosen.long()).reshape(-1, self.n_agents)
            q_next_count, _ = self.model.Q_tran_target(state[:, 1:], hidden_state_next, actions_next_greedy, agent_mask)
            actions_next_choosen = torch.stack([actions_next_greedy[k] for k in self.agent_keys], dim=2)

            actions_next_choosen = actions_next_choosen.reshape(-1, self.n_agents, 1)
            q_joint_next_choosen = q_next_count.gather(-1, actions_next_choosen.long()).reshape(-1, self.n_agents)

            y_dqn = rewards_tot + (1 - terminals_tot) * self.gamma * q_joint_next_choosen
            td_errors = (q_joint_choosen - y_dqn.detach()) * filled_n
            loss_td = (td_errors ** 2).sum() / filled_n.sum()  # TD loss

            # -- Opt Loss -- (Computed for all agents)
            q_tot_greedy = self.model.Q_tot(q_eval_greedy_a)
            q_joint_greedy_hat, _ = self.model.Q_tran(state[:, :-1], hidden_state, actions_greedy_eval, agent_mask)
            actions_greedy_current = torch.stack([actions_greedy_eval[k] for k in self.agent_keys], dim=2)
            actions_greedy_current = actions_greedy_current.reshape(-1, self.n_agents, 1)
            q_joint_greedy_hat_all = q_joint_greedy_hat.gather(
                -1, actions_greedy_current.long()).reshape(-1, self.n_agents)
            error_opt = (q_tot_greedy - q_joint_greedy_hat_all.detach() + v_joint) * filled_n
            loss_opt = (error_opt ** 2).sum() / filled_n.sum()  # Opt loss

            # -- Nopt Loss --
            q_eval_count = torch.stack([q_eval[k][:, :-1] for k in self.agent_keys],
                                       dim=2).reshape(batch_size, seq_len, self.n_agents, -1)
            q_eval_count = q_eval_count.reshape(batch_size * seq_len * self.n_agents, -1)
            q_sums = torch.stack([q_eval_a[k] for k in self.agent_keys], dim=2).reshape(-1, seq_len, self.n_agents)
            q_sums = q_sums.reshape(batch_size * seq_len, self.n_agents)
            q_sums_repeat = q_sums.unsqueeze(dim=1).repeat(1, self.n_agents, 1)
            agent_mask_diag = (1 - torch.eye(self.n_agents, dtype=torch.float32,
                                             device=self.device)).unsqueeze(0).repeat(batch_size * seq_len, 1, 1)
            q_sum_mask = (q_sums_repeat * agent_mask_diag).sum(dim=-1)
            q_count_for_nopt = q_count.view(batch_size * seq_len * self.n_agents, -1)
            v_joint_repeated = v_joint.repeat(1, self.n_agents).view(-1, 1)
            error_nopt = q_eval_count + q_sum_mask.view(-1, 1) - q_count_for_nopt.detach() + v_joint_repeated
            error_nopt_min = torch.min(error_nopt, dim=-1).values * filled_n.reshape(-1)
            loss_nopt = (error_nopt_min ** 2).sum() / filled_n.sum()  # NOPT loss

            info["Q_joint"] = q_joint_choosen.mean().item()

        else:
            raise ValueError("Mixer {} not recognised.".format(self.config.agent))

        # calculate the loss function
        loss = loss_td + self.config.lambda_opt * loss_opt + self.config.lambda_nopt * loss_nopt
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        if self.scheduler is not None:
            self.scheduler.step()

        if self.iterations % self.sync_frequency == 0:
            self.model.copy_target()
        lr = self.optimizer.state_dict()['param_groups'][0]['lr']

        info.update({
            "learning_rate": lr,
            "loss_td": loss_td.item(),
            "loss_opt": loss_opt.item(),
            "loss_nopt": loss_nopt.item(),
            "loss": loss.item()
        })

        info.update(self.callback.on_update_end(self.iterations, method="update_rnn", model=self.model, info=info,
                                                v_joint=v_joint, y_dqn=y_dqn, q_tot_greedy=q_tot_greedy,
                                                q_joint_greedy_hat=q_joint_greedy_hat, error_opt=error_opt,
                                                error_nopt=error_nopt))

        return info
