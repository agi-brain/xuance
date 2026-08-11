"""
MFQ: Mean Field Q-Learning
Paper link:
http://proceedings.mlr.press/v80/yang18d/yang18d.pdf
Implementation: Pytorch
"""
import torch
from xuance.torch import Module
from xuance.torch.learners import LearnerMAS
from xuance.common import AgentGrouping, Optional
from argparse import Namespace


class MFQ_Learner(LearnerMAS):
    def __init__(self,
                 config: Namespace,
                 agent_grouping: AgentGrouping,
                 model: Module,
                 callback):
        super(MFQ_Learner, self).__init__(config, agent_grouping, model, callback)
        self.optimizer = torch.optim.Adam(self.model.parameters_model, config.learning_rate, eps=1e-5)
        self.scheduler = torch.optim.lr_scheduler.LinearLR(self.optimizer,
                                                           start_factor=1.0,
                                                           end_factor=self.end_factor_lr_decay,
                                                           total_iters=self.total_iters)
        self.gamma = config.gamma
        self.sync_frequency = config.sync_frequency
        self.n_actions = {k: self.model.individual_q_networks[k].action_space.n for k in self.group_keys}
        self.policy_type = self.model.policy_type

    def build_actions_mean_input(self, sample: Optional[dict], use_parameter_sharing: Optional[bool] = False):
        batch_size = sample['batch_size']
        seq_length = sample['sequence_length'] if self.use_rnn else 1
        actions_mean, actions_mean_next = {}, {}
        actions_mean_tensor = torch.stack([torch.as_tensor(v, device=self.device)
                                           for v in sample['actions_mean'].values()], dim=1)
        if self.use_rnn:
            actions_mean_next_tensor = None
        else:
            actions_mean_next_tensor = torch.stack([torch.as_tensor(v, device=self.device)
                                                    for v in sample['actions_mean_next'].values()], dim=1)

        for group, n_agents in self.n_group_agents.items():
            bs = batch_size * n_agents

            if self.use_rnn:
                actions_mean[group] = actions_mean_tensor.reshape([bs, seq_length + 1, -1])
            else:
                actions_mean[group] = actions_mean_tensor.reshape([bs, -1])
                actions_mean_next[group] = actions_mean_next_tensor.reshape([bs, -1])

        return actions_mean, actions_mean_next

    def update(self, sample):
        self.iterations += 1

        # prepare training data
        act_mean, act_mean_next = self.build_actions_mean_input(sample=sample,
                                                                use_parameter_sharing=self.use_parameter_sharing)
        sample_Tensor = self.build_training_data(sample=sample,
                                                 use_parameter_sharing=self.use_parameter_sharing,
                                                 use_actions_mask=self.use_actions_mask)

        batch_size = sample_Tensor['batch_size']
        obs = sample_Tensor['obs']
        actions = sample_Tensor['actions']
        obs_next = sample_Tensor['obs_next']
        rewards = sample_Tensor['rewards']
        terminals = sample_Tensor['terminals']
        agent_mask = sample_Tensor['agent_mask']
        avail_actions = sample_Tensor['avail_actions']
        avail_actions_next = sample_Tensor['avail_actions_next']
        agent_indices = sample_Tensor['agent_indices']

        if not self.agent_grouping.full_independent:
            obs = self.packed_tensor(obs)
            actions = self.packed_tensor(actions)
            obs_next = self.packed_tensor(obs_next)
            avail_actions = self.packed_tensor(avail_actions)
            avail_actions_next = self.packed_tensor(avail_actions_next)
            agent_mask = self.packed_tensor(agent_mask)
            rewards = self.packed_tensor(rewards)
            terminals = self.packed_tensor(terminals)

        info = self.callback.on_update_start(self.iterations, method="update", model=self.model)

        q_eval = self.model(observations=obs, mean_actions=act_mean,
                            agent_indices=agent_indices, avail_actions=avail_actions).values
        q_next = self.model.Qtarget(observations=obs_next, mean_actions=act_mean_next,
                                    agent_indices=agent_indices).values
        if not self.agent_grouping.full_independent:
            q_eval = self.packed_tensor(q_eval)
            q_next = self.packed_tensor(q_next)

        individual_loss = []
        for group, agent_keys in self.groups.items():
            bs = batch_size * self.n_group_agents[group]
            mask_values = agent_mask[group]
            q_eval_a = q_eval[group].gather(-1, actions[group].long().unsqueeze(-1)).reshape(bs)

            if self.use_actions_mask:
                q_next[group][avail_actions_next[group] == 0] = -1e10

            if self.policy_type == "Boltzmann":
                pi_probs = self.model.get_boltzmann_policy(q_next[group])
                v_mf = (pi_probs * q_next[group]).sum(-1).reshape(-1)
                q_target = rewards[group] + (1 - terminals[group]) * self.gamma * v_mf
            elif self.policy_type == "greedy":
                actions_next_greedy = self.model(observations=obs_next, mean_actions=act_mean_next,
                                                 group_key=group, agent_indices=agent_indices,
                                                 avail_actions=avail_actions).actions
                q_next_a = q_next[group].gather(-1, actions_next_greedy[group].unsqueeze(-1).long()).reshape(bs)
                q_target = rewards[group] + (1 - terminals[group]) * self.gamma * q_next_a
            else:
                raise NotImplementedError

            # calculate the loss function
            td_error = (q_eval_a - q_target.detach()) * mask_values
            loss_i = (td_error ** 2).sum() / mask_values.sum()
            individual_loss.append(loss_i)

            info.update({
                f"{group}/predictQ": q_eval_a.mean().item()
            })

            info.update(self.callback.on_update_agent_wise(self.iterations, group, info=info, method="update",
                                                           mask_values=mask_values, q_eval_a=q_eval_a,
                                                           q_next=q_next[group], q_target=q_target,
                                                           td_error=td_error))
        loss = sum(individual_loss)
        self.optimizer.zero_grad()
        loss.backward()
        if self.use_grad_clip:
            torch.nn.utils.clip_grad_norm_(self.model.parameters_model, self.grad_clip_norm)
        self.optimizer.step()
        if self.scheduler is not None:
            self.scheduler.step()

        lr = self.optimizer.state_dict()['param_groups'][0]['lr']

        if self.iterations % self.sync_frequency == 0:
            self.model.copy_target()

        info.update({
            "learning_rate": lr,
            "loss_Q": loss.item()
        })

        info.update(self.callback.on_update_end(self.iterations, method="update", model=self.model, info=info))

        return info

    def update_rnn(self, sample):
        self.iterations += 1

        # prepare training data
        act_mean, act_mean_next = self.build_actions_mean_input(sample=sample,
                                                                use_parameter_sharing=self.use_parameter_sharing)
        sample_Tensor = self.build_training_data(sample=sample,
                                                 use_parameter_sharing=self.use_parameter_sharing,
                                                 use_actions_mask=self.use_actions_mask)
        batch_size = sample_Tensor['batch_size']
        seq_len = sample_Tensor['seq_length']
        obs = sample_Tensor['obs']
        actions = sample_Tensor['actions']
        rewards = sample_Tensor['rewards']
        terminals = sample_Tensor['terminals']
        agent_mask = sample_Tensor['agent_mask']
        avail_actions = sample_Tensor['avail_actions']
        filled = sample_Tensor['filled']
        agent_indices = sample_Tensor['agent_indices']

        if not self.agent_grouping.full_independent:
            obs = self.packed_tensor(obs)
            actions = self.packed_tensor(actions)
            avail_actions = self.packed_tensor(avail_actions)
            terminals = self.packed_tensor(terminals)
            rewards = self.packed_tensor(rewards)
            agent_mask = self.packed_tensor(agent_mask)

        info = self.callback.on_update_start(self.iterations, method="update_rnn",
                                             model=self.model, sample_Tensor=sample_Tensor)

        # calculate the individual Q values.
        rnn_states = self.model.init_rnn_states(batch_size)
        model_output = self.model(observations=obs, mean_actions=act_mean,
                                  agent_indices=agent_indices, avail_actions=avail_actions,
                                  rnn_states=rnn_states)
        actions_greedy, q_eval = model_output.actions, model_output.values
        q_next_seq = self.model.Qtarget(observations=obs, mean_actions=act_mean, agent_indices=agent_indices,
                                        rnn_states=rnn_states).values

        if not self.agent_grouping.full_independent:
            q_eval = self.packed_tensor(q_eval)
            actions_greedy = self.packed_tensor(actions_greedy)
            q_next_seq = self.packed_tensor(q_next_seq)

        individual_loss = []
        for group, agent_keys in self.groups.items():
            n_agents = len(agent_keys)
            bs = batch_size * n_agents
            group_filled = filled.unsqueeze(1).expand(batch_size, n_agents, seq_len).reshape([bs, seq_len])
            mask_values = agent_mask[group] * group_filled
            # calculate the target Q values
            q_eval_a = q_eval[group][:, :-1].gather(-1, actions[group].long().unsqueeze(-1)).reshape(bs, seq_len)
            q_next = q_next_seq[group][:, 1:]

            if self.use_actions_mask:
                q_next[avail_actions[group][:, 1:] == 0] = -1e10

            if self.policy_type == "Boltzmann":
                pi_probs = self.model.get_boltzmann_policy(q_next)
                v_mf = (pi_probs * q_next).sum(-1).reshape(bs, seq_len)
                q_target = rewards[group] + (1 - terminals[group]) * self.gamma * v_mf
            elif self.policy_type == "greedy":
                actions_next_greedy = actions_greedy[group][:, 1:].unsqueeze(-1)
                q_next_a = q_next.gather(-1, actions_next_greedy.long().detach()).reshape(bs, seq_len)
                q_target = rewards[group] + (1 - terminals[group]) * self.gamma * q_next_a
            else:
                raise NotImplementedError

            # calculate the loss function
            td_errors = (q_eval_a - q_target.detach()) * mask_values
            loss_i = (td_errors ** 2).sum() / mask_values.sum()
            individual_loss.append(loss_i)

            info.update({
                f"{group}/predictQ": q_eval_a.mean().item()
            })

            info.update(self.callback.on_update_agent_wise(self.iterations, group, info=info, method="update_rnn",
                                                           mask_values=mask_values, q_eval_a=q_eval_a,
                                                           q_next_a=q_next, q_target=q_target,
                                                           td_error=td_errors))
        loss = sum(individual_loss)
        self.optimizer.zero_grad()
        loss.backward()
        if self.use_grad_clip:
            torch.nn.utils.clip_grad_norm_(self.model.parameters_model, self.grad_clip_norm)
        self.optimizer.step()
        if self.scheduler is not None:
            self.scheduler.step()

        lr = self.optimizer.state_dict()['param_groups'][0]['lr']

        if self.iterations % self.sync_frequency == 0:
            self.model.copy_target()

        info.update({
            "learning_rate": lr,
            "loss_Q": loss.item(),
        })

        info.update(self.callback.on_update_end(self.iterations, method="update_rnn", model=self.model, info=info))

        return info
