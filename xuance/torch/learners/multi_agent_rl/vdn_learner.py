"""
Value Decomposition Networks (VDN)
Paper link: https://arxiv.org/pdf/1706.05296.pdf
Implementation: Pytorch
"""
import torch
from torch import nn
from argparse import Namespace
from xuance.common import AgentGrouping
from xuance.torch.learners.multi_agent_rl.iql_learner import IQL_Learner


class VDN_Learner(IQL_Learner):
    def __init__(self,
                 config: Namespace,
                 agent_grouping: AgentGrouping,
                 model: nn.Module,
                 callback):
        super(VDN_Learner, self).__init__(config, agent_grouping, model, callback)

    def build_optimizer(self):
        super(IQL_Learner, self).build_optimizer()

    def update(self, sample):
        self.iterations += 1

        # prepare training data
        batch = self.build_training_data(
            sample=sample,
            use_actions_mask=self.use_actions_mask,
        )

        rewards_tot = torch.stack([r for r in batch.rewards.agent_wise.values()], dim=1).mean(dim=1)
        terminals_tot = torch.stack([d for d in batch.terminals.agent_wise.values()], dim=1).all(dim=1).float()

        info = self.callback.on_update_start(self.iterations, model=self.model, batch=batch,
                                             rewards_tot=rewards_tot, terminals_tot=terminals_tot)

        # feedforward
        q_eval, q_next, actions_next = self._forward_transitions(batch)

        # calculate target values
        q_eval_a, q_next_a = {}, {}
        for group, n_agents in self.n_group_agents.items():
            mask_values = batch.valid_mask(group, n_agents).reshape([batch.batch_size, n_agents, batch.seq_length])

            actions_taken = batch.actions.group(group)
            q_eval_taken = q_eval.group(group).gather(-1, actions_taken.long().unsqueeze(-1)).reshape(
                [batch.batch_size, n_agents, batch.seq_length])

            if self.config.double_q:
                actions_next_taken = actions_next.group(group)
                q_next_taken = q_next.group(group).gather(-1, actions_next_taken.long().unsqueeze(-1)).reshape(
                    [batch.batch_size, n_agents, batch.seq_length])
            else:
                q_next_taken = q_next.group(group).max(dim=-1, keepdim=True).values.reshape(
                    [batch.batch_size, n_agents, batch.seq_length])

            q_eval_taken *= mask_values
            q_next_taken *= mask_values

            # get agent-wise values
            for i, agent_key in enumerate(self.groups[group]):
                q_eval_a[agent_key] = q_eval_taken[:, i]
                q_next_a[agent_key] = q_next_taken[:, i]

        q_tot_eval = self.model.Q_tot(q_eval_a).reshape(-1)
        q_tot_next = self.model.Qtarget_tot(q_next_a).reshape(-1)
        rewards_tot = rewards_tot.reshape(-1)
        terminals_tot = terminals_tot.reshape(-1)
        q_tot_target = rewards_tot + (1 - terminals_tot) * self.gamma * q_tot_next

        # calculate the loss
        if self.use_rnn:
            filled = batch.filled_masks.reshape(-1)
            td_errors = (q_tot_eval - q_tot_target.detach()) * filled
            loss = (td_errors ** 2).sum() / filled.sum()
        else:
            loss = self.mse_loss(q_tot_eval, q_tot_target.detach())

        # update the networks
        self.optimizer.zero_grad()
        loss.backward()
        if self.use_grad_clip:
            torch.nn.utils.clip_grad_norm_(self.model.parameters_model, self.grad_clip_norm)
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
                                                q_tot_eval=q_tot_eval, q_tot_next=q_tot_next,
                                                q_tot_target=q_tot_target))

        return info
