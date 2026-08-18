"""
MFQ: Mean Field Q-Learning
Paper link:
http://proceedings.mlr.press/v80/yang18d/yang18d.pdf
Implementation: Pytorch
"""
import torch
from argparse import Namespace
from xuance.torch import Module
from xuance.common import AgentGrouping, Optional
from xuance.torch.utils import AgentGroupedTensor
from xuance.torch.learners import OffPolicyMultiAgentLearner


class MFQ_Learner(OffPolicyMultiAgentLearner):
    def __init__(self,
                 config: Namespace,
                 agent_grouping: AgentGrouping,
                 model: Module,
                 callback):
        super(MFQ_Learner, self).__init__(config, agent_grouping, model, callback)
        self.sync_frequency = config.sync_frequency
        self.n_actions = {k: self.model.individual_q_networks[k].action_space.n for k in self.group_keys}
        self.policy_type = self.model.policy_type

    def build_actions_mean_input(self, sample: Optional[dict]):
        actions_mean_agent_wise = {
            agent: torch.as_tensor(sample['actions_mean'][agent], device=self.device)
            for agent in self.agent_keys
        }
        if not self.use_rnn:
            actions_mean_next_agent_wise = {
                agent: torch.as_tensor(sample['actions_mean_next'][agent], device=self.device)
                for agent in self.agent_keys
            }
        else:
            actions_mean_next_tensor = None

        return (AgentGroupedTensor.from_agent_wise(actions_mean_agent_wise, self.agent_grouping),
                AgentGroupedTensor.from_agent_wise(actions_mean_next_tensor, self.agent_grouping))

    def update(self, sample):
        self.iterations += 1

        # prepare training data
        act_mean, act_mean_next = self.build_actions_mean_input(sample=sample)
        batch = self.build_training_data(
            sample=sample,
            use_actions_mask=self.use_actions_mask
        )

        info = self.callback.on_update_start(self.iterations, model=self.model, batch=batch)

        # feedforward
        rnn_states = self.model.init_rnn_states(batch.batch_size)
        model_output = self.model(observations=batch.observations, mean_actions=act_mean,
                                  agent_indices=batch.agent_indices, avail_actions=batch.avail_actions,
                                  rnn_states=rnn_states)
        actions_greedy = model_output.actions
        q_eval = model_output.values

        with torch.no_grad():
            if self.use_rnn:
                q_next = self.model.Qtarget(observations=batch.observations, mean_actions=act_mean,
                                            agent_indices=batch.agent_indices,
                                            rnn_states=rnn_states).values
                q_eval.grouped_tensor = {k: v[:, :, :-1] for k, v in q_eval.grouped_tensor.items()}
                q_next.grouped_tensor = {k: v[:, :, 1:] for k, v in q_next.grouped_tensor.items()}
            else:
                q_next = self.model.Qtarget(observations=batch.next_observations, mean_actions=act_mean_next,
                                            agent_indices=batch.agent_indices).values

        # calculate losses and update networks for each group of agents
        individual_loss = []
        for group, n_agents in self.n_group_agents.items():
            mask_values = batch.valid_mask(group, n_agents).reshape(-1)

            rewards = batch.rewards.packed(group).reshape(-1)
            terminals = batch.terminals.packed(group).reshape(-1)

            q_eval_taken = q_eval.packed(group).gather(-1, batch.actions.packed(group).long().unsqueeze(-1)).reshape(-1)

            if self.use_actions_mask:
                if self.use_rnn:
                    next_avail_actions = batch.avail_actions.packed(group)[:, 1:]
                else:
                    next_avail_actions = batch.next_avail_actions.packed(group)
                q_next.packed(group)[next_avail_actions == 0] = -1e10

            if self.policy_type == "Boltzmann":
                pi_probs = self.model.get_boltzmann_policy(q_next.packed(group))
                v_mf = (pi_probs * q_next.packed(group)).sum(-1).reshape(-1)
                q_target = rewards + (1 - terminals) * self.gamma * v_mf
            elif self.policy_type == "greedy":
                if self.use_rnn:
                    group_actions_next = actions_greedy.packed(group)[:, 1:]
                    q_next_taken = q_next.packed(group).gather(
                        -1, group_actions_next.unsqueeze(-1).long()).reshape(-1)
                else:
                    group_actions_next = self.model(observations=batch.next_observations, mean_actions=act_mean_next,
                                                    group_key=group, agent_indices=batch.agent_indices,
                                                    avail_actions=batch.avail_actions).actions
                    q_next_taken = q_next.packed(group).gather(
                        -1, group_actions_next.packed(group).unsqueeze(-1).long()).reshape(-1)
                q_target = rewards + (1 - terminals) * self.gamma * q_next_taken
            else:
                raise NotImplementedError

            # calculate the loss function
            td_error = (q_eval_taken - q_target.detach()) * mask_values
            loss_i = (td_error ** 2).sum() / mask_values.sum()
            individual_loss.append(loss_i)

            info.update({
                f"{group}/predictQ": q_eval_taken.mean().item()
            })

            info.update(self.callback.on_update_agent_wise(self.iterations, group, info=info,
                                                           mask_values=mask_values, q_eval_a=q_eval_taken,
                                                           q_next=q_next, q_target=q_target,
                                                           td_error=td_error))
        loss = sum(individual_loss)
        self.optimizer.zero_grad()
        loss.backward()
        if self.use_grad_clip:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)
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

        info.update(self.callback.on_update_end(self.iterations, model=self.model, info=info))

        return info
