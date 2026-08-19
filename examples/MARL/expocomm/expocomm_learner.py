from xuance.torch.learners import QMIX_Learner
from xuance.common import List
from argparse import Namespace
from operator import itemgetter
import torch
from torch import nn

class ExpoComm_Learner(QMIX_Learner):
    def __init__(self,
                 config: Namespace,
                 model_keys: List[str],
                 agent_keys: List[str],
                 policy: nn.Module,
                 callback):
        super(ExpoComm_Learner, self).__init__(config, model_keys, agent_keys, policy, callback)
        self.aux_coef = config.aux_coef

    def update_rnn(self, sample):
        self.iterations += 1

        # prepare training data
        sample_Tensor = self.build_training_data(sample=sample,
                                                 use_parameter_sharing=self.use_parameter_sharing,
                                                 use_actions_mask=self.use_actions_mask,
                                                 use_global_state=True)
        batch_size = sample_Tensor['batch_size']
        seq_len = sample['sequence_length']
        state = sample_Tensor['state']
        obs = sample_Tensor['obs']
        actions = sample_Tensor['actions']
        rewards = sample_Tensor['rewards']
        terminals = sample_Tensor['terminals']
        agent_mask = sample_Tensor['agent_mask']
        avail_actions = sample_Tensor['avail_actions']
        filled = sample_Tensor['filled'].reshape([-1, 1])
        IDs = sample_Tensor['agent_ids']

        key = self.model_keys[0]
        bs_rnn = batch_size * self.n_agents
        rewards_tot = rewards[key].mean(dim=1).reshape([-1, 1])
        terminals_tot = terminals[key].all(dim=1, keepdim=False).float().reshape([-1, 1])

        info = self.callback.on_update_start(self.iterations, method="update_rnn", policy=self.policy,
                                             sample_Tensor=sample_Tensor, bs_rnn=bs_rnn,
                                             rewards_tot=rewards_tot, terminals_tot=terminals_tot)
        key = self.model_keys[0]
        # agent_mask: [batch_size*self.n_agents, seq_length]
        # 扩展 agent_mask 到 seq_len + 1 以匹配 obs 的维度
        agent_mask_expanded = torch.cat([agent_mask[key], agent_mask[key][:, -1:]], dim=-1)  # [bs, seq_len + 1]
        alive_ally = agent_mask_expanded.view(batch_size, self.n_agents, seq_len + 1).unsqueeze(-1)
        alive_ally = {k: alive_ally[:, i] for i, k in enumerate(self.agent_keys)}
        # calculate the individual Q values.
        rnn_hidden = {k: self.policy.representation[k].init_hidden(bs_rnn) for k in self.model_keys}
        self.policy.init_msg_prev(rnn_hidden)
        # _, actions_greedy, q_eval, state_predicts = self.policy(obs, agent_ids=IDs, avail_actions=avail_actions, rnn_hidden=rnn_hidden, alive_ally=alive_ally)
        _, actions_greedy, q_eval, state_predicts = self.policy(obs, agent_ids=IDs, avail_actions=avail_actions, rnn_hidden=rnn_hidden, alive_ally=alive_ally)

        target_rnn_hidden = {k: self.policy.target_representation[k].init_hidden(bs_rnn) for k in self.model_keys}
        self.policy.init_msg_prev_target(target_rnn_hidden)
        _, q_next_seq = self.policy.Qtarget(obs, agent_ids=IDs, rnn_hidden=target_rnn_hidden, alive_ally=alive_ally)

        q_eval_a = {}
        for key in self.model_keys:
            mask_values = agent_mask[key]
            q_eval_a[key] = q_eval[key][:, :-1].gather(-1, actions[key].long().unsqueeze(-1)).reshape(bs_rnn, seq_len)
            q_eval_a[key] = q_eval_a[key] * mask_values

            if self.use_parameter_sharing:
                q_eval_a[key] = q_eval_a[key].reshape(batch_size, self.n_agents, seq_len).transpose(1, 2).reshape(-1, self.n_agents)
            else:
                q_eval_a[key] = q_eval_a[key].reshape(-1, 1)

            info.update(self.callback.on_update_agent_wise(self.iterations, key, info=info, method="update_rnn",
                                                           mask_values=mask_values, q_eval_a=q_eval_a,
                                                           actions_greedy=actions_greedy))

        q_eval_a, q_next, q_next_a = {}, {}, {}
        for key in self.model_keys:
            mask_values = agent_mask[key]
            q_eval_a[key] = q_eval[key][:, :-1].gather(-1, actions[key].long().unsqueeze(-1)).reshape(bs_rnn, seq_len)
            q_next[key] = q_next_seq[key][:, 1:]

            if self.use_actions_mask:
                q_next[key][avail_actions[key][:, 1:] == 0] = -1e10

            if self.config.double_q:
                act_next = {k: actions_greedy[k].unsqueeze(-1)[:, 1:] for k in self.model_keys}
                q_next_a[key] = q_next[key].gather(-1, act_next[key].long().detach()).reshape(bs_rnn, seq_len)
            else:
                q_next_a[key] = q_next[key].max(dim=-1, keepdim=True).values.reshape(bs_rnn, seq_len)

            q_eval_a[key] = q_eval_a[key] * mask_values
            q_next_a[key] = q_next_a[key] * mask_values

            if self.use_parameter_sharing:
                q_eval_a[key] = q_eval_a[key].reshape(batch_size, self.n_agents, seq_len).transpose(1, 2).reshape(-1, self.n_agents)
                q_next_a[key] = q_next_a[key].reshape(batch_size, self.n_agents, seq_len).transpose(1, 2).reshape(-1, self.n_agents)
            else:
                q_eval_a[key] = q_eval_a[key].reshape(-1, 1)
                q_next_a[key] = q_next_a[key].reshape(-1, 1)

            info.update(self.callback.on_update_agent_wise(self.iterations, key, info=info, method="update_rnn",
                                                           mask_values=mask_values, q_eval_a=q_eval_a,
                                                           q_next=q_next, q_next_a=q_next_a,
                                                           actions_greedy=actions_greedy))

        # calculate the aux loss
        state_predicts = state_predicts[:, :-1,].reshape(batch_size, self.n_agents, seq_len, -1)
        predict_state_error = state_predicts - state[:, None, :-1]
        predict_state_error = predict_state_error.reshape(batch_size * seq_len, -1) * filled
        aux_loss = (predict_state_error ** 2).sum() / filled.sum()

        # calculate the total Q values.
        q_tot_eval = self.policy.Q_tot(q_eval_a, state[:, :-1].reshape([batch_size * seq_len, -1]))
        q_tot_next = self.policy.Qtarget_tot(q_next_a, state[:, 1:].reshape([batch_size * seq_len, -1]))
        q_tot_target = rewards_tot + (1 - terminals_tot) * self.gamma * q_tot_next

        # calculate the loss function
        td_errors = (q_tot_eval - q_tot_target.detach()) * filled
        lossQ = (td_errors ** 2).sum() / filled.sum()
        loss = lossQ + self.aux_coef * aux_loss
        self.optimizer.zero_grad()
        loss.backward()
        if self.use_grad_clip:
            torch.nn.utils.clip_grad_norm_(self.policy.parameters_model, self.grad_clip_norm)
        self.optimizer.step()
        if self.scheduler is not None:
            self.scheduler.step()

        lr = self.optimizer.state_dict()['param_groups'][0]['lr']

        info.update({
            "learning_rate": lr,
            "loss_Q": lossQ.item(),
            "loss_aux": aux_loss.item(),
            "loss": loss.item(),
            "loss_Q": loss.item(),
            "predictQ": q_tot_eval.mean().item()
        })

        if self.iterations % self.sync_frequency == 0:
            self.policy.copy_target()

        info.update(self.callback.on_update_end(self.iterations, method="update_rnn", policy=self.policy, info=info,
                                                q_tot_eval=q_tot_eval, q_tot_next=q_tot_next,
                                                q_tot_target=q_tot_target, td_errors=td_errors))

        return info