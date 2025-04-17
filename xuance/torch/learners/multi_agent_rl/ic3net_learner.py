from argparse import Namespace
from operator import itemgetter
from typing import List, Optional

import numpy as np
from torch import nn, Tensor

import torch
from xuance.torch.learners import LearnerMAS
from xuance.torch.utils import ValueNorm


class IC3Net_Learner(LearnerMAS):
    def __init__(self,
                 config: Namespace,
                 model_keys: List[str],
                 agent_keys: List[str],
                 policy: nn.Module):
        super(IC3Net_Learner, self).__init__(config, model_keys, agent_keys, policy)
        self.use_global_state = self.config.use_global_state
        self.build_optimizer()
        self.use_value_clip, self.value_clip_range = config.use_value_clip, config.value_clip_range
        self.use_huber_loss, self.huber_delta = config.use_huber_loss, config.huber_delta
        self.use_value_norm = config.use_value_norm
        self.vf_coef, self.ent_coef = config.vf_coef, config.ent_coef
        self.mse_loss = nn.MSELoss()
        self.huber_loss = nn.HuberLoss(reduction="none", delta=self.huber_delta)
        if self.use_value_norm:
            self.value_normalizer = {key: ValueNorm(1).to(self.device) for key in self.model_keys}
        else:
            self.value_normalizer = None

    def build_optimizer(self):
        self.optimizer = torch.optim.Adam(self.policy.parameters_model, lr=self.learning_rate, eps=1e-5,
                                          weight_decay=self.config.weight_decay)
        self.scheduler = torch.optim.lr_scheduler.LinearLR(self.optimizer,
                                                           start_factor=1.0,
                                                           end_factor=self.end_factor_lr_decay,
                                                           total_iters=self.config.running_steps)

    def build_training_data(self, sample: Optional[dict],
                            use_parameter_sharing: Optional[bool] = False,
                            use_actions_mask: Optional[bool] = False,
                            use_global_state: Optional[bool] = False):
        batch_size = sample['batch_size']
        seq_length = sample['sequence_length'] if self.use_rnn else 1
        state, avail_actions, filled, IDs = None, None, None, None

        obs = {k: Tensor(sample['obs'][k]).to(self.device) for k in self.agent_keys}
        actions = {k: Tensor(sample['actions'][k]).to(self.device) for k in self.agent_keys}
        message = {k: Tensor(sample['message'][k]).to(self.device) for k in self.agent_keys}
        values = {k: Tensor(sample['values'][k]).to(self.device) for k in self.agent_keys}
        returns = {k: Tensor(sample['returns'][k]).to(self.device) for k in self.agent_keys}
        advantages = {k: Tensor(sample['advantages'][k]).to(self.device) for k in self.agent_keys}
        log_pi_old = {k: Tensor(sample['log_pi_old'][k]).to(self.device) for k in self.agent_keys}
        terminals = {k: Tensor(sample['terminals'][k]).float().to(self.device) for k in self.agent_keys}
        agent_mask = {k: Tensor(sample['agent_mask'][k]).float().to(self.device) for k in self.agent_keys}
        if use_actions_mask:
            avail_actions = {k: Tensor(sample['avail_actions'][k]).float().to(self.device) for k in self.agent_keys}

        if use_global_state:
            state = Tensor(sample['state']).to(self.device)

        if self.use_rnn:
            filled = Tensor(sample['filled']).float().to(self.device)

        sample_Tensor = {
            'batch_size': batch_size,
            'state': state,
            'obs': obs,
            'actions': actions,
            'message': message,
            'values': values,
            'returns': returns,
            'advantages': advantages,
            'log_pi_old': log_pi_old,
            'terminals': terminals,
            'agent_mask': agent_mask,
            'avail_actions': avail_actions,
            'agent_ids': IDs,
            'filled': filled,
            'seq_length': seq_length,
        }
        return sample_Tensor

    def update(self, sample):
        pass

    def update_rnn(self, sample):
        self.iterations += 1
        info = {}

        sample_Tensor = self.build_training_data(sample=sample,
                                                 use_parameter_sharing=self.use_parameter_sharing,
                                                 use_actions_mask=self.use_actions_mask)
        batch_size = sample_Tensor['batch_size']
        state = sample_Tensor['state']
        bs_rnn = batch_size * self.n_agents if self.use_parameter_sharing else batch_size
        obs = sample_Tensor['obs']
        actions = sample_Tensor['actions']
        message = sample_Tensor['message']
        values = sample_Tensor['values']
        returns = sample_Tensor['returns']
        advantages = sample_Tensor['advantages']
        avail_actions = sample_Tensor['avail_actions']
        agent_mask = sample_Tensor['agent_mask']
        filled = sample_Tensor['filled']
        seq_len = filled.shape[1]
        IDs = sample_Tensor['agent_ids']

        if self.use_parameter_sharing:
            filled = filled.unsqueeze(1).expand(batch_size, self.n_agents, seq_len).reshape(bs_rnn, seq_len)

        rnn_hidden_actor = {k: self.policy.actor_representation[k].init_hidden(bs_rnn) for k in self.model_keys}
        rnn_hidden_critic = {k: self.policy.critic_representation[k].init_hidden(bs_rnn) for k in self.model_keys}

        obs = self.policy.observation_encode(obs)
        if self.use_global_state:
            critic_input = {k: state.reshape(batch_size, seq_len, -1) for k in self.agent_keys}
        else:
            joint_obs = self.get_joint_input(obs, (batch_size, seq_len, -1))
            critic_input = {k: joint_obs for k in self.agent_keys}

        # feedforward
        _, pi_dist_dict, _ = self.policy(obs, agent_ids=IDs, avail_actions=avail_actions, rnn_hidden=rnn_hidden_actor, message_input=message)
        _, values_pred_dict = self.policy.get_values(critic_input, agent_ids=IDs, rnn_hidden=rnn_hidden_critic, message_input=message)

        # calculate losses for each agent
        loss_a, loss_e, loss_c = [], [], []
        for key in self.model_keys:
            mask_values = agent_mask[key] * filled
            # policy gradient loss
            log_pi = pi_dist_dict[key].log_prob(actions[key]).reshape(bs_rnn, seq_len)
            pg_loss = -((advantages[key].detach() * log_pi) * mask_values).sum() / mask_values.sum()
            loss_a.append(pg_loss)

            # entropy loss
            entropy = pi_dist_dict[key].entropy()
            entropy_loss = (entropy * mask_values).sum() / mask_values.sum()
            loss_e.append(entropy_loss)

            # value loss
            value_pred_i = values_pred_dict[key].reshape(bs_rnn, seq_len)
            value_target = returns[key].reshape(bs_rnn, seq_len)
            values_i = values[key].reshape(bs_rnn, seq_len)
            if self.use_value_clip:
                value_clipped = values_i + (value_pred_i - values_i).clamp(-self.value_clip_range,
                                                                           self.value_clip_range)
                if self.use_value_norm:
                    self.value_normalizer[key].update(value_target.reshape(-1, 1))
                    value_target = self.value_normalizer[key].normalize(value_target.reshape(-1, 1))
                    value_target = value_target.reshape(bs_rnn, seq_len)
                if self.use_huber_loss:
                    loss_v = self.huber_loss(value_pred_i, value_target)
                    loss_v_clipped = self.huber_loss(value_clipped, value_target)
                else:
                    loss_v = (value_pred_i - value_target) ** 2
                    loss_v_clipped = (value_clipped - value_target) ** 2
                loss_c_ = torch.max(loss_v, loss_v_clipped) * mask_values
                loss_c.append(loss_c_.sum() / mask_values.sum())
            else:
                if self.use_value_norm:
                    self.value_normalizer[key].update(value_target)
                    value_target = self.value_normalizer[key].normalize(value_target)
                if self.use_huber_loss:
                    loss_v = self.huber_loss(value_pred_i, value_target)
                else:
                    loss_v = (value_pred_i - value_target) ** 2
                loss_c.append((loss_v * mask_values).sum() / mask_values.sum())

            info.update({
                f"predict_value/{key}": value_pred_i.mean().item()
            })

        loss = sum(loss_a) + self.vf_coef * sum(loss_c) - self.ent_coef * sum(loss_e)
        self.optimizer.zero_grad()
        loss.backward()
        if self.use_grad_clip:
            grad_norm = torch.nn.utils.clip_grad_norm_(self.policy.parameters_model, self.grad_clip_norm)
            info["gradient_norm"] = grad_norm.item()
        self.optimizer.step()
        if self.scheduler is not None:
            self.scheduler.step()

        # Logger
        lr = self.optimizer.state_dict()['param_groups'][0]['lr']

        info.update({
            "learning_rate": lr,
            "pg_loss": sum(loss_a).item(),
            "vf_loss": sum(loss_c).item(),
            "entropy_loss": sum(loss_e).item(),
            "loss": loss.item(),
        })

        return info