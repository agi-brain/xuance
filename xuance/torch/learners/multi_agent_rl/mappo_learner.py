"""
Multi-Agent Proximal Policy Optimization (MAPPO)
Paper link:
https://proceedings.neurips.cc/paper_files/paper/2022/file/9c1535a02f0ce079433344e14d910597-Paper-Datasets_and_Benchmarks.pdf
Implementation: Pytorch
"""
import torch
import numpy as np
from torch import nn
from typing import Optional, List
from argparse import Namespace
from xuance.torch import Tensor
from xuance.torch.learners.multi_agent_rl.ippo_learner import IPPO_Learner


class MAPPO_Clip_Learner(IPPO_Learner):
    def __init__(self,
                 config: Namespace,
                 model_keys: List[str],
                 agent_keys: List[str],
                 episode_length: int,
                 policy: nn.Module,
                 optimizer: Optional[torch.optim.Adam],
                 scheduler: Optional[torch.optim.lr_scheduler.LinearLR] = None):
        super(MAPPO_Clip_Learner, self).__init__(config, model_keys, agent_keys, episode_length,
                                                 policy, optimizer, scheduler)
    
    def update(self, sample):
        self.iterations += 1
        info = {}

        # prepare training data
        sample_Tensor = self.build_training_data(sample=sample,
                                                 use_parameter_sharing=self.use_parameter_sharing,
                                                 use_actions_mask=self.use_actions_mask)
        batch_size = sample_Tensor['batch_size']
        obs = sample_Tensor['obs']
        actions = sample_Tensor['actions']
        agent_mask = sample_Tensor['agent_mask']
        avail_actions = sample_Tensor['avail_actions']
        values = sample_Tensor['values']
        returns = sample_Tensor['returns']
        advantages = sample_Tensor['advantages']
        log_pi_old = sample_Tensor['log_pi_old']
        IDs = sample_Tensor['agent_ids']

        # prepare critic inputs
        if self.use_parameter_sharing:
            key = self.model_keys[0]
            critic_input_array = obs[key].reshape(batch_size, 1, -1).expand(batch_size, self.n_agents, -1)
            if self.use_global_state:
                state_input = sample_Tensor['state'].reshape(batch_size, 1, -1).expand(batch_size, self.n_agents, -1)
                critic_input_array = torch.concat([critic_input_array, state_input], dim=-1)
            critic_input = {key: critic_input_array}
        else:
            critic_input_array = torch.concat([obs[k].reshape(batch_size, 1, -1) for k in self.agent_keys], dim=1).reshape(batch_size, -1)
            if self.use_global_state:
                state_input = sample_Tensor['state'].reshape(batch_size, -1)
                critic_input_array = torch.concat([critic_input_array, state_input], dim=-1)
            critic_input = {k: critic_input_array for k in self.agent_keys}

        # feedforward
        _, pi_dists_dict = self.policy(observation=obs, agent_ids=IDs, avail_actions=avail_actions)
        _, value_pred_dict = self.policy.get_values(observation=critic_input, agent_ids=IDs)

        # calculate losses for each agent
        loss_a, loss_e, loss_c = [], [], []
        for key in self.model_keys:
            # actor loss
            log_pi = pi_dists_dict[key].log_prob(actions[key]).reshape(-1, 1)
            ratio = torch.exp(log_pi - log_pi_old[key]).reshape(-1, 1)
            advantages_mask = advantages[key].detach() * agent_mask[key]
            surrogate1 = ratio * advantages_mask
            surrogate2 = torch.clip(ratio, 1 - self.clip_range, 1 + self.clip_range) * advantages_mask
            loss_a.append(-torch.min(surrogate1, surrogate2).mean())

            # entropy loss
            entropy = pi_dists_dict[key].entropy().reshape(-1, 1) * agent_mask[key]
            loss_e.append(entropy.mean())

            # critic loss
            value_pred_i = value_pred_dict[key].reshape(-1, 1)
            value_target = returns[key].reshape(-1, 1)
            values_i = values[key].reshape(-1, 1)
            agent_mask_flatten = agent_mask[key].reshape(-1, 1)
            if self.use_value_clip:
                value_clipped = values_i + (value_pred_i - values_i).clamp(-self.value_clip_range, self.value_clip_range)
                if self.use_value_norm:
                    self.value_normalizer[key].update(value_target)
                    value_target = self.value_normalizer[key].normalize(value_target)
                if self.use_huber_loss:
                    loss_v = self.huber_loss(value_pred_i, value_target)
                    loss_v_clipped = self.huber_loss(value_clipped, value_target)
                else:
                    loss_v = (value_pred_i - value_target) ** 2
                    loss_v_clipped = (value_clipped - value_target) ** 2
                loss_c_ = torch.max(loss_v, loss_v_clipped) * agent_mask_flatten
                loss_c.append(loss_c_.sum() / agent_mask_flatten.sum())
            else:
                if self.use_value_norm:
                    self.value_normalizer[key].update(value_target)
                    value_target = self.value_normalizer[key].normalize(value_target)
                if self.use_huber_loss:
                    loss_v = self.huber_loss(value_pred_i, value_target) * agent_mask_flatten
                else:
                    loss_v = ((value_pred_i - value_target) ** 2) * agent_mask_flatten
                loss_c.append(loss_v.sum() / agent_mask_flatten.sum())

            info.update({
                f"{key}/actor_loss": loss_a[-1].item(),
                f"{key}/critic_loss": loss_c[-1].item(),
                f"{key}/entropy": loss_e[-1].item(),
                f"{key}/predict_value": value_pred_i.mean().item()
            })

        loss = (sum(loss_a) + self.vf_coef * sum(loss_c) - self.ent_coef * sum(loss_e)) / self.n_agents
        self.optimizer.zero_grad()
        loss.backward()
        if self.use_grad_clip:
            grad_norm = torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.grad_clip_norm)
            info["gradient_norm"] = grad_norm.item()
        self.optimizer.step()
        if self.scheduler is not None and self.use_linear_lr_decay:
            self.scheduler.step()

        # Logger
        lr = self.optimizer.state_dict()['param_groups'][0]['lr']

        info.update({
            "learning_rate": lr,
            "loss": loss.item(),
        })

        return info

    def update_rnn(self, sample):
        self.iterations += 1
        info = {}

        sample_Tensor = self.build_training_data(sample=sample,
                                                 use_parameter_sharing=self.use_parameter_sharing,
                                                 use_actions_mask=self.use_actions_mask)
        batch_size = sample_Tensor['batch_size']
        bs_rnn = batch_size * self.n_agents if self.use_parameter_sharing else batch_size
        obs = sample_Tensor['obs']
        actions = sample_Tensor['actions']
        values = sample_Tensor['values']
        returns = sample_Tensor['returns']
        advantages = sample_Tensor['advantages']
        log_pi_old = sample_Tensor['log_pi_old']
        avail_actions = sample_Tensor['avail_actions']
        filled = sample_Tensor['filled']
        seq_len = filled.shape[1]
        IDs = sample_Tensor['agent_ids']

        if self.use_parameter_sharing:
            key = self.model_keys[0]
            filled = filled.unsqueeze(1).expand(batch_size, self.n_agents, seq_len).reshape(-1, 1)
            obs_concate = Tensor(np.concatenate([sample['obs'][k][:, None, :, None] for k in self.agent_keys],
                                                axis=3).reshape([batch_size, 1, seq_len, -1])).to(self.device)
            critic_input_array = obs_concate.expand(-1, self.n_agents, -1, -1).reshape(bs_rnn, seq_len, -1)
            if self.use_global_state:
                state_input = sample_Tensor['state'].unsqueeze(1).expand(
                    -1, self.n_agents, -1, -1).reshape(bs_rnn, seq_len, -1)
                critic_input_array = torch.concat([critic_input_array, state_input], dim=-1)
            critic_input = {key: critic_input_array}
        else:
            filled = filled.reshape(-1, 1)
            critic_input_array = torch.concat([obs[k].reshape(batch_size, seq_len, 1, -1) for k in self.agent_keys],
                                              dim=2).reshape(batch_size, seq_len, -1)
            if self.use_global_state:
                state_input = sample_Tensor['state'].reshape(batch_size, seq_len, -1)
                critic_input_array = torch.concat([critic_input_array, state_input], dim=-1)
            critic_input = {k: critic_input_array for k in self.agent_keys}
        rnn_hidden_actor = {k: self.policy.actor_representation[k].init_hidden(bs_rnn) for k in self.model_keys}
        rnn_hidden_critic = {k: self.policy.critic_representation[k].init_hidden(bs_rnn) for k in self.model_keys}

        # feedforward
        _, pi_dist_dict = self.policy(obs, agent_ids=IDs, avail_actions=avail_actions, rnn_hidden=rnn_hidden_actor)
        _, value_pred_dict = self.policy.get_values(observation=critic_input, agent_ids=IDs, rnn_hidden=rnn_hidden_critic)

        # calculate losses for each agent
        loss_a, loss_e, loss_c = [], [], []
        for key in self.model_keys:
            log_pi = pi_dist_dict[key].log_prob(actions[key]).reshape(-1, 1)
            ratio = torch.exp(log_pi - log_pi_old[key])
            surrogate1 = ratio * advantages[key]
            surrogate2 = torch.clip(ratio, 1 - self.clip_range, 1 + self.clip_range) * advantages[key]
            loss_a.append(-(torch.min(surrogate1, surrogate2) * filled).sum() / filled.sum())

            # entropy loss
            entropy = pi_dist_dict[key].entropy().reshape(-1, 1)
            entropy = entropy * filled
            loss_e.append(entropy.sum() / filled.sum())

            # critic loss
            value_pred_i = value_pred_dict[key].reshape(-1, 1)
            value_target = returns[key].reshape(-1, 1)
            values_i = values[key].reshape(-1, 1)
            if self.use_value_clip:
                value_clipped = values_i + (value_pred_i - values_i).clamp(-self.value_clip_range,
                                                                           self.value_clip_range)
                if self.use_value_norm:
                    self.value_normalizer[key].update(value_target)
                    value_target = self.value_normalizer[key].normalize(value_target)
                if self.use_huber_loss:
                    loss_v = self.huber_loss(value_pred_i, value_target)
                    loss_v_clipped = self.huber_loss(value_clipped, value_target)
                else:
                    loss_v = (value_pred_i - value_target) ** 2
                    loss_v_clipped = (value_clipped - value_target) ** 2
                loss_c_ = torch.max(loss_v, loss_v_clipped) * filled
                loss_c.append(loss_c_.sum() / filled.sum())
            else:
                if self.use_value_norm:
                    self.value_normalizer[key].update(value_target)
                    value_target = self.value_normalizer[key].normalize(value_target)
                if self.use_huber_loss:
                    loss_v = self.huber_loss(value_pred_i, value_target)
                else:
                    loss_v = (value_pred_i - value_target) ** 2
                loss_c.append((loss_v * filled).sum() / filled.sum())

            info.update({
                f"{key}/actor_loss": loss_a[-1].item(),
                f"{key}/critic_loss": loss_c[-1].item(),
                f"{key}/entropy": loss_e[-1].item(),
                f"{key}/predict_value": value_pred_i.mean().item()
            })

        loss = sum(loss_a) + self.vf_coef * sum(loss_c) - self.ent_coef * sum(loss_e)
        self.optimizer.zero_grad()
        loss.backward()
        if self.use_grad_clip:
            grad_norm = torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.grad_clip_norm)
            info["gradient_norm"] = grad_norm.item()
        self.optimizer.step()
        if self.scheduler is not None:
            self.scheduler.step()

        # Logger
        lr = self.optimizer.state_dict()['param_groups'][0]['lr']

        info.update({
            "learning_rate": lr,
            "loss": loss.item(),
        })

        return info