"""
Independent Proximal Policy Optimization (IPPO)
Paper link: https://arxiv.org/pdf/2103.01955.pdf
Implementation: Pytorch
"""
import torch
from torch import nn
from typing import Optional, List
from argparse import Namespace
from operator import itemgetter
from numpy import concatenate as concat
from xuance.torch import Tensor
from xuance.torch.utils import ValueNorm
from xuance.torch.learners import LearnerMAS


class IPPO_Learner(LearnerMAS):
    def __init__(self,
                 config: Namespace,
                 model_keys: List[str],
                 agent_keys: List[str],
                 episode_length: int,
                 policy: nn.Module,
                 optimizer: Optional[torch.optim.Adam],
                 scheduler: Optional[torch.optim.lr_scheduler.LinearLR] = None):
        self.gamma = config.gamma
        self.clip_range = config.clip_range
        self.use_linear_lr_decay = config.use_linear_lr_decay
        self.use_value_clip, self.value_clip_range = config.use_value_clip, config.value_clip_range
        self.use_huber_loss, self.huber_delta = config.use_huber_loss, config.huber_delta
        self.use_value_norm = config.use_value_norm
        self.use_global_state = config.use_global_state
        self.vf_coef, self.ent_coef = config.vf_coef, config.ent_coef
        self.mse_loss = nn.MSELoss()
        self.huber_loss = nn.HuberLoss(reduction="none", delta=self.huber_delta)
        super(IPPO_Learner, self).__init__(config, model_keys, agent_keys, episode_length, policy, optimizer, scheduler)
        if self.use_value_norm:
            self.value_normalizer = {key: ValueNorm(1).to(self.device) for key in self.model_keys}
        else:
            self.value_normalizer = None
        self.lr = config.learning_rate
        self.end_factor_lr_decay = config.end_factor_lr_decay

    def build_training_data(self, sample: Optional[dict],
                            use_parameter_sharing: Optional[bool] = False,
                            use_actions_mask: Optional[bool] = False,
                            use_global_state: Optional[bool] = False):
        """
        Prepare the training data.

        Parameters:
            sample (dict): The raw sampled data.
            use_parameter_sharing (bool): Whether to use parameter sharing for individual agent models.
            use_actions_mask (bool): Whether to use actions mask for unavailable actions.
            use_global_state (bool): Whether to use global state.

        Returns:
            sample_Tensor (dict): The formatted sampled data.
        """
        batch_size = sample['batch_size']
        state, avail_actions, filled, IDs = None, None, None, None
        if use_parameter_sharing:
            k = self.model_keys[0]
            obs_array = itemgetter(*self.agent_keys)(sample['obs'])
            obs = {k: Tensor(concat([obs_array[i][:, None] for i in range(self.n_agents)], 1)).to(self.device)}
            act_array = itemgetter(*self.agent_keys)(sample['actions'])
            actions = {k: Tensor(concat([act_array[i][:, None] for i in range(self.n_agents)], 1)).to(self.device)}
            values_array = itemgetter(*self.agent_keys)(sample['values'])
            values = {k: Tensor(concat([values_array[i][:, None] for i in range(self.n_agents)], 1)).to(self.device)}
            returns_array = itemgetter(*self.agent_keys)(sample['returns'])
            returns = {k: Tensor(concat([returns_array[i][:, None] for i in range(self.n_agents)], 1)).to(self.device)}
            advantages_array = itemgetter(*self.agent_keys)(sample['advantages'])
            advantages = {k: Tensor(concat([advantages_array[i][:, None] for i in range(self.n_agents)], 1)).to(self.device)}
            log_pi_old_array = itemgetter(*self.agent_keys)(sample['log_pi_old'])
            log_pi_old = {k: Tensor(concat([log_pi_old_array[i][:, None] for i in range(self.n_agents)], 1)).to(self.device)}
            ter_array = itemgetter(*self.agent_keys)(sample['terminals'])
            terminals = {k: Tensor(concat([ter_array[i][:, None] for i in range(self.n_agents)], 1)).float().to(self.device)}
            agt_mask_array = itemgetter(*self.agent_keys)(sample['agent_mask'])
            agent_mask = {k: Tensor(concat([agt_mask_array[i][:, None] for i in range(self.n_agents)], 1)).float().to(self.device)}
            if self.use_rnn:
                bs_rnn = batch_size * self.n_agents
                seq_len = act_array[0].shape[1]
                obs[k] = obs[k].reshape([bs_rnn, seq_len, -1])
                if len(actions[k].shape) == 4:
                    actions[k] = actions[k].reshape([bs_rnn, seq_len, -1])
                else:
                    actions[k] = actions[k].reshape([bs_rnn, seq_len])
                IDs = torch.eye(self.n_agents).unsqueeze(1).unsqueeze(0).expand(batch_size, -1, seq_len, -1).to(self.device)
                IDs = IDs.reshape(bs_rnn, seq_len, -1)
            else:
                IDs = torch.eye(self.n_agents).unsqueeze(0).expand(batch_size, -1, -1).to(self.device)
            values[k] = values[k].reshape(-1, 1)
            returns[k] = returns[k].reshape(-1, 1)
            advantages[k] = advantages[k].reshape(-1, 1)
            log_pi_old[k] = log_pi_old[k].reshape(-1, 1)
            terminals[k] = terminals[k].reshape(-1, 1)
            agent_mask[k] = agent_mask[k].reshape(-1, 1)
            if use_actions_mask:
                act_mask_array = itemgetter(*self.agent_keys)(sample['avail_actions'])
                avail_actions = {k: Tensor(concat([act_mask_array[i][:, None] for i in range(self.n_agents)], 1)).float().to(self.device)}

        else:
            obs = {k: Tensor(sample['obs'][k]).to(self.device) for k in self.agent_keys}
            actions = {k: Tensor(sample['actions'][k]).to(self.device) for k in self.agent_keys}
            values = {k: Tensor(sample['values'][k]).to(self.device).reshape(-1, 1) for k in self.agent_keys}
            returns = {k: Tensor(sample['returns'][k]).to(self.device).reshape(-1, 1) for k in self.agent_keys}
            advantages = {k: Tensor(sample['advantages'][k]).to(self.device).reshape(-1, 1) for k in self.agent_keys}
            log_pi_old = {k: Tensor(sample['log_pi_old'][k]).to(self.device).reshape(-1, 1) for k in self.agent_keys}
            terminals = {k: Tensor(sample['terminals'][k]).float().to(self.device).reshape(-1, 1) for k in self.agent_keys}
            agent_mask = {k: Tensor(sample['agent_mask'][k]).float().to(self.device).reshape(-1, 1) for k in self.agent_keys}
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
            'values': values,
            'returns': returns,
            'advantages': advantages,
            'log_pi_old': log_pi_old,
            'terminals': terminals,
            'agent_mask': agent_mask,
            'avail_actions': avail_actions,
            'agent_ids': IDs,
            'filled': filled,
        }
        return sample_Tensor

    def update(self, sample):
        self.iterations += 1
        info = {}

        # prepare training data
        sample_Tensor = self.build_training_data(sample=sample,
                                                 use_parameter_sharing=self.use_parameter_sharing,
                                                 use_actions_mask=self.use_actions_mask)
        obs = sample_Tensor['obs']
        actions = sample_Tensor['actions']
        agent_mask = sample_Tensor['agent_mask']
        avail_actions = sample_Tensor['avail_actions']
        values = sample_Tensor['values']
        returns = sample_Tensor['returns']
        advantages = sample_Tensor['advantages']
        log_pi_old = sample_Tensor['log_pi_old']
        IDs = sample_Tensor['agent_ids']

        # feedforward
        _, pi_dists_dict = self.policy(observation=obs, agent_ids=IDs, avail_actions=avail_actions)
        _, value_pred_dict = self.policy.get_values(observation=obs, agent_ids=IDs)

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

        rnn_hidden_actor, rnn_hidden_critic = {}, {}
        for key in self.model_keys:
            if self.use_parameter_sharing:
                filled = filled.unsqueeze(1).expand(batch_size, self.n_agents, seq_len).reshape(-1, 1)
            else:
                filled = filled.reshape(-1, 1)
            rnn_hidden_actor[key] = self.policy.actor_representation[key].init_hidden(bs_rnn)
            rnn_hidden_critic[key] = self.policy.critic_representation[key].init_hidden(bs_rnn)

        # feedforward
        _, pi_dist_dict = self.policy(obs, agent_ids=IDs, avail_actions=avail_actions, rnn_hidden=rnn_hidden_actor)
        # calculate values
        if self.use_global_state:
            state = sample_Tensor['state']
            _, value_pred_dict = self.policy.get_values(observation=state, agent_ids=IDs, rnn_hidden=rnn_hidden_critic)
        else:
            _, value_pred_dict = self.policy.get_values(observation=obs, agent_ids=IDs, rnn_hidden=rnn_hidden_critic)

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
