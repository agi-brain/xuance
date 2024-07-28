"""
Independent Soft Actor-critic (ISAC)
Implementation: Pytorch
"""
import torch
from torch import nn
from xuance.torch.learners import LearnerMAS
from xuance.common import List
from argparse import Namespace


class ISAC_Learner(LearnerMAS):
    def __init__(self,
                 config: Namespace,
                 model_keys: List[str],
                 agent_keys: List[str],
                 policy: nn.Module):
        super(ISAC_Learner, self).__init__(config, model_keys, agent_keys, policy)
        self.optimizer = {
            key: {'actor': torch.optim.Adam(self.policy.parameters_actor[key], self.config.learning_rate_actor, eps=1e-5),
                  'critic': torch.optim.Adam(self.policy.parameters_critic[key], self.config.learning_rate_critic, eps=1e-5)}
            for key in self.model_keys}
        self.scheduler = {
            key: {'actor': torch.optim.lr_scheduler.LinearLR(self.optimizer[key]['actor'], start_factor=1.0,
                                                             end_factor=0.5, total_iters=self.config.running_steps),
                  'critic': torch.optim.lr_scheduler.LinearLR(self.optimizer[key]['critic'], start_factor=1.0,
                                                              end_factor=0.5, total_iters=self.config.running_steps)}
            for key in self.model_keys}
        self.gamma = config.gamma
        self.tau = config.tau
        self.alpha = {key: config.alpha for key in self.model_keys}
        self.mse_loss = nn.MSELoss()
        self.use_automatic_entropy_tuning = config.use_automatic_entropy_tuning
        if self.use_automatic_entropy_tuning:
            self.target_entropy = {key: -policy.action_space[key].shape[-1] for key in self.model_keys}
            self.log_alpha = {key: nn.Parameter(torch.zeros(1, requires_grad=True, device=self.device))
                              for key in self.model_keys}
            self.alpha = {key: self.log_alpha[key].exp() for key in self.model_keys}
            self.alpha_optimizer = {key: torch.optim.Adam([self.log_alpha[key]], lr=config.learning_rate_actor)
                                    for key in self.model_keys}

    def update(self, sample):
        self.iterations += 1
        info = {}

        # Prepare training data.
        sample_Tensor = self.build_training_data(sample,
                                                 use_parameter_sharing=self.use_parameter_sharing,
                                                 use_actions_mask=False)
        batch_size = sample_Tensor['batch_size']
        obs = sample_Tensor['obs']
        actions = sample_Tensor['actions']
        obs_next = sample_Tensor['obs_next']
        rewards = sample_Tensor['rewards']
        terminals = sample_Tensor['terminals']
        agent_mask = sample_Tensor['agent_mask']
        IDs = sample_Tensor['agent_ids']
        if self.use_parameter_sharing:
            key = self.model_keys[0]
            bs = batch_size * self.n_agents
            rewards[key] = rewards[key].reshape(batch_size * self.n_agents)
            terminals[key] = terminals[key].reshape(batch_size * self.n_agents)
        else:
            bs = batch_size

        # feedforward
        _, actions_eval, log_pi_eval = self.policy(observation=obs, agent_ids=IDs)
        _, actions_next, log_pi_next = self.policy(observation=obs_next, agent_ids=IDs)
        _, _, action_q_1, action_q_2 = self.policy.Qaction(observation=obs, actions=actions, agent_ids=IDs)
        _, _, next_q = self.policy.Qtarget(next_observation=obs_next, next_actions=actions_next, agent_ids=IDs)

        for key in self.model_keys:
            mask_values = agent_mask[key]
            # update critic
            action_q_1_i, action_q_2_i = action_q_1[key].reshape(bs), action_q_2[key].reshape(bs)
            log_pi_next_eval = log_pi_next[key].reshape(bs)
            next_q_i = next_q[key].reshape(bs)
            target_value = next_q_i - self.alpha[key] * log_pi_next_eval
            backup = rewards[key] + (1 - terminals[key]) * self.gamma * target_value
            td_error_1, td_error_2 = action_q_1_i - backup.detach(), action_q_2_i - backup.detach()
            td_error_1 *= mask_values
            td_error_2 *= mask_values
            loss_c = ((td_error_1 ** 2).sum() + (td_error_2 ** 2).sum()) / mask_values.sum()
            self.optimizer[key]['critic'].zero_grad()
            loss_c.backward()
            if self.use_grad_clip:
                torch.nn.utils.clip_grad_norm_(self.policy.parameters_critic[key], self.grad_clip_norm)
            self.optimizer[key]['critic'].step()
            if self.scheduler[key]['critic'] is not None:
                self.scheduler[key]['critic'].step()

            # update actor
            _, _, policy_q_1, policy_q_2 = self.policy.Qpolicy(observation=obs, actions=actions_eval, agent_ids=IDs,
                                                               agent_key=key)
            log_pi_eval_i = log_pi_eval[key].reshape(bs)
            policy_q = torch.min(policy_q_1[key], policy_q_2[key]).reshape(bs)
            loss_a = ((self.alpha[key] * log_pi_eval_i - policy_q) * mask_values).sum() / mask_values.sum()
            self.optimizer[key]['actor'].zero_grad()
            loss_a.backward()
            if self.use_grad_clip:
                torch.nn.utils.clip_grad_norm_(self.policy.parameters_actor[key], self.grad_clip_norm)
            self.optimizer[key]['actor'].step()
            if self.scheduler[key]['actor'] is not None:
                self.scheduler[key]['actor'].step()

            # automatic entropy tuning
            if self.use_automatic_entropy_tuning:
                alpha_loss = -(self.log_alpha[key] * (log_pi_eval_i + self.target_entropy[key]).detach()).mean()
                self.alpha_optimizer[key].zero_grad()
                alpha_loss.backward()
                self.alpha_optimizer[key].step()
                self.alpha[key] = self.log_alpha[key].exp()
            else:
                alpha_loss = 0

            learning_rate_actor = self.optimizer[key]['actor'].state_dict()['param_groups'][0]['lr']
            learning_rate_critic = self.optimizer[key]['critic'].state_dict()['param_groups'][0]['lr']

            info.update({
                f"{key}/learning_rate_actor": learning_rate_actor,
                f"{key}/learning_rate_critic": learning_rate_critic,
                f"{key}/loss_actor": loss_a.item(),
                f"{key}/loss_critic": loss_c.item(),
                f"{key}/predictQ": policy_q.mean().item(),
                f"{key}/alpha_loss": alpha_loss.item(),
                f"{key}/alpha": self.alpha[key].item(),
            })

        self.policy.soft_update(self.tau)
        return info

    def update_rnn(self, sample):
        self.iterations += 1
        info = {}

        # prepare training data
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
        filled = sample_Tensor['filled']
        IDs = sample_Tensor['agent_ids']

        if self.use_parameter_sharing:
            key = self.model_keys[0]
            bs_rnn = batch_size * self.n_agents
            filled = filled.unsqueeze(1).expand(-1, self.n_agents, -1).reshape(bs_rnn, seq_len)
            rewards[key] = rewards[key].reshape(bs_rnn, seq_len)
            terminals[key] = terminals[key].reshape(bs_rnn, seq_len)
            IDs_t = IDs[:, :-1]
        else:
            bs_rnn, IDs_t = batch_size, None

        # initial hidden states for rnn
        rnn_hidden_actor = {k: self.policy.actor_representation[k].init_hidden(bs_rnn) for k in self.model_keys}
        rnn_hidden_critic = {k: self.policy.critic_1_representation[k].init_hidden(bs_rnn) for k in self.model_keys}

        _, actions_eval, log_pi_eval = self.policy(observation=obs, agent_ids=IDs, rnn_hidden=rnn_hidden_actor)
        obs_t = {k: v[:, :-1] for k, v in obs.items()}
        _, _, action_q_1, action_q_2 = self.policy.Qaction(observation=obs_t, actions=actions, agent_ids=IDs_t,
                                                           rnn_hidden_critic_1=rnn_hidden_critic,
                                                           rnn_hidden_critic_2=rnn_hidden_critic)
        _, _, next_q = self.policy.Qtarget(next_observation=obs, next_actions=actions_eval, agent_ids=IDs,
                                           rnn_hidden_critic_1=rnn_hidden_critic,
                                           rnn_hidden_critic_2=rnn_hidden_critic)
        for key in self.model_keys:
            mask_values = agent_mask[key] * filled
            # update critic
            action_q_1_i = action_q_1[key].reshape(bs_rnn, seq_len)
            action_q_2_i = action_q_2[key].reshape(bs_rnn, seq_len)
            log_pi_next_eval = log_pi_eval[key][:, 1:].reshape(bs_rnn, seq_len)
            next_q_i = next_q[key][:, 1:].reshape(bs_rnn, seq_len)
            target_value = next_q_i - self.alpha[key] * log_pi_next_eval
            backup = rewards[key] + (1 - terminals[key]) * self.gamma * target_value
            td_error_1, td_error_2 = action_q_1_i - backup.detach(), action_q_2_i - backup.detach()
            td_error_1 *= mask_values
            td_error_2 *= mask_values
            loss_c = ((td_error_1 ** 2).sum() + (td_error_2 ** 2).sum()) / mask_values.sum()
            self.optimizer[key]['critic'].zero_grad()
            loss_c.backward()
            if self.use_grad_clip:
                torch.nn.utils.clip_grad_norm_(self.policy.parameters_critic[key], self.grad_clip_norm)
            self.optimizer[key]['critic'].step()
            if self.scheduler[key]['critic'] is not None:
                self.scheduler[key]['critic'].step()

            # update actor
            _, _, policy_q_1, policy_q_2 = self.policy.Qpolicy(observation=obs, actions=actions_eval,
                                                               agent_ids=IDs, agent_key=key,
                                                               rnn_hidden_critic_1=rnn_hidden_critic,
                                                               rnn_hidden_critic_2=rnn_hidden_critic)
            log_pi_eval_i = log_pi_eval[key][:, :-1].reshape(bs_rnn, seq_len)
            policy_q = torch.min(policy_q_1[key][:, :-1], policy_q_2[key][:, :-1]).reshape(bs_rnn, seq_len)
            loss_a = ((self.alpha[key] * log_pi_eval_i - policy_q) * mask_values).sum() / mask_values.sum()
            self.optimizer[key]['actor'].zero_grad()
            loss_a.backward()
            if self.use_grad_clip:
                torch.nn.utils.clip_grad_norm_(self.policy.parameters_actor[key], self.grad_clip_norm)
            self.optimizer[key]['actor'].step()
            if self.scheduler[key]['actor'] is not None:
                self.scheduler[key]['actor'].step()

            # automatic entropy tuning
            if self.use_automatic_entropy_tuning:
                alpha_loss = -(self.log_alpha[key] * (log_pi_eval_i + self.target_entropy[key]).detach()).mean()
                self.alpha_optimizer[key].zero_grad()
                alpha_loss.backward()
                self.alpha_optimizer[key].step()
                self.alpha = self.log_alpha[key].exp()
            else:
                alpha_loss = 0

            learning_rate_actor = self.optimizer[key]['actor'].state_dict()['param_groups'][0]['lr']
            learning_rate_critic = self.optimizer[key]['critic'].state_dict()['param_groups'][0]['lr']

            info.update({
                f"{key}/learning_rate_actor": learning_rate_actor,
                f"{key}/learning_rate_critic": learning_rate_critic,
                f"{key}/loss_actor": loss_a.item(),
                f"{key}/loss_critic": loss_c.item(),
                f"{key}/predictQ": policy_q.mean().item(),
                f"{key}/alpha_loss": alpha_loss.item(),
                f"{key}/alpha": self.alpha[key].item(),
            })

        self.policy.soft_update(self.tau)
        return info
