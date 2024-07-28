"""
Multi-agent Soft Actor-critic (MASAC)
Implementation: Pytorch
"""
import torch
from torch import nn
from xuance.common import List
from argparse import Namespace
from xuance.torch.learners.multi_agent_rl.isac_learner import ISAC_Learner
from operator import itemgetter


class MASAC_Learner(ISAC_Learner):
    def __init__(self,
                 config: Namespace,
                 model_keys: List[str],
                 agent_keys: List[str],
                 policy: nn.Module):
        super(MASAC_Learner, self).__init__(config, model_keys, agent_keys, policy)

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
            obs_joint = obs[key].reshape(batch_size, -1)
            next_obs_joint = obs_next[key].reshape(batch_size, -1)
            actions_joint = actions[key].reshape(batch_size, -1)
            rewards[key] = rewards[key].reshape(batch_size * self.n_agents)
            terminals[key] = terminals[key].reshape(batch_size * self.n_agents)
        else:
            bs = batch_size
            obs_joint = torch.concat(itemgetter(*self.agent_keys)(obs), dim=-1).reshape(batch_size, -1)
            next_obs_joint = torch.concat(itemgetter(*self.agent_keys)(obs_next), dim=-1).reshape(batch_size, -1)
            actions_joint = torch.concat(itemgetter(*self.agent_keys)(actions), dim=-1).reshape(batch_size, -1)

        # train the model
        _, actions_eval, log_pi_eval = self.policy(observation=obs, agent_ids=IDs)
        _, actions_next, log_pi_next = self.policy(observation=obs_next, agent_ids=IDs)
        if self.use_parameter_sharing:
            key = self.model_keys[0]
            actions_next_joint = actions_next[key].reshape(batch_size, self.n_agents, -1).reshape(batch_size, -1)
        else:
            actions_next_joint = torch.concat(itemgetter(*self.model_keys)(actions_next), -1).reshape(batch_size, -1)
        _, _, action_q_1, action_q_2 = self.policy.Qaction(joint_observation=obs_joint, joint_actions=actions_joint,
                                                           agent_ids=IDs)
        _, _, target_q = self.policy.Qtarget(joint_observation=next_obs_joint, joint_actions=actions_next_joint,
                                             agent_ids=IDs)
        for key in self.model_keys:
            mask_values = agent_mask[key]
            # critic update
            action_q_1_i = action_q_1[key].reshape(bs)
            action_q_2_i = action_q_2[key].reshape(bs)
            log_pi_next_eval = log_pi_next[key].reshape(bs)
            target_value = target_q[key].reshape(bs) - self.alpha[key] * log_pi_next_eval
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

            # actor update
            if self.use_parameter_sharing:
                actions_eval_joint = actions_eval[key].reshape(batch_size, self.n_agents, -1).reshape(batch_size, -1)
            else:
                actions_eval_detach_others = {k: actions_eval[k] if k == key else actions_eval[k].detach()
                                              for k in self.model_keys}
                actions_eval_joint = torch.concat(itemgetter(*self.model_keys)(actions_eval_detach_others),
                                                  dim=-1).reshape(batch_size, -1)
            _, _, policy_q_1, policy_q_2 = self.policy.Qpolicy(joint_observation=obs_joint,
                                                               joint_actions=actions_eval_joint,
                                                               agent_ids=IDs, agent_key=key)
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
            obs_joint = obs[key].reshape(batch_size, self.n_agents, seq_len + 1, -1).transpose(
                1, 2).reshape(batch_size, seq_len + 1, -1)
            actions_joint = actions[key].reshape(batch_size, self.n_agents, seq_len, -1).transpose(
                1, 2).reshape(batch_size, seq_len, -1)
            rewards[key] = rewards[key].reshape(bs_rnn, seq_len)
            terminals[key] = terminals[key].reshape(bs_rnn, seq_len)
            IDs_t = IDs[:, :-1]
        else:
            bs_rnn, IDs_t = batch_size, None
            obs_joint = torch.concat(itemgetter(*self.agent_keys)(obs), dim=-1).reshape(batch_size, seq_len + 1, -1)
            actions_joint = torch.concat(itemgetter(*self.agent_keys)(actions), dim=-1).reshape(batch_size, seq_len, -1)

        # initial hidden states for rnn
        rnn_hidden_actor = {k: self.policy.actor_representation[k].init_hidden(bs_rnn) for k in self.model_keys}
        rnn_hidden_critic = {k: self.policy.critic_1_representation[k].init_hidden(batch_size) for k in self.model_keys}

        _, actions_eval, log_pi_eval = self.policy(observation=obs, agent_ids=IDs, rnn_hidden=rnn_hidden_actor)
        if self.use_parameter_sharing:
            key = self.model_keys[0]
            actions_eval_joint = actions_eval[key].reshape(batch_size, self.n_agents, seq_len + 1, -1).transpose(
                1, 2).reshape(batch_size, seq_len + 1, -1)
        else:
            actions_eval_joint = torch.concat(itemgetter(*self.agent_keys)(actions_eval),
                                              dim=-1).reshape(batch_size, seq_len + 1, -1)
        _, _, action_q_1, action_q_2 = self.policy.Qaction(joint_observation=obs_joint[:, :-1],
                                                           joint_actions=actions_joint,
                                                           agent_ids=IDs_t,
                                                           rnn_hidden_critic_1=rnn_hidden_critic,
                                                           rnn_hidden_critic_2=rnn_hidden_critic)
        _, _, target_q = self.policy.Qtarget(joint_observation=obs_joint, joint_actions=actions_eval_joint,
                                             agent_ids=IDs,
                                             rnn_hidden_critic_1=rnn_hidden_critic,
                                             rnn_hidden_critic_2=rnn_hidden_critic)
        for key in self.model_keys:
            mask_values = agent_mask[key] * filled
            # critic update
            action_q_1_i = action_q_1[key].reshape(bs_rnn, seq_len)
            action_q_2_i = action_q_2[key].reshape(bs_rnn, seq_len)
            log_pi_next_eval = log_pi_eval[key][:, 1:].reshape(bs_rnn, seq_len)
            target_value = target_q[key][:, 1:].reshape(bs_rnn, seq_len) - self.alpha[key] * log_pi_next_eval
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

            # actor update
            if self.use_parameter_sharing:
                actions_eval_joint = actions_eval_joint[:, :-1]
            else:
                actions_eval_detach_others = {k: actions_eval[k] if k == key else actions_eval[k].detach()
                                              for k in self.model_keys}
                actions_eval_joint = torch.concat(itemgetter(*self.model_keys)(actions_eval_detach_others),
                                                  dim=-1).reshape(batch_size, seq_len + 1, -1)[:, :-1]
            _, _, policy_q_1, policy_q_2 = self.policy.Qpolicy(joint_observation=obs_joint[:, :-1],
                                                               joint_actions=actions_eval_joint,
                                                               agent_ids=IDs_t, agent_key=key,
                                                               rnn_hidden_critic_1=rnn_hidden_critic,
                                                               rnn_hidden_critic_2=rnn_hidden_critic)
            log_pi_eval_i = log_pi_eval[key][:, :-1].reshape(bs_rnn, seq_len)
            policy_q = torch.min(policy_q_1[key], policy_q_2[key]).reshape(bs_rnn, seq_len)
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
