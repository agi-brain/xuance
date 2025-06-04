"""
Independent Soft Actor-critic (ISAC) with discrete action spaces.
Implementation: Pytorch
"""
import torch
from torch import nn
from xuance.torch.learners import LearnerMAS
from xuance.common import List
from argparse import Namespace


class ISACDIS_Learner(LearnerMAS):
    def __init__(self,
                 config: Namespace,
                 model_keys: List[str],
                 agent_keys: List[str],
                 policy: nn.Module,
                 callback):
        super(ISACDIS_Learner, self).__init__(config, model_keys, agent_keys, policy, callback)
        self.optimizer = {
            key: {'actor': torch.optim.Adam(self.policy.parameters_actor[key], self.config.learning_rate_actor, eps=1e-5),
                  'critic': torch.optim.Adam(self.policy.parameters_critic[key], self.config.learning_rate_critic, eps=1e-5)}
            for key in self.model_keys}
        self.scheduler = {
            key: {'actor': torch.optim.lr_scheduler.LinearLR(self.optimizer[key]['actor'],
                                                             start_factor=1.0,
                                                             end_factor=self.end_factor_lr_decay,
                                                             total_iters=self.config.running_steps),
                  'critic': torch.optim.lr_scheduler.LinearLR(self.optimizer[key]['critic'],
                                                              start_factor=1.0,
                                                              end_factor=self.end_factor_lr_decay,
                                                              total_iters=self.config.running_steps)}
            for key in self.model_keys}
        self.gamma = config.gamma
        self.tau = config.tau
        self.alpha = {key: config.alpha for key in self.model_keys}
        self.mse_loss = nn.MSELoss()
        self.use_automatic_entropy_tuning = config.use_automatic_entropy_tuning
        if self.use_automatic_entropy_tuning:
            self.target_entropy = {key: -float(policy.action_space[key].n) for key in self.model_keys}
            self.log_alpha = {key: nn.Parameter(torch.zeros(1, requires_grad=True, device=self.device))
                              for key in self.model_keys}
            self.alpha = {key: self.log_alpha[key].exp() for key in self.model_keys}
            self.alpha_optimizer = {key: torch.optim.Adam([self.log_alpha[key]], lr=config.learning_rate_actor)
                                    for key in self.model_keys}

    def update(self, sample):
        self.iterations += 1

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

        info = self.callback.on_update_start(self.iterations, method="update",
                                             policy=self.policy, sample_Tensor=sample_Tensor, bs=bs)

        # feedforward
        _, _, action_q_1, action_q_2 = self.policy.Qaction(observation=obs, agent_ids=IDs)
        _, _, _, action_prob_next, log_pi_next, next_q = self.policy.Qtarget(next_observation=obs_next, agent_ids=IDs)

        for key in self.model_keys:
            mask_values = agent_mask[key]
            # update critic
            action_q_1_a = action_q_1[key].gather(dim=-1, index=actions[key].long().unsqueeze(-1)).reshape(bs)
            action_q_2_a = action_q_2[key].gather(dim=-1, index=actions[key].long().unsqueeze(-1)).reshape(bs)
            target_value = action_prob_next[key] * (next_q[key] - self.alpha[key] * log_pi_next[key])
            target_value = target_value.sum(dim=1).reshape(bs)
            backup = rewards[key] + (1 - terminals[key]) * self.gamma * target_value
            td_error_1, td_error_2 = action_q_1_a - backup.detach(), action_q_2_a - backup.detach()
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
            _, _, _, action_prob, log_pi, policy_q_1, policy_q_2 = self.policy.Qpolicy(observation=obs,
                                                                                       agent_ids=IDs, agent_key=key)
            policy_q = torch.min(policy_q_1[key], policy_q_2[key])
            loss_a = (action_prob[key] * (self.alpha[key] * log_pi[key] - policy_q)).sum(dim=1)
            loss_a = (loss_a * mask_values).sum() / mask_values.sum()
            self.optimizer[key]['actor'].zero_grad()
            loss_a.backward()
            if self.use_grad_clip:
                torch.nn.utils.clip_grad_norm_(self.policy.parameters_actor[key], self.grad_clip_norm)
            self.optimizer[key]['actor'].step()
            if self.scheduler[key]['actor'] is not None:
                self.scheduler[key]['actor'].step()

            # automatic entropy tuning
            if self.use_automatic_entropy_tuning:
                alpha_loss = -(self.log_alpha[key] * (log_pi[key] + self.target_entropy[key]).detach()).mean()
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
            })
            if self.use_automatic_entropy_tuning:
                info.update({f"{key}/alpha_loss": alpha_loss.item(),
                             f"{key}/alpha": self.alpha[key].item()})

            info.update(self.callback.on_update_agent_wise(self.iterations, key, info=info, method="update",
                                                           mask_values=mask_values,
                                                           action_q_1_a=action_q_1_a, action_q_2_a=action_q_2_a,
                                                           target_value=target_value, backup=backup,
                                                           td_error_1=td_error_1, td_error_2=td_error_2,
                                                           action_prob=action_prob, log_pi=log_pi,
                                                           policy_q_1=policy_q_1, policy_q_2=policy_q_2,
                                                           policy_q=policy_q))

        self.policy.soft_update(self.tau)
        info.update(self.callback.on_update_end(self.iterations, method="update", policy=self.policy, info=info))
        return info

    def update_rnn(self, sample):
        self.iterations += 1
        info = {}
        return info
