"""
Independent Soft Actor-critic (ISAC)
Implementation: Pytorch
"""
import torch
from torch import nn
from xuance.torch.learners import LearnerMAS
from typing import Optional, List
from argparse import Namespace


class ISAC_Learner(LearnerMAS):
    def __init__(self,
                 config: Namespace,
                 model_keys: List[str],
                 agent_keys: List[str],
                 episode_length: int,
                 policy: nn.Module,
                 optimizer: Optional[dict],
                 scheduler: Optional[dict] = None):
        super(ISAC_Learner, self).__init__(config, model_keys, agent_keys, episode_length, policy, optimizer, scheduler)
        self.gamma = config.gamma
        self.tau = config.tau
        self.alpha = config.alpha
        self.mse_loss = nn.MSELoss()
        self.use_automatic_entropy_tuning = config.use_automatic_entropy_tuning
        self.optimizer = {key: {'actor': optimizer[key][0],
                                'critic': optimizer[key][1]} for key in self.model_keys}
        self.scheduler = {key: {'actor': scheduler[key][0],
                                'critic': scheduler[key][1]} for key in self.model_keys}
        if self.use_automatic_entropy_tuning:
            self.target_entropy = -policy.action_space[self.agent_keys[0]].shape[-1]
            self.log_alpha = nn.Parameter(torch.zeros(1, requires_grad=True, device=self.device))
            self.alpha = self.log_alpha.exp()
            self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=config.lr_a)

    def update(self, sample):
        self.iterations += 1
        info = {}

        # Prepare training data.
        sample_Tensor = self.build_training_data(sample,
                                                 use_parameter_sharing=self.use_parameter_sharing,
                                                 use_actions_mask=False)
        obs = sample_Tensor['obs']
        actions = sample_Tensor['actions']
        obs_next = sample_Tensor['obs_next']
        rewards = sample_Tensor['rewards']
        terminals = sample_Tensor['terminals']
        agent_mask = sample_Tensor['agent_mask']
        IDs = sample_Tensor['agent_ids']

        # train the model
        log_pi, policy_q_1, policy_q_2 = self.policy.Qpolicy(observation=obs, agent_ids=IDs)
        for key in self.model_keys:
            # actor update
            log_pi_eval = log_pi[key].unsqueeze(-1)
            policy_q = torch.min(policy_q_1[key], policy_q_2[key])
            loss_a = ((self.alpha * log_pi_eval - policy_q.detach()) * agent_mask[key]).sum() / agent_mask[key].sum()
            self.optimizer[key]['actor'].zero_grad()
            loss_a.backward()
            if self.use_grad_clip:
                torch.nn.utils.clip_grad_norm_(self.policy.parameters_actor[key], self.grad_clip_norm)
            self.optimizer[key]['actor'].step()
            if self.scheduler[key]['actor'] is not None:
                self.scheduler[key]['actor'].step()

            # critic update
            action_q_1, action_q_2 = self.policy.Qaction(observation=obs, actions=actions,
                                                         agent_ids=IDs, agent_key=key)
            log_pi_next, target_q = self.policy.Qtarget(next_observation=obs_next, agent_ids=IDs, agent_key=key)
            log_pi_next_eval = log_pi_next[key].unsqueeze(-1)
            target_value = target_q[key] - self.alpha * log_pi_next_eval
            backup = rewards[key] + (1 - terminals[key]) * self.gamma * target_value
            td_error_1, td_error_2 = action_q_1[key] - backup.detach(), action_q_2[key] - backup.detach()
            td_error_1 *= agent_mask[key]
            td_error_2 *= agent_mask[key]
            loss_c = ((td_error_1 ** 2).sum() + (td_error_2 ** 2).sum()) / agent_mask[key].sum()
            self.optimizer[key]['critic'].zero_grad()
            loss_c.backward()
            if self.use_grad_clip:
                torch.nn.utils.clip_grad_norm_(self.policy.parameters_critic[key], self.grad_clip_norm)
            self.optimizer[key]['critic'].step()
            if self.scheduler[key]['critic'] is not None:
                self.scheduler[key]['critic'].step()

            # automatic entropy tuning
            if self.use_automatic_entropy_tuning:
                alpha_loss = -(self.log_alpha * (log_pi[key] + self.target_entropy).detach()).mean()
                self.alpha_optimizer.zero_grad()
                alpha_loss.backward()
                self.alpha_optimizer.step()
                self.alpha = self.log_alpha.exp()
            else:
                alpha_loss = 0

            lr_a = self.optimizer[key]['actor'].state_dict()['param_groups'][0]['lr']
            lr_c = self.optimizer[key]['critic'].state_dict()['param_groups'][0]['lr']

            info.update({
                f"{key}/learning_rate_actor": lr_a,
                f"{key}/learning_rate_critic": lr_c,
                f"{key}/loss_actor": loss_a.item(),
                f"{key}/loss_critic": loss_c.item(),
                f"{key}/predictQ": policy_q.mean().item(),
                f"{key}/alpha_loss": alpha_loss.item(),
                f"{key}/alpha": self.alpha.item(),
            })

        self.policy.soft_update(self.tau)
        return info

    def update_rnn(self, *args):
        raise NotImplementedError
