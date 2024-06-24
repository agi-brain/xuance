"""
Multi-agent Soft Actor-critic (MASAC)
Implementation: Pytorch
"""
import torch
from torch import nn
from numpy import concatenate
from typing import Optional, List
from argparse import Namespace
from xuance.torch.learners.multi_agent_rl.isac_learner import ISAC_Learner
from operator import itemgetter
from xuance.torch import Tensor


class MASAC_Learner(ISAC_Learner):
    def __init__(self,
                 config: Namespace,
                 model_keys: List[str],
                 agent_keys: List[str],
                 episode_length: int,
                 policy: nn.Module,
                 optimizer: Optional[dict],
                 scheduler: Optional[dict] = None):
        super(MASAC_Learner, self).__init__(config, model_keys, agent_keys, episode_length,
                                            policy, optimizer, scheduler)

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
            joint_actions = actions[key].reshape(batch_size, -1)
            rewards[key] = rewards[key].reshape(batch_size * self.n_agents)
            terminals[key] = terminals[key].reshape(batch_size * self.n_agents)
        else:
            bs = batch_size
            joint_actions = torch.concat(itemgetter(*self.agent_keys)(actions), dim=-1).reshape(batch_size, -1)

        obs_joint = Tensor(concatenate(itemgetter(*self.agent_keys)(sample['obs']), axis=-1)).to(self.device)
        next_obs_joint = Tensor(concatenate(itemgetter(*self.agent_keys)(sample['obs_next']), axis=-1)).to(self.device)

        # train the model
        action_q_1, action_q_2 = self.policy.Qaction(observation=obs, joint_observation=obs_joint,
                                                     joint_actions=joint_actions, agent_ids=IDs)
        log_pi_next, target_q = self.policy.Qtarget(next_observation=obs_next, joint_observation=next_obs_joint,
                                                    agent_ids=IDs)
        for key in self.model_keys:
            mask_values = agent_mask[key]
            # critic update
            action_q_1_i = action_q_1[key].reshape(bs)
            action_q_2_i = action_q_2[key].reshape(bs)
            log_pi_next_eval = log_pi_next[key].reshape(bs)
            target_value = target_q[key].reshape(bs) - self.alpha * log_pi_next_eval
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
            log_pi, policy_q_1, policy_q_2 = self.policy.Qpolicy(observation=obs, joint_observation=obs_joint,
                                                                 agent_ids=IDs, agent_key=key)
            log_pi_eval = log_pi[key].reshape(bs)
            policy_q = torch.min(policy_q_1[key], policy_q_2[key]).reshape(bs)
            loss_a = ((self.alpha * log_pi_eval - policy_q) * mask_values).sum() / mask_values.sum()
            self.optimizer[key]['actor'].zero_grad()
            loss_a.backward()
            if self.use_grad_clip:
                torch.nn.utils.clip_grad_norm_(self.policy.parameters_actor[key], self.grad_clip_norm)
            self.optimizer[key]['actor'].step()
            if self.scheduler[key]['actor'] is not None:
                self.scheduler[key]['actor'].step()

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

    def update_rnn(self, sample):
        return
