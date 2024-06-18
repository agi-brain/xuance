"""
Independent Deep Deterministic Policy Gradient (IDDPG)
Implementation: Pytorch
"""
import torch
from torch import nn
from xuance.torch.learners import LearnerMAS
from typing import Optional, List
from argparse import Namespace


class IDDPG_Learner(LearnerMAS):
    def __init__(self,
                 config: Namespace,
                 model_keys: List[str],
                 agent_keys: List[str],
                 episode_length: int,
                 policy: nn.Module,
                 optimizer: Optional[dict],
                 scheduler: Optional[dict] = None):
        self.gamma = config.gamma
        self.tau = config.tau
        self.mse_loss = nn.MSELoss()
        super(IDDPG_Learner, self).__init__(config, model_keys, agent_keys, episode_length,
                                            policy, optimizer, scheduler)
        self.optimizer = {key: {'actor': optimizer[key][0],
                                'critic': optimizer[key][1]} for key in self.model_keys}
        self.scheduler = {key: {'actor': scheduler[key][0],
                                'critic': scheduler[key][1]} for key in self.model_keys}

    def update(self, sample):
        self.iterations += 1
        info = {}

        # prepare training data.
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

        # train the model
        _, actions_eval = self.policy(observation=obs, agent_ids=IDs)
        _, q_policy = self.policy.Qpolicy(observation=obs, actions=actions_eval, agent_ids=IDs)
        _, q_eval = self.policy.Qpolicy(observation=obs, actions=actions, agent_ids=IDs)
        _, next_actions = self.policy.Atarget(next_observation=obs_next, agent_ids=IDs)
        _, q_next = self.policy.Qtarget(next_observation=obs_next, next_actions=next_actions, agent_ids=IDs)

        for key in self.model_keys:
            # update actor
            q_policy_i = q_policy[key].reshape(bs)
            loss_a = -(q_policy_i * agent_mask[key]).sum() / agent_mask[key].sum()
            self.optimizer[key]['actor'].zero_grad()
            loss_a.backward()
            if self.use_grad_clip:
                torch.nn.utils.clip_grad_norm_(self.policy.parameters_actor[key], self.grad_clip_norm)
            self.optimizer[key]['actor'].step()
            if self.scheduler[key]['actor'] is not None:
                self.scheduler[key]['actor'].step()

            # updata critic
            q_eval_a = q_eval[key].reshape(bs)
            q_target = rewards[key] + (1 - terminals[key]) * self.gamma * q_next[key]
            td_error = (q_eval_a - q_target.detach()) * agent_mask[key]
            loss_c = (td_error ** 2).sum() / agent_mask[key].sum()
            self.optimizer[key]['critic'].zero_grad()
            loss_c.backward()
            if self.use_grad_clip:
                torch.nn.utils.clip_grad_norm_(self.policy.parameters_critic[key], self.grad_clip_norm)
            self.optimizer[key]['critic'].step()
            if self.scheduler[key]['critic'] is not None:
                self.scheduler[key]['critic'].step()

            lr_a = self.optimizer[key]['actor'].state_dict()['param_groups'][0]['lr']
            lr_c = self.optimizer[key]['critic'].state_dict()['param_groups'][0]['lr']

            info.update({
                f"{key}/learning_rate_actor": lr_a,
                f"{key}/learning_rate_critic": lr_c,
                f"{key}/loss_actor": loss_a.item(),
                f"{key}/loss_critic": loss_c.item(),
                f"{key}/predictQ": q_eval[key].mean().item()
            })

        self.policy.soft_update(self.tau)
        return info

    def update_rnn(self, *args):
        raise NotImplementedError
