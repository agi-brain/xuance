"""
MFQ: Mean Field Q-Learning
Paper link:
http://proceedings.mlr.press/v80/yang18d/yang18d.pdf
Implementation: Pytorch
"""
import torch
from torch import nn
from xuance.torch.learners import LearnerMAS
from xuance.common import List, Optional, Union
from argparse import Namespace


class MFQ_Learner(LearnerMAS):
    def __init__(self,
                 config: Namespace,
                 model_keys: List[str],
                 agent_keys: List[str],
                 policy: nn.Module,
                 callback):
        super(MFQ_Learner, self).__init__(config, model_keys, agent_keys, policy, callback)
        self.optimizer = {key: torch.optim.Adam(self.policy.parameters_model[key], config.learning_rate, eps=1e-5)
                          for key in self.model_keys}
        self.scheduler = {key: torch.optim.lr_scheduler.LinearLR(self.optimizer[key],
                                                                 start_factor=1.0,
                                                                 end_factor=self.end_factor_lr_decay,
                                                                 total_iters=self.config.running_steps)
                          for key in self.model_keys}
        self.gamma = config.gamma
        self.sync_frequency = config.sync_frequency
        self.n_actions = {k: self.policy.action_space[k].n for k in self.model_keys}
        self.temperature = config.temperature
        self.mse_loss = nn.MSELoss()
        self.softmax = torch.nn.Softmax(dim=-1)

    def get_boltzmann_policy(self, q):
        return self.softmax(q / self.temperature)

    def update(self, sample):
        self.iterations += 1
        obs = torch.Tensor(sample['obs']).to(self.device)
        actions = torch.Tensor(sample['actions']).to(self.device)
        obs_next = torch.Tensor(sample['obs_next']).to(self.device)
        act_mean = torch.Tensor(sample['act_mean']).to(self.device)
        act_mean_next = torch.Tensor(sample['act_mean_next']).to(self.device)
        rewards = torch.Tensor(sample['rewards']).to(self.device)
        terminals = torch.Tensor(sample['terminals']).float().reshape(-1, self.n_agents, 1).to(self.device)
        agent_mask = torch.Tensor(sample['agent_mask']).float().reshape(-1, self.n_agents, 1).to(self.device)
        IDs = torch.eye(self.n_agents).unsqueeze(0).expand(self.args.batch_size, -1, -1).to(self.device)

        act_mean = act_mean.unsqueeze(1).repeat([1, self.n_agents, 1])
        act_mean_next = act_mean_next.unsqueeze(1).repeat([1, self.n_agents, 1])

        info = self.callback.on_update_start(self.iterations, method="update", policy=self.policy)

        _, _, q_eval = self.policy(obs, act_mean, IDs)
        q_eval_a = q_eval.gather(-1, actions.long().reshape([self.args.batch_size, self.n_agents, 1]))
        q_next = self.policy.target_Q(obs_next, act_mean_next, IDs)
        shape = q_next.shape
        pi = self.get_boltzmann_policy(q_next)
        v_mf = torch.bmm(q_next.reshape(-1, 1, shape[-1]), pi.unsqueeze(-1).reshape(-1, shape[-1], 1))
        v_mf = v_mf.reshape(*(list(shape[0:-1]) + [1]))
        q_target = rewards + (1 - terminals) * self.args.gamma * v_mf

        # calculate the loss function
        td_error = (q_eval_a - q_target.detach()) * agent_mask
        loss = (td_error ** 2).sum() / agent_mask.sum()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        if self.scheduler is not None:
            self.scheduler.step()

        if self.iterations % self.sync_frequency == 0:
            self.policy.copy_target()

        lr = self.optimizer.state_dict()['param_groups'][0]['lr']

        info.update({
            "learning_rate": lr,
            "loss_Q": loss.item(),
            "predictQ": q_eval_a.mean().item()
        })

        info.update(self.callback.on_update_end(self.iterations, method="update", policy=self.policy, info=info))

        return info
