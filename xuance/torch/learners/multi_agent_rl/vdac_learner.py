"""
Value-Dcomposition Actor-Critic (VDAC)
Paper link:
https://ojs.aaai.org/index.php/AAAI/article/view/17353
Implementation: Pytorch
"""
import torch
from torch import nn
from xuance.torch.learners import LearnerMAS
from typing import Optional, Union
from argparse import Namespace
from xuance.torch.utils.value_norm import ValueNorm
from xuance.torch.utils.operations import update_linear_decay


class VDAC_Learner(LearnerMAS):
    def __init__(self,
                 config: Namespace,
                 policy: nn.Module,
                 optimizer: torch.optim.Optimizer,
                 scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
                 device: Optional[Union[int, str, torch.device]] = None,
                 model_dir: str = "./",
                 gamma: float = 0.99,
                 ):
        self.gamma = gamma
        self.clip_range = config.clip_range
        self.use_linear_lr_decay = config.use_linear_lr_decay
        self.use_grad_clip, self.grad_clip_norm = config.use_grad_clip, config.grad_clip_norm
        self.use_value_norm = config.use_value_norm
        self.vf_coef, self.ent_coef = config.vf_coef, config.ent_coef
        super(VDAC_Learner, self).__init__(config, policy, optimizer, scheduler, device, model_dir)
        if self.use_value_norm:
            self.value_normalizer = ValueNorm(1).to(device)
        else:
            self.value_normalizer = None
        self.lr = config.learning_rate
        self.end_factor_lr_decay = config.end_factor_lr_decay

    def lr_decay(self, i_step):
        if self.use_linear_lr_decay:
            update_linear_decay(self.optimizer, i_step, self.running_steps, self.lr, self.end_factor_lr_decay)

    def update(self, sample):
        info = {}
        self.iterations += 1
        state = torch.Tensor(sample['state']).to(self.device)
        obs = torch.Tensor(sample['obs']).to(self.device)
        actions = torch.Tensor(sample['actions']).to(self.device)
        returns = torch.Tensor(sample['returns']).to(self.device)
        agent_mask = torch.Tensor(sample['agent_mask']).float().reshape(-1, self.n_agents, 1).to(self.device)
        batch_size = obs.shape[0]
        IDs = torch.eye(self.n_agents).unsqueeze(0).expand(batch_size, -1, -1).to(self.device)

        # actor loss
        _, pi_dist, value_pred = self.policy(obs, IDs)
        log_pi = pi_dist.log_prob(actions).unsqueeze(-1)
        entropy = pi_dist.entropy().reshape(agent_mask.shape) * agent_mask

        targets = returns
        advantages = targets - value_pred
        td_error = value_pred - targets.detach()

        pg_loss = -((advantages.detach() * log_pi) * agent_mask).sum() / agent_mask.sum()
        vf_loss = ((td_error ** 2) * agent_mask).sum() / agent_mask.sum()
        entropy_loss = (entropy * agent_mask).sum() / agent_mask.sum()
        loss = pg_loss + self.vf_coef * vf_loss - self.ent_coef * entropy_loss

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
            "pg_loss": pg_loss.item(),
            "vf_loss": vf_loss.item(),
            "entropy_loss": entropy_loss.item(),
            "loss": loss.item(),
            "predict_value": value_pred.mean().item()
        })

        return info

    def update_recurrent(self, sample):
        info = {}
        self.iterations += 1
        state = torch.Tensor(sample['state']).to(self.device)
        obs = torch.Tensor(sample['obs']).to(self.device)
        actions = torch.Tensor(sample['actions']).to(self.device)
        returns = torch.Tensor(sample['returns']).to(self.device)
        avail_actions = torch.Tensor(sample['avail_actions']).float().to(self.device)
        filled = torch.Tensor(sample['filled']).float().to(self.device)
        batch_size = obs.shape[0]
        episode_length = actions.shape[2]
        IDs = torch.eye(self.n_agents).unsqueeze(1).unsqueeze(0).expand(batch_size, -1, episode_length + 1, -1).to(
            self.device)

        filled_n = filled.unsqueeze(1).expand(batch_size, self.n_agents, episode_length, 1)

        # actor loss
        rnn_hidden = self.policy.representation.init_hidden(batch_size * self.n_agents)
        _, pi_dist, value_pred = self.policy(obs[:, :, :-1].reshape(-1, episode_length, self.dim_obs),
                                             IDs[:, :, :-1].reshape(-1, episode_length, self.n_agents),
                                             *rnn_hidden,
                                             avail_actions=avail_actions[:, :, :-1].reshape(-1, episode_length, self.dim_act),
                                             state=state[:, :-1])
        log_pi = pi_dist.log_prob(actions.reshape(-1, episode_length)).reshape(batch_size, self.n_agents, episode_length, 1)
        entropy = pi_dist.entropy().reshape(batch_size, self.n_agents, episode_length, 1)

        targets = returns
        advantages = targets - value_pred
        td_error = value_pred - targets.detach()

        pg_loss = -((advantages.detach() * log_pi) * filled_n).sum() / filled_n.sum()
        vf_loss = ((td_error ** 2) * filled_n).sum() / filled_n.sum()
        entropy_loss = (entropy * filled_n).sum() / filled_n.sum()
        loss = pg_loss + self.vf_coef * vf_loss - self.ent_coef * entropy_loss

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
            "pg_loss": pg_loss.item(),
            "vf_loss": vf_loss.item(),
            "entropy_loss": entropy_loss.item(),
            "loss": loss.item(),
            "predict_value": value_pred.mean().item()
        })

        return info
