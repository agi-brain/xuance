"""
Deep Deterministic Policy Gradient (DDPG)
Paper link: https://arxiv.org/pdf/1509.02971.pdf
Implementation: Pytorch
"""
import torch
from torch import nn
from xuance.torch.learners import Learner
from argparse import Namespace


class DDPG_Learner(Learner):
    def __init__(self,
                 config: Namespace,
                 model: nn.Module,
                 callback):
        super(DDPG_Learner, self).__init__(config, model, callback)
        self.optimizer = {
            'actor': torch.optim.Adam(self.model.actor.parameters(), self.config.learning_rate_actor),
            'critic': torch.optim.Adam(self.model.critic.parameters(), self.config.learning_rate_critic)}
        self.scheduler = {
            'actor': torch.optim.lr_scheduler.LinearLR(self.optimizer['actor'],
                                                       start_factor=1.0,
                                                       end_factor=self.end_factor_lr_decay,
                                                       total_iters=self.total_iters),
            'critic': torch.optim.lr_scheduler.LinearLR(self.optimizer['critic'],
                                                        start_factor=1.0,
                                                        end_factor=self.end_factor_lr_decay,
                                                        total_iters=self.total_iters)}
        self.tau = config.tau
        self.gamma = config.gamma
        self.mse_loss = nn.MSELoss()

    def update(self, **samples):
        self.iterations += 1
        obs_batch = torch.as_tensor(samples['obs'], device=self.device)
        act_batch = torch.as_tensor(samples['actions'], device=self.device)
        next_batch = torch.as_tensor(samples['obs_next'], device=self.device)
        rew_batch = torch.as_tensor(samples['rewards'], device=self.device)
        ter_batch = torch.as_tensor(samples['terminals'], dtype=torch.float, device=self.device)
        info = self.callback.on_update_start(self.iterations,
                                             model=self.model, obs=obs_batch, act=act_batch,
                                             next_obs=next_batch, rew=rew_batch, termination=ter_batch)

        # critic update
        action_q = self.model.Qaction(obs_batch, act_batch).reshape([-1])
        next_q = self.model.Qtarget(next_batch).reshape([-1])
        target_q = rew_batch + (1 - ter_batch) * self.gamma * next_q
        q_loss = self.mse_loss(action_q, target_q.detach())
        self.optimizer['critic'].zero_grad()
        q_loss.backward()
        if self.use_grad_clip:
            torch.nn.utils.clip_grad_norm_(self.model.critic.parameters(), self.grad_clip_norm)
        self.optimizer['critic'].step()

        # actor update
        model_q = self.model.Qpolicy(obs_batch)
        p_loss = -model_q.mean()
        self.optimizer['actor'].zero_grad()
        p_loss.backward()
        if self.use_grad_clip:
            torch.nn.utils.clip_grad_norm_(self.model.actor.parameters(), self.grad_clip_norm)
        self.optimizer['actor'].step()

        if self.scheduler is not None:
            self.scheduler['actor'].step()
            self.scheduler['critic'].step()

        self.model.soft_update(self.tau)

        actor_lr = self.optimizer['actor'].state_dict()['param_groups'][0]['lr']
        critic_lr = self.optimizer['critic'].state_dict()['param_groups'][0]['lr']

        if self.distributed_training:
            info.update({
                f"Qloss/rank_{self.rank}": q_loss.item(),
                f"Ploss/rank_{self.rank}": p_loss.item(),
                f"Qvalue/rank_{self.rank}": action_q.mean().item(),
                f"actor_lr/rank_{self.rank}": actor_lr,
                f"critic_lr/rank_{self.rank}": critic_lr
            })
        else:
            info.update({
                "Qloss": q_loss.item(),
                "Ploss": p_loss.item(),
                "Qvalue": action_q.mean().item(),
                "actor_lr": actor_lr,
                "critic_lr": critic_lr
            })
        info.update(self.callback.on_update_end(self.iterations,
                                                model=self.model, info=info,
                                                action_q=action_q, next_q=next_q, target_q=target_q, model_q=model_q,
                                                q_loss=q_loss, p_loss=p_loss))
        return info
