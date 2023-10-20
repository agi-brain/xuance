from xuance.torch.learners import *


class SACDIS_Learner(Learner):
    def __init__(self,
                 policy: nn.Module,
                 optimizers: Sequence[torch.optim.Optimizer],
                 schedulers: Sequence[torch.optim.lr_scheduler._LRScheduler],
                 device: Optional[Union[int, str, torch.device]] = None,
                 model_dir: str = "./",
                 gamma: float = 0.99,
                 tau: float = 0.01):
        self.tau = tau
        self.gamma = gamma
        super(SACDIS_Learner, self).__init__(policy, optimizers, schedulers, device, model_dir)

    def update(self, obs_batch, act_batch, rew_batch, next_batch, terminal_batch):
        self.iterations += 1
        act_batch = torch.as_tensor(act_batch, device=self.device)
        rew_batch = torch.as_tensor(rew_batch, device=self.device)
        ter_batch = torch.as_tensor(terminal_batch, device=self.device).reshape([-1, 1])
        act_batch = torch.unsqueeze(act_batch, -1)
        # critic update
        _, action_q = self.policy.Qaction(obs_batch)
        action_q = action_q.gather(1, act_batch.long())
        # with torch.no_grad():
        action_prob_next, log_pi_next, target_q = self.policy.Qtarget(next_batch)
        target_q = action_prob_next * (target_q - 0.01 * log_pi_next)
        target_q = target_q.sum(dim=1).unsqueeze(-1)
        rew = torch.unsqueeze(rew_batch, -1)
        backup = rew + (1 - ter_batch) * self.gamma * target_q
        q_loss = F.mse_loss(action_q, backup.detach())
        self.optimizer[1].zero_grad()
        q_loss.backward()
        self.optimizer[1].step()

        # actor update
        action_prob, log_pi, policy_q = self.policy.Qpolicy(obs_batch)
        inside_term = 0.01 * log_pi - policy_q
        p_loss = (action_prob * inside_term).sum(dim=1).mean()
        # p_loss = (inside_term).sum(dim=1).mean()
        # p_loss = (0.01 * log_pi - policy_q).mean()
        self.optimizer[0].zero_grad()
        p_loss.backward()
        self.optimizer[0].step()

        if self.scheduler is not None:
            self.scheduler[0].step()
            self.scheduler[1].step()

        self.policy.soft_update(self.tau)

        actor_lr = self.optimizer[0].state_dict()['param_groups'][0]['lr']
        critic_lr = self.optimizer[1].state_dict()['param_groups'][0]['lr']

        info = {
            "Qloss": q_loss.item(),
            "Ploss": p_loss.item(),
            "Qvalue": action_q.mean().item(),
            "actor_lr": actor_lr,
            "critic_lr": critic_lr
        }

        return info
