from xuance.torch.learners import *


class SAC_Learner(Learner):
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
        super(SAC_Learner, self).__init__(policy, optimizers, schedulers, device, model_dir)

    def update(self, obs_batch, act_batch, rew_batch, next_batch, terminal_batch):
        self.iterations += 1
        act_batch = torch.as_tensor(act_batch, device=self.device)
        rew_batch = torch.as_tensor(rew_batch, device=self.device)
        ter_batch = torch.as_tensor(terminal_batch, device=self.device)
        # critic update
        action_q = self.policy.Qaction(obs_batch, act_batch)
        # with torch.no_grad():
        log_pi_next, target_q = self.policy.Qtarget(next_batch)
        backup = rew_batch + (1-ter_batch) * self.gamma * (target_q - 0.01 * log_pi_next.reshape([-1]))
        q_loss = F.mse_loss(action_q, backup.detach())
        self.optimizer[1].zero_grad()
        q_loss.backward()
        self.optimizer[1].step()

        # actor update
        log_pi, policy_q = self.policy.Qpolicy(obs_batch)
        p_loss = (0.01 * log_pi.reshape([-1]) - policy_q).mean()
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
