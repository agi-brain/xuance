from xuanpolicy.xuanpolicy_torch.learners import *


class DDPG_Learner(Learner):
    def __init__(self,
                 policy: nn.Module,
                 optimizers: Sequence[torch.optim.Optimizer],
                 schedulers: Sequence[torch.optim.lr_scheduler._LRScheduler],
                 summary_writer: Optional[SummaryWriter] = None,
                 device: Optional[Union[int, str, torch.device]] = None,
                 modeldir: str = "./",
                 gamma: float = 0.99,
                 tau: float = 0.01):
        self.tau = tau
        self.gamma = gamma
        super(DDPG_Learner, self).__init__(policy, optimizers, schedulers, summary_writer, device, modeldir)

    def update(self, obs_batch, act_batch, rew_batch, next_batch, terminal_batch):
        self.iterations += 1
        act_batch = torch.as_tensor(act_batch, device=self.device)
        rew_batch = torch.as_tensor(rew_batch, device=self.device)
        ter_batch = torch.as_tensor(terminal_batch, device=self.device)
        # critic update
        _, action_q = self.policy.Qaction(obs_batch, act_batch)
        # with torch.no_grad():
        _, target_q = self.policy.Qtarget(next_batch)
        backup = rew_batch + self.gamma * target_q
        q_loss = F.mse_loss(backup.detach(), action_q)
        self.optimizer[1].zero_grad()
        q_loss.backward()
        self.optimizer[1].step()

        # actor update
        _, policy_q = self.policy.Qpolicy(obs_batch)
        p_loss = -policy_q.mean()
        self.optimizer[0].zero_grad()
        p_loss.backward()
        self.optimizer[0].step()

        if self.scheduler is not None:
            self.scheduler[0].step()
            self.scheduler[1].step()

        self.policy.soft_update(self.tau)

        actor_lr = self.optimizer[0].state_dict()['param_groups'][0]['lr']
        critic_lr = self.optimizer[1].state_dict()['param_groups'][0]['lr']
        self.writer.add_scalar("Qloss", q_loss.item(), self.iterations)
        self.writer.add_scalar("Ploss", p_loss.item(), self.iterations)
        self.writer.add_scalar("Qvalue", action_q.mean().item(), self.iterations)
        self.writer.add_scalar("actor_lr", actor_lr, self.iterations)
        self.writer.add_scalar("critic_lr", critic_lr, self.iterations)
