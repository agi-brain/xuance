from xuance.torch.learners import *


class SPDQN_Learner(Learner):
    def __init__(self,
                 policy: nn.Module,
                 optimizers: Sequence[torch.optim.Optimizer],
                 schedulers: Sequence[torch.optim.lr_scheduler._LRScheduler],
                 summary_writer: Optional[SummaryWriter] = None,
                 device: Optional[Union[int, str, torch.device]] = None,
                 model_dir: str = "./",
                 gamma: float = 0.99,
                 tau: float = 0.01):
        self.tau = tau
        self.gamma = gamma
        super(SPDQN_Learner, self).__init__(policy, optimizers, schedulers, summary_writer, device, model_dir)

    def update(self, obs_batch, act_batch, rew_batch, next_batch, terminal_batch):
        self.iterations += 1
        obs_batch = torch.as_tensor(obs_batch, device=self.device)
        hyact_batch = torch.as_tensor(act_batch, device=self.device)
        disact_batch = hyact_batch[:, 0].long()
        conact_batch = hyact_batch[:, 1:]
        rew_batch = torch.as_tensor(rew_batch, device=self.device)
        next_batch = torch.as_tensor(next_batch, device=self.device)
        ter_batch = torch.as_tensor(terminal_batch, device=self.device)

        # optimize Q-network
        with torch.no_grad():
            target_conact = self.policy.Atarget(next_batch)
            target_q = self.policy.Qtarget(next_batch, target_conact)
            target_q = torch.max(target_q, 1, keepdim=True)[0].squeeze()

            target_q = rew_batch + (1 - ter_batch) * self.gamma * target_q

        eval_qs = self.policy.Qeval(obs_batch, conact_batch)
        eval_q = eval_qs.gather(1, disact_batch.view(-1, 1)).squeeze()
        q_loss = F.mse_loss(eval_q, target_q)

        self.optimizer[1].zero_grad()
        q_loss.backward()
        self.optimizer[1].step()

        # optimize actor network
        policy_q = self.policy.Qpolicy(obs_batch)
        p_loss = - policy_q.mean()
        self.optimizer[0].zero_grad()
        p_loss.backward()
        self.optimizer[0].step()

        if self.scheduler is not None:
            self.scheduler[0].step()
            self.scheduler[1].step()

        self.policy.soft_update(self.tau)

        self.writer.add_scalar("Q_loss", q_loss.item(), self.iterations)
        self.writer.add_scalar("P_loss", q_loss.item(), self.iterations)
        self.writer.add_scalar('Qvalue', eval_q.mean().item(), self.iterations)
