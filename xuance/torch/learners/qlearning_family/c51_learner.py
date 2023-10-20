from xuance.torch.learners import *


class C51_Learner(Learner):
    def __init__(self,
                 policy: nn.Module,
                 optimizer: torch.optim.Optimizer,
                 scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
                 device: Optional[Union[int, str, torch.device]] = None,
                 model_dir: str = "./",
                 gamma: float = 0.99,
                 sync_frequency: int = 100):
        self.gamma = gamma
        self.sync_frequency = sync_frequency
        super(C51_Learner, self).__init__(policy, optimizer, scheduler, device, model_dir)

    def update(self, obs_batch, act_batch, rew_batch, next_batch, terminal_batch):
        self.iterations += 1
        act_batch = torch.as_tensor(act_batch, device=self.device).long()
        rew_batch = torch.as_tensor(rew_batch, device=self.device)
        ter_batch = torch.as_tensor(terminal_batch, device=self.device)
        _, _, evalZ = self.policy(obs_batch)
        _, targetA, targetZ = self.policy.target(next_batch)

        current_dist = (evalZ * F.one_hot(act_batch, evalZ.shape[1]).unsqueeze(-1)).sum(1)
        target_dist = (targetZ * F.one_hot(targetA.detach(), evalZ.shape[1]).unsqueeze(-1)).sum(1).detach()

        current_supports = self.policy.supports
        next_supports = rew_batch.unsqueeze(1) + self.gamma * self.policy.supports * (1 - ter_batch.unsqueeze(1))
        next_supports = next_supports.clamp(self.policy.vmin, self.policy.vmax)

        projection = 1 - (next_supports.unsqueeze(-1) - current_supports.unsqueeze(0)).abs() / self.policy.deltaz
        target_dist = torch.bmm(target_dist.unsqueeze(1), projection.clamp(0, 1)).squeeze(1)
        loss = -(target_dist * torch.log(current_dist + 1e-8)).sum(1).mean()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        if self.scheduler is not None:
            self.scheduler.step()
        # hard update for target network
        if self.iterations % self.sync_frequency == 0:
            self.policy.copy_target()
        lr = self.optimizer.state_dict()['param_groups'][0]['lr']

        info = {
            "Qloss": loss.item(),
            "learning_rate": lr
        }

        return info
