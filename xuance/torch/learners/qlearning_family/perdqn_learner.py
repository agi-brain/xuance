from xuance.torch.learners import *


class PerDQN_Learner(Learner):
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
        super(PerDQN_Learner, self).__init__(policy, optimizer, scheduler, device, model_dir)

    def update(self, obs_batch, act_batch, rew_batch, next_batch, terminal_batch):
        self.iterations += 1
        act_batch = torch.as_tensor(act_batch, device=self.device)
        rew_batch = torch.as_tensor(rew_batch, device=self.device)
        ter_batch = torch.as_tensor(terminal_batch, device=self.device)

        _, _, evalQ = self.policy(obs_batch)
        _, _, targetQ = self.policy.target(next_batch)
        targetQ = targetQ.max(dim=-1).values
        targetQ = rew_batch + self.gamma * (1 - ter_batch) * targetQ
        predictQ = (evalQ * F.one_hot(act_batch.long(), evalQ.shape[1])).sum(dim=-1)

        td_error = targetQ - predictQ
        loss = F.mse_loss(predictQ, targetQ)
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
            "learning_rate": lr,
            "predictQ": predictQ.mean().item()
        }
        
        return np.abs(td_error.cpu().detach().numpy()), info
