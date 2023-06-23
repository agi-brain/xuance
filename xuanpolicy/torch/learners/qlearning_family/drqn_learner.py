import torch

from xuanpolicy.torch.learners import *


class DRQN_Learner(Learner):
    def __init__(self,
                 policy: nn.Module,
                 optimizer: torch.optim.Optimizer,
                 scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
                 device: Optional[Union[int, str, torch.device]] = None,
                 modeldir: str = "./",
                 gamma: float = 0.99,
                 sync_frequency: int = 100):
        self.gamma = gamma
        self.sync_frequency = sync_frequency
        super(DRQN_Learner, self).__init__(policy, optimizer, scheduler, device, modeldir)

    def update(self, obs_batch, act_batch, rew_batch, terminal_batch, fill_batch):
        self.iterations += 1
        act_batch = torch.as_tensor(act_batch, device=self.device)
        rew_batch = torch.as_tensor(rew_batch, device=self.device)
        ter_batch = torch.as_tensor(terminal_batch, device=self.device)
        fill_batch = torch.as_tensor(fill_batch, device=self.device)
        batch_size = obs_batch.shape[0]

        self.policy.init_hidden(batch_size)
        self.policy.init_hidden_target(batch_size)
        _, _, evalQ = self.policy(obs_batch[:, 0:-1])
        _, _, targetQ = self.policy.target(obs_batch[:, 1:])
        targetQ = targetQ.max(dim=-1).values

        targetQ = rew_batch + self.gamma * (1 - ter_batch) * targetQ
        predictQ = (evalQ * F.one_hot(act_batch.long(), evalQ.shape[-1])).sum(dim=-1)

        predictQ *= fill_batch
        targetQ *= fill_batch

        loss = ((predictQ - targetQ) ** 2).sum() / fill_batch.sum()
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

        return info
