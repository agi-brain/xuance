"""
Multi-Agent Deep Q Network
Code link: github.com/opendilab/DI-engine/blob/main/ding/policy/madqn.py
Implementation: TensorFlow 2.X
Trick: Parameter sharing for all agents, with agents' one-hot IDs as actor-critic's inputs.
"""
from xuanpolicy.xuanpolicy_torch.learners import *


class MADQN_Learner(LearnerMAS):
    def __init__(self,
                 config: Namespace,
                 policy: nn.Module,
                 optimizer: torch.optim.Optimizer,
                 scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
                 summary_writer: Optional[SummaryWriter] = None,
                 device: Optional[Union[int, str, torch.device]] = None,
                 modeldir: str = "./",
                 gamma: float = 0.99,
                 sync_frequency: int = 100
                 ):
        self.gamma = gamma
        self.sync_frequency = sync_frequency
        self.mse_loss = nn.MSELoss()
        super(MADQN_Learner, self).__init__(config, policy, optimizer, scheduler, summary_writer, device, modeldir)

    def update(self, sample):
        self.iterations += 1
        obs = torch.Tensor(sample['obs']).to(self.device)
        actions = torch.Tensor(sample['actions']).to(self.device)
        obs_next = torch.Tensor(sample['obs_next']).to(self.device)
        rewards = torch.Tensor(sample['rewards']).to(self.device)
        terminals = torch.Tensor(sample['terminals']).float().view(-1, self.n_agents, 1).to(self.device)
        agent_mask = torch.Tensor(sample['agent_mask']).float().view(-1, self.n_agents, 1).to(self.device)
        IDs = torch.eye(self.n_agents).unsqueeze(0).expand(self.args.batch_size, -1, -1).to(self.device)

        # train actor
        _, _, eval_q = self.policy(obs, IDs)
        _, _, target_q = self.policy(obs_next, IDs)
        target_q = target_q.max(dim=-1).values
        target_q = rewards + self.gamma * (1 - terminals) * target_q
        predict_q = (eval_q * F.one_hot(actions.long(), eval_q.shape[1])).sum(dim=-1)

        loss = F.mse_loss(predict_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        if self.scheduler is not None:
            self.scheduler.step()

        # hard update for target network
        if self.iterations % self.sync_frequency == 0:
            self.policy.copy_target()
        lr = self.optimizer.state_dict()['param_groups'][0]['lr']
        self.writer.add_scalar("Qloss", loss.item(), self.iterations)
        self.writer.add_scalar("learning_rate", lr, self.iterations)
        self.writer.add_scalar("predictQ", predict_q.mean().item(), self.iterations)
