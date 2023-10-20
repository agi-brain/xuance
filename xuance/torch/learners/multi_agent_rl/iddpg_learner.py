"""
Independent Deep Deterministic Policy Gradient (IDDPG)
Implementation: Pytorch
"""
from xuance.torch.learners import *


class IDDPG_Learner(LearnerMAS):
    def __init__(self,
                 config: Namespace,
                 policy: nn.Module,
                 optimizer: Sequence[torch.optim.Optimizer],
                 scheduler: Sequence[torch.optim.lr_scheduler._LRScheduler] = None,
                 device: Optional[Union[int, str, torch.device]] = None,
                 model_dir: str = "./",
                 gamma: float = 0.99,
                 sync_frequency: int = 100
                 ):
        self.gamma = gamma
        self.tau = config.tau
        self.sync_frequency = sync_frequency
        self.mse_loss = nn.MSELoss()
        super(IDDPG_Learner, self).__init__(config, policy, optimizer, scheduler, device, model_dir)
        self.optimizer = {
            'actor': optimizer[0],
            'critic': optimizer[1]
        }
        self.scheduler = {
            'actor': scheduler[0],
            'critic': scheduler[1]
        }

    def update(self, sample):
        self.iterations += 1
        obs = torch.Tensor(sample['obs']).to(self.device)
        actions = torch.Tensor(sample['actions']).to(self.device)
        obs_next = torch.Tensor(sample['obs_next']).to(self.device)
        rewards = torch.Tensor(sample['rewards']).to(self.device)
        terminals = torch.Tensor(sample['terminals']).float().reshape(-1, self.n_agents, 1).to(self.device)
        agent_mask = torch.Tensor(sample['agent_mask']).float().reshape(-1, self.n_agents, 1).to(self.device)
        IDs = torch.eye(self.n_agents).unsqueeze(0).expand(self.args.batch_size, -1, -1).to(self.device)

        q_eval = self.policy.critic(obs, actions, IDs)
        q_next = self.policy.target_critic(obs_next, self.policy.target_actor(obs_next, IDs), IDs)
        q_target = rewards + (1-terminals) * self.args.gamma * q_next

        # calculate the loss function
        _, actions_eval = self.policy(obs, IDs)
        loss_a = -(self.policy.critic(obs, actions_eval, IDs) * agent_mask).sum() / agent_mask.sum()
        self.optimizer['actor'].zero_grad()
        loss_a.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters_actor, self.args.grad_clip_norm)
        self.optimizer['actor'].step()
        if self.scheduler['actor'] is not None:
            self.scheduler['actor'].step()

        td_error = (q_eval - q_target.detach()) * agent_mask
        loss_c = (td_error ** 2).sum() / agent_mask.sum()
        self.optimizer['critic'].zero_grad()
        loss_c.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters_critic, self.args.grad_clip_norm)
        self.optimizer['critic'].step()
        if self.scheduler['critic'] is not None:
            self.scheduler['critic'].step()

        self.policy.soft_update(self.tau)

        lr_a = self.optimizer['actor'].state_dict()['param_groups'][0]['lr']
        lr_c = self.optimizer['critic'].state_dict()['param_groups'][0]['lr']

        info = {
            "learning_rate_actor": lr_a,
            "learning_rate_critic": lr_c,
            "loss_actor": loss_a.item(),
            "loss_critic": loss_c.item(),
            "predictQ": q_eval.mean().item()
        }

        return info
