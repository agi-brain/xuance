"""
Value Decomposition Actor-Critic (VDAC)
Paper link: https://ojs.aaai.org/index.php/AAAI/article/view/17353
Implementation: Pytorch
"""
from xuanpolicy.torch.learners import *


class VDAC_Learner(LearnerMAS):
    def __init__(self,
                 config: Namespace,
                 policy: nn.Module,
                 optimizer: torch.optim.Optimizer,
                 scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
                 device: Optional[Union[int, str, torch.device]] = None,
                 model_dir: str = "./",
                 gamma: float = 0.99,
                 ):
        self.gamma = gamma
        self.mse_loss = nn.MSELoss()
        super(VDAC_Learner, self).__init__(config, policy, optimizer, scheduler, device, model_dir)

    def update(self, sample):
        self.iterations += 1
        state = torch.Tensor(sample['state']).to(self.device)
        obs = torch.Tensor(sample['obs']).to(self.device)
        actions = torch.Tensor(sample['actions']).to(self.device)
        returns = torch.Tensor(sample['values']).mean(dim=1).to(self.device)
        rewards = torch.Tensor(sample['rewards']).mean(dim=1).to(self.device)
        advantages = torch.Tensor(sample['advantages']).to(self.device)
        agent_mask = torch.Tensor(sample['agent_mask']).float().view(-1, self.n_agents, 1).to(self.device)
        batch_size = obs.shape[0]
        IDs = torch.eye(self.n_agents).unsqueeze(0).expand(batch_size, -1, -1).to(self.device)

        _, a_dist, v_pred = self.policy(obs, IDs)
        v_pred_tot = self.policy.value_tot(v_pred*agent_mask, state)
        v_true = rewards + self.gamma * returns
        log_prob = a_dist.log_prob(actions).reshape(advantages.shape)
        entropy = a_dist.entropy().reshape(agent_mask.shape) * agent_mask

        loss_a = -(advantages * log_prob * agent_mask).mean()
        loss_c = F.mse_loss(v_pred_tot, v_true)
        loss_e = entropy.mean()

        loss = loss_a + self.args.vf_coef * loss_c - self.args.ent_coef * loss_e
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.args.clip_grad)
        self.optimizer.step()
        if self.scheduler is not None:
            self.scheduler.step()

        # Logger
        lr = self.optimizer.state_dict()['param_groups'][0]['lr']

        info = {
            "learning_rate": lr,
            "actor_loss": loss_a.item(),
            "critic_loss": loss_c.item(),
            "entropy": loss_e.item(),
            "loss": loss.item(),
            "predict_value": v_pred.mean().item()
        }

        return info
