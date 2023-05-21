"""
Multi-Agent Proximal Policy Optimization (MAPPO)
Paper link:
https://arxiv.org/pdf/2103.01955.pdf
Implementation: Pytorch
"""
import torch

from xuanpolicy.xuanpolicy_torch.learners import *
from xuanpolicy.xuanpolicy_torch.utils.operations import merge_distributions


class MAPPO_KL_Learner(LearnerMAS):
    def __init__(self,
                 config: Namespace,
                 policy: nn.Module,
                 optimizer: torch.optim.Optimizer,
                 scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
                 summary_writer: Optional[SummaryWriter] = None,
                 device: Optional[Union[int, str, torch.device]] = None,
                 modeldir: str = "./",
                 gamma: float = 0.99,
                 ):
        self.gamma = gamma
        self.clip_range = config.clip_range
        self.value_clip_range = config.value_clip_range
        self.mse_loss = nn.MSELoss()
        super(MAPPO_KL_Learner, self).__init__(config, policy, optimizer, scheduler, summary_writer, device, modeldir)
        self.target_kl = config.target_kl
        self.kl_coef = 1.0

    def update(self, sample):
        self.iterations += 1
        state = torch.Tensor(sample['state']).to(self.device)
        obs = torch.Tensor(sample['obs']).to(self.device)
        actions = torch.Tensor(sample['actions']).to(self.device)
        pi_dists_old = sample['pi_dist_old']
        returns = torch.Tensor(sample['values']).to(self.device)
        advantages = torch.Tensor(sample['advantages']).to(self.device)
        agent_mask = torch.Tensor(sample['agent_mask']).float().view(-1, self.n_agents, 1).to(self.device)
        batch_size = obs.shape[0]
        IDs = torch.eye(self.n_agents).unsqueeze(0).expand(batch_size, -1, -1).to(self.device)

        _, pi_dist = self.policy(obs, IDs)
        log_pi = pi_dist.log_prob(actions)
        old_dist = merge_distributions(pi_dists_old)
        kl = pi_dist.kl_divergence(old_dist).mean()
        log_pi_old = old_dist.log_prob(actions)
        ratio = torch.exp(log_pi - log_pi_old).view(batch_size, self.n_agents, 1)
        advantages_mask = advantages.detach() * agent_mask
        loss_a = -(ratio * advantages_mask).mean() + self.kl_coef * kl

        entropy = pi_dist.entropy().reshape(agent_mask.shape) * agent_mask
        loss_e = entropy.mean()

        state_expand = state.unsqueeze(-2).repeat(1, self.n_agents, 1)
        value = self.policy.values(state_expand, IDs) * agent_mask
        if self.args.use_value_clip:
            value_clipped = returns + (value - returns).clamp(-self.value_clip_range, self.value_clip_range)
            value_target = advantages_mask + returns * agent_mask
            loss_v = (value - value_target) ** 2
            loss_v_clipped = (value_clipped * agent_mask - value_target) ** 2
            loss_c = torch.max(loss_v, loss_v_clipped).mean()
        else:
            loss_c = self.mse_loss(value, returns.detach() * agent_mask)

        loss = loss_a + self.args.vf_coef * loss_c - self.args.ent_coef * loss_e
        self.optimizer.zero_grad()
        loss.backward()
        if self.args.use_grad_norm:
            grad_norm = torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.args.clip_grad)
            self.writer.add_scalar("gradient_norm", grad_norm.item(), self.iterations)
        self.optimizer.step()
        if self.scheduler is not None:
            self.scheduler.step()

        if kl > self.target_kl * 1.5:
            self.kl_coef = self.kl_coef * 2.
        elif kl < self.target_kl * 0.5:
            self.kl_coef = self.kl_coef / 2.
        self.kl_coef = np.clip(self.kl_coef, 0.1, 20)

        # Logger
        lr = self.optimizer.state_dict()['param_groups'][0]['lr']
        self.writer.add_scalar("learning_rate", lr, self.iterations)
        self.writer.add_scalar("actor_loss", loss_a.item(), self.iterations)
        self.writer.add_scalar("critic_loss", loss_c.item(), self.iterations)
        self.writer.add_scalar("entropy", loss_e.item(), self.iterations)
        self.writer.add_scalar("loss", loss.item(), self.iterations)
        self.writer.add_scalar("predict_value", value.mean().item(), self.iterations)
