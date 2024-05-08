"""
Independent Soft Actor-critic (ISAC)
Implementation: Pytorch
"""
from xuance.torch.learners import *


class ISAC_Learner(LearnerMAS):
    def __init__(self,
                 config: Namespace,
                 policy: nn.Module,
                 optimizer: Sequence[torch.optim.Optimizer],
                 scheduler: Sequence[torch.optim.lr_scheduler._LRScheduler] = None,
                 device: Optional[Union[int, str, torch.device]] = None,
                 model_dir: str = "./",
                 **kwargs):
        self.gamma = config.gamma
        self.tau = config.tau
        self.alpha = config.alpha
        self.mse_loss = nn.MSELoss()
        self.use_automatic_entropy_tuning = kwargs['use_automatic_entropy_tuning']
        super(ISAC_Learner, self).__init__(config, policy, optimizer, scheduler, device, model_dir)
        self.optimizer = {
            'actor': optimizer[0],
            'critic': optimizer[1]
        }
        self.scheduler = {
            'actor': scheduler[0],
            'critic': scheduler[1]
        }
        if self.use_automatic_entropy_tuning:
            self.target_entropy = kwargs['target_entropy']
            self.log_alpha = nn.Parameter(torch.zeros(1, requires_grad=True, device=device))
            self.alpha = self.log_alpha.exp()
            self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=kwargs['lr_policy'])

    def update(self, sample):
        self.iterations += 1
        obs = torch.Tensor(sample['obs']).to(self.device)
        actions = torch.Tensor(sample['actions']).to(self.device)
        obs_next = torch.Tensor(sample['obs_next']).to(self.device)
        rewards = torch.Tensor(sample['rewards']).to(self.device)
        terminals = torch.Tensor(sample['terminals']).float().reshape(-1, self.n_agents, 1).to(self.device)
        agent_mask = torch.Tensor(sample['agent_mask']).float().reshape(-1, self.n_agents, 1).to(self.device)
        IDs = torch.eye(self.n_agents).unsqueeze(0).expand(self.args.batch_size, -1, -1).to(self.device)

        # actor update
        log_pi, policy_q_1, policy_q_2 = self.policy.Qpolicy(obs, IDs)
        policy_q = torch.min(policy_q_1, policy_q_2)
        log_pi = log_pi.reshape([-1, self.n_agents, 1])
        loss_a = ((self.alpha * log_pi - policy_q) * agent_mask).sum() / agent_mask.sum()
        self.optimizer['actor'].zero_grad()
        loss_a.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters_actor, self.args.grad_clip_norm)
        self.optimizer['actor'].step()
        if self.scheduler['actor'] is not None:
            self.scheduler['actor'].step()

        # critic update
        action_q_1, action_q_2 = self.policy.Qaction(obs, actions, IDs)
        log_pi_next, target_q = self.policy.Qtarget(obs_next, IDs)
        log_pi_next = log_pi_next.reshape([-1, self.n_agents, 1])
        target_value = target_q - self.alpha * log_pi_next
        backup = rewards + (1 - terminals) * self.gamma * target_value
        td_error_1, td_error_2 = action_q_1 - backup.detach(), action_q_2 - backup.detach()
        td_error_1 *= agent_mask
        td_error_2 *= agent_mask
        loss_c = ((td_error_1 ** 2).sum() + (td_error_2 ** 2).sum()) / agent_mask.sum()
        self.optimizer['critic'].zero_grad()
        loss_c.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters_critic, self.args.grad_clip_norm)
        self.optimizer['critic'].step()
        if self.scheduler['critic'] is not None:
            self.scheduler['critic'].step()

        # automatic entropy tuning
        if self.use_automatic_entropy_tuning:
            alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            self.alpha = self.log_alpha.exp()
        else:
            alpha_loss = 0

        self.policy.soft_update(self.tau)

        lr_a = self.optimizer['actor'].state_dict()['param_groups'][0]['lr']
        lr_c = self.optimizer['critic'].state_dict()['param_groups'][0]['lr']

        info = {
            "learning_rate_actor": lr_a,
            "learning_rate_critic": lr_c,
            "loss_actor": loss_a.item(),
            "loss_critic": loss_c.item(),
            "predictQ": policy_q.mean().item(),
            "alpha_loss": alpha_loss.item(),
            "alpha": self.alpha.item(),
        }

        return info
