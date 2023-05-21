"""
Weighted QMIX
Paper link:
https://proceedings.neurips.cc/paper/2020/file/73a427badebe0e32caa2e1fc7530b7f3-Paper.pdf
Implementation: Pytorch
"""
from xuanpolicy.xuanpolicy_torch.learners import *


class WQMIX_Learner(LearnerMAS):
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
        self.alpha = config.alpha
        self.gamma = gamma
        self.sync_frequency = sync_frequency
        self.mse_loss = nn.MSELoss()
        super(WQMIX_Learner, self).__init__(config, policy, optimizer, scheduler, summary_writer, device, modeldir)

    def update(self, sample):
        self.iterations += 1
        state = torch.Tensor(sample['state']).to(self.device)
        obs = torch.Tensor(sample['obs']).to(self.device)
        actions = torch.Tensor(sample['actions']).to(self.device)
        state_next = torch.Tensor(sample['state_next']).to(self.device)
        obs_next = torch.Tensor(sample['obs_next']).to(self.device)
        rewards = torch.Tensor(sample['rewards']).mean(dim=1).to(self.device)
        terminals = torch.Tensor(sample['terminals']).float().view(-1, self.n_agents, 1).to(self.device)
        agent_mask = torch.Tensor(sample['agent_mask']).float().view(-1, self.n_agents, 1).to(self.device)
        IDs = torch.eye(self.n_agents).unsqueeze(0).expand(self.args.batch_size, -1, -1).to(self.device)

        # calculate Q_tot
        _, action_max, q_eval = self.policy(obs, IDs)
        action_max = action_max.unsqueeze(-1)
        q_eval_a = q_eval.gather(-1, actions.long().view([self.args.batch_size, self.n_agents, 1]))
        q_tot_eval = self.policy.Q_tot(q_eval_a * agent_mask, state)

        # calculate centralized Q
        q_eval_centralized = self.policy.q_centralized(obs, IDs).gather(-1, action_max.long())
        q_tot_centralized = self.policy.q_feedforward(q_eval_centralized*agent_mask, state)

        # calculate y_i
        if self.args.double_q:
            _, action_next_greedy, _ = self.policy(obs_next, IDs)
            action_next_greedy = action_next_greedy.unsqueeze(-1)
        else:
            q_next_eval = self.policy.target_Q(obs_next, IDs)
            action_next_greedy = q_next_eval.argmax(dim=-1, keepdim=True)
        q_eval_next_centralized = self.policy.target_q_centralized(obs_next, IDs).gather(-1, action_next_greedy)
        q_tot_next_centralized = self.policy.target_q_feedforward(q_eval_next_centralized*agent_mask, state_next)

        if self.args.consider_terminal_states:
            target_value = rewards + (1 - terminals) * self.args.gamma * q_tot_next_centralized
        else:
            target_value = rewards + self.args.gamma * q_tot_next_centralized
        td_error = q_tot_eval - target_value.detach()

        # calculate weights
        ones = torch.ones_like(td_error)
        w = ones * self.alpha
        if self.args.agent == "CWQMIX":
            condition_1 = ((action_max == actions.view([-1, self.n_agents, 1])) * agent_mask).all(dim=1)
            condition_2 = target_value > q_tot_centralized
            conditions = condition_1 | condition_2
            w = torch.where(conditions, ones, w)
        elif self.args.agent == "OWQMIX":
            condition = td_error < 0
            w = torch.where(condition, ones, w)
        else:
            AttributeError("You have assigned an unexpected WQMIX learner!")

        # calculate losses and train
        loss_central = self.mse_loss(q_tot_centralized, target_value.detach())
        loss_qmix = (w.detach() * (td_error ** 2)).mean()
        loss = loss_qmix + loss_central
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        if self.scheduler is not None:
            self.scheduler.step()

        if self.iterations % self.sync_frequency == 0:
            self.policy.copy_target()
        lr = self.optimizer.state_dict()['param_groups'][0]['lr']
        self.writer.add_scalar("learning_rate", lr, self.iterations)
        self.writer.add_scalar("loss_Qmix", loss_qmix.item(), self.iterations)
        self.writer.add_scalar("loss_central", loss_central.item(), self.iterations)
        self.writer.add_scalar("loss", loss.item(), self.iterations)
        self.writer.add_scalar("predictQ", q_tot_eval.mean().item(), self.iterations)
