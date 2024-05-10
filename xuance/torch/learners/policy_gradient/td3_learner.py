# TD3 add three tricks to DDPG:
# 1. noisy action in target actor
# 2. double critic network
# 3. delayed actor update
from xuance.torch.learners import *


class TD3_Learner(Learner):
    def __init__(self,
                 policy: nn.Module,
                 optimizers: Sequence[torch.optim.Optimizer],
                 schedulers: Sequence[torch.optim.lr_scheduler._LRScheduler],
                 device: Optional[Union[int, str, torch.device]] = None,
                 model_dir: str = "./",
                 gamma: float = 0.99,
                 tau: float = 0.01,
                 delay: int = 3):
        self.tau = tau
        self.gamma = gamma
        self.delay = delay
        super(TD3_Learner, self).__init__(policy, optimizers, schedulers, device, model_dir)

    def update(self, obs_batch, act_batch, rew_batch, next_batch, terminal_batch):
        self.iterations += 1
        act_batch = torch.as_tensor(act_batch, device=self.device)
        rew_batch = torch.as_tensor(rew_batch, device=self.device)
        ter_batch = torch.as_tensor(terminal_batch, device=self.device)

        # critic update
        action_q_A, action_q_B = self.policy.Qaction(obs_batch, act_batch)
        action_q_A = action_q_A.reshape([-1])
        action_q_B = action_q_B.reshape([-1])
        next_q = self.policy.Qtarget(next_batch).reshape([-1])
        target_q = rew_batch + self.gamma * (1 - ter_batch) * next_q
        q_loss = F.mse_loss(action_q_A, target_q.detach()) + F.mse_loss(action_q_B, target_q.detach())
        self.optimizer[1].zero_grad()
        q_loss.backward()
        self.optimizer[1].step()
        if self.scheduler is not None:
            self.scheduler[1].step()

        # actor update
        if self.iterations % self.delay == 0:
            policy_q = self.policy.Qpolicy(obs_batch)
            p_loss = -policy_q.mean()
            self.optimizer[0].zero_grad()
            p_loss.backward()
            self.optimizer[0].step()
            if self.scheduler is not None:
                self.scheduler[0].step()
            self.policy.soft_update(self.tau)

        actor_lr = self.optimizer[0].state_dict()['param_groups'][0]['lr']
        critic_lr = self.optimizer[1].state_dict()['param_groups'][0]['lr']

        info = {
            "Qloss": q_loss.item(),
            "QvalueA": action_q_A.mean().item(),
            "QvalueB": action_q_B.mean().item(),
            "actor_lr": actor_lr,
            "critic_lr": critic_lr
        }
        if self.iterations % self.delay == 0:
            info["Ploss"] = p_loss.item()

        return info
