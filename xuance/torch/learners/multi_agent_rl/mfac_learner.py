"""
MFAC: Mean Field Actor-Critic
Paper link:
http://proceedings.mlr.press/v80/yang18d/yang18d.pdf
Implementation: Pytorch
"""
import torch

from xuance.torch.learners import *


class MFAC_Learner(LearnerMAS):
    def __init__(self,
                 config: Namespace,
                 policy: nn.Module,
                 optimizer: Sequence[torch.optim.Optimizer],
                 scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
                 device: Optional[Union[int, str, torch.device]] = None,
                 model_dir: str = "./",
                 gamma: float = 0.99,
                 ):
        self.gamma = gamma
        self.tau = config.tau
        self.mse_loss = nn.MSELoss()
        super(MFAC_Learner, self).__init__(config, policy, optimizer, scheduler, device, model_dir)
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
        act_mean = torch.Tensor(sample['act_mean']).to(self.device)
        # act_mean_next = torch.Tensor(sample['act_mean_next']).to(self.device)
        rewards = torch.Tensor(sample['rewards']).to(self.device)
        terminals = torch.Tensor(sample['terminals']).float().reshape(-1, self.n_agents, 1).to(self.device)
        agent_mask = torch.Tensor(sample['agent_mask']).float().reshape(-1, self.n_agents, 1).to(self.device)
        batch_size = obs.shape[0]
        IDs = torch.eye(self.n_agents).unsqueeze(0).expand(batch_size, -1, -1).to(self.device)

        act_mean_n = act_mean.unsqueeze(1).repeat([1, self.n_agents, 1])

        # train critic network
        target_pi_dist_next = self.policy.target_actor(obs_next, IDs)
        target_pi_next = target_pi_dist_next.logits.softmax(dim=-1)
        actions_next = target_pi_dist_next.stochastic_sample()
        actions_next_onehot = self.onehot_action(actions_next, self.dim_act).type(torch.float)
        act_mean_next = actions_next_onehot.mean(dim=-2, keepdim=False)
        act_mean_n_next = act_mean_next.unsqueeze(1).repeat([1, self.n_agents, 1])

        q_eval = self.policy.critic(obs, act_mean_n, IDs)
        q_eval_a = q_eval.gather(-1, actions.long().reshape([batch_size, self.n_agents, 1]))

        q_eval_next = self.policy.target_critic(obs_next, act_mean_n_next, IDs)
        shape = q_eval_next.shape
        v_mf = torch.bmm(q_eval_next.reshape(-1, 1, shape[-1]), target_pi_next.reshape(-1, shape[-1], 1))
        v_mf = v_mf.reshape(*(list(shape[0:-1]) + [1]))
        q_target = rewards + (1 - terminals) * self.args.gamma * v_mf
        td_error = (q_eval_a - q_target.detach()) * agent_mask
        loss_c = (td_error ** 2).sum() / agent_mask.sum()
        self.optimizer["critic"].zero_grad()
        loss_c.backward()
        self.optimizer["critic"].step()
        if self.scheduler['critic'] is not None:
            self.scheduler['critic'].step()

        # train actor network
        _, pi_dist = self.policy(obs, IDs)
        actions_ = pi_dist.stochastic_sample()
        advantages = self.policy.target_critic(obs, act_mean_n, IDs)
        advantages = advantages.gather(-1, actions_.long().reshape([batch_size, self.n_agents, 1]))
        log_pi_prob = pi_dist.log_prob(actions_).unsqueeze(-1)
        advantages = log_pi_prob * advantages.detach()
        loss_a = -(advantages.sum() / agent_mask.sum())
        self.optimizer["actor"].zero_grad()
        loss_a.backward()
        grad_norm_actor = torch.nn.utils.clip_grad_norm_(self.policy.parameters_actor, self.args.clip_grad)
        self.optimizer["actor"].step()
        if self.scheduler['actor'] is not None:
            self.scheduler['actor'].step()

        self.policy.soft_update(self.tau)
        # Logger
        lr_a = self.optimizer['actor'].state_dict()['param_groups'][0]['lr']
        lr_c = self.optimizer['critic'].state_dict()['param_groups'][0]['lr']

        info = {
            "learning_rate_actor": lr_a,
            "learning_rate_critic": lr_c,
            "actor_loss": loss_a.item(),
            "critic_loss": loss_c.item(),
            "actor_gradient_norm": grad_norm_actor.item()
        }

        return info
