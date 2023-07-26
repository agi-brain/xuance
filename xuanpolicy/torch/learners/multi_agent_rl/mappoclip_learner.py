"""
Multi-Agent Proximal Policy Optimization (MAPPO)
Paper link:
https://arxiv.org/pdf/2103.01955.pdf
Implementation: Pytorch
"""
from xuanpolicy.torch.learners import *


class MAPPO_Clip_Learner(LearnerMAS):
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
        self.clip_range = config.clip_range
        self.use_grad_norm, self.clip_grad = config.use_grad_norm, config.clip_grad
        self.use_value_clip, self.value_clip_range = config.use_value_clip, config.value_clip_range
        self.vf_coef, self.ent_coef = config.vf_coef, config.ent_coef
        self.mse_loss = nn.MSELoss()
        super(MAPPO_Clip_Learner, self).__init__(config, policy, optimizer, scheduler, device, model_dir)

    def update(self, sample):
        info = {}
        self.iterations += 1
        state = torch.Tensor(sample['state']).to(self.device)
        obs = torch.Tensor(sample['obs']).to(self.device)
        actions = torch.Tensor(sample['actions']).to(self.device)
        returns = torch.Tensor(sample['returns']).to(self.device)
        advantages = torch.Tensor(sample['advantages']).to(self.device)
        log_pi_old = torch.Tensor(sample['log_pi_old']).to(self.device)
        agent_mask = torch.Tensor(sample['agent_mask']).float().view(-1, self.n_agents, 1).to(self.device)
        batch_size = obs.shape[0]
        IDs = torch.eye(self.n_agents).unsqueeze(0).expand(batch_size, -1, -1).to(self.device)

        # actor loss
        _, pi_dist = self.policy(obs, IDs)
        log_pi = pi_dist.log_prob(actions)
        ratio = torch.exp(log_pi - log_pi_old).view(batch_size, self.n_agents, 1)
        advantages_mask = advantages.detach() * agent_mask
        surrogate1 = ratio * advantages_mask
        surrogate2 = torch.clip(ratio, 1 - self.clip_range, 1 + self.clip_range) * advantages_mask
        loss_a = -torch.sum(torch.min(surrogate1, surrogate2), dim=-2, keepdim=True).mean()

        # entropy loss
        entropy = pi_dist.entropy().reshape(agent_mask.shape) * agent_mask
        loss_e = entropy.mean()

        # critic loss
        _, value = self.policy.get_values(obs, IDs)
        value = value * agent_mask
        if self.use_value_clip:
            value_clipped = returns + (value - returns).clamp(-self.value_clip_range, self.value_clip_range)
            value_target = advantages_mask + returns * agent_mask
            loss_v = (value - value_target) ** 2
            loss_v_clipped = (value_clipped * agent_mask - value_target) ** 2
            loss_c = torch.max(loss_v, loss_v_clipped).mean()
        else:
            loss_c = self.mse_loss(value, returns.detach() * agent_mask)

        loss = loss_a + self.vf_coef * loss_c - self.ent_coef * loss_e
        self.optimizer.zero_grad()
        loss.backward()
        if self.use_grad_norm:
            grad_norm = torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.clip_grad)
            info["gradient_norm"] = grad_norm.item()
        self.optimizer.step()
        if self.scheduler is not None:
            self.scheduler.step()

        # Logger
        lr = self.optimizer.state_dict()['param_groups'][0]['lr']

        info.update({
            "learning_rate": lr,
            "actor_loss": loss_a.item(),
            "critic_loss": loss_c.item(),
            "entropy": loss_e.item(),
            "loss": loss.item(),
            "predict_value": value.mean().item()
        })

        return info

    def update_recurrent(self, sample):
        info = {}
        self.iterations += 1
        state = torch.Tensor(sample['state']).to(self.device)
        obs = torch.Tensor(sample['obs']).to(self.device)
        actions = torch.Tensor(sample['actions']).to(self.device)
        returns = torch.Tensor(sample['returns']).to(self.device)
        advantages = torch.Tensor(sample['advantages']).to(self.device)
        log_pi_old = torch.Tensor(sample['log_pi_old']).to(self.device)
        avail_actions = torch.Tensor(sample['avail_actions']).float().to(self.device)
        filled = torch.Tensor(sample['filled']).float().to(self.device)
        batch_size = obs.shape[0]
        episode_length = actions.shape[2]
        IDs = torch.eye(self.n_agents).unsqueeze(1).unsqueeze(0).expand(batch_size, -1, episode_length + 1, -1).to(
            self.device)

        # actor loss
        rnn_hidden_actor = self.policy.representation.init_hidden(batch_size * self.n_agents)
        _, pi_dist = self.policy(obs[:, :, :-1].view(-1, episode_length, self.dim_obs),
                                 IDs[:, :, :-1].view(-1, episode_length, self.n_agents),
                                 *rnn_hidden_actor,
                                 avail_actions=avail_actions[:, :, :-1].view(-1, episode_length, self.dim_act))
        log_pi = pi_dist.log_prob(actions.view(-1, episode_length)).view(batch_size, self.n_agents, episode_length)
        log_pi_old = log_pi_old.transpose(1, 2)
        ratio = torch.exp(log_pi - log_pi_old).unsqueeze(-1)
        filled_n = filled.unsqueeze(1).expand(batch_size, self.n_agents, episode_length, 1)
        advantages_mask = advantages.detach().transpose(1, 2) * filled_n
        surrogate1 = ratio * advantages_mask
        surrogate2 = torch.clip(ratio, 1 - self.clip_range, 1 + self.clip_range) * advantages_mask
        loss_a = -torch.sum(torch.min(surrogate1, surrogate2)) / filled_n.sum()

        # entropy loss
        entropy = pi_dist.entropy().reshape(batch_size, self.n_agents, episode_length, 1)
        entropy = entropy * filled_n
        loss_e = entropy.sum() / filled_n.sum()

        # critic loss
        rnn_hidden_critic = self.policy.representation_critic.init_hidden(batch_size * self.n_agents)
        _, value = self.policy.get_values(obs[:, :, :-1], IDs[:, :, :-1], *rnn_hidden_critic)
        value = value * filled_n
        returns = returns.transpose(1, 2) * filled_n
        if self.use_value_clip:
            value_clipped = returns + (value - returns).clamp(-self.value_clip_range, self.value_clip_range)
            value_target = advantages_mask + returns
            loss_v = (value - value_target) ** 2
            loss_v_clipped = (value_clipped - value_target) ** 2
            loss_c = torch.max(loss_v, loss_v_clipped)
            loss_c = loss_c.sum() / filled_n.sum()
        else:
            error_c = (value - returns.detach()) * filled_n
            loss_c = (error_c ** 2).sum() / filled_n.sum()

        loss = loss_a + self.vf_coef * loss_c - self.ent_coef * loss_e
        self.optimizer.zero_grad()
        loss.backward()
        if self.use_grad_norm:
            grad_norm = torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.clip_grad)
            info["gradient_norm"] = grad_norm.item()
        self.optimizer.step()
        if self.scheduler is not None:
            self.scheduler.step()

        # Logger
        lr = self.optimizer.state_dict()['param_groups'][0]['lr']

        info.update({
            "learning_rate": lr,
            "actor_loss": loss_a.item(),
            "critic_loss": loss_c.item(),
            "entropy": loss_e.item(),
            "loss": loss.item(),
            "predict_value": value.mean().item()
        })

        return info
