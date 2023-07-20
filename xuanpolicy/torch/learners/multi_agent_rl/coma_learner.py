"""
COMA: Counterfactual Multi-Agent Policy Gradients
Paper link: https://ojs.aaai.org/index.php/AAAI/article/view/11794
Implementation: Pytorch
"""
from xuanpolicy.torch.learners import *


class COMA_Learner(LearnerMAS):
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
        self.td_lambda = config.td_lambda
        self.sync_frequency = sync_frequency
        self.mse_loss = nn.MSELoss()
        super(COMA_Learner, self).__init__(config, policy, optimizer, scheduler, device, model_dir)
        self.optimizer = {
            'actor': optimizer[0],
            'critic': optimizer[1]
        }
        self.scheduler = {
            'actor': scheduler[0],
            'critic': scheduler[1]
        }
        self.iterations_actor = self.iterations
        self.iterations_critic = 0

    def build_td_lambda(self, rewards, terminated, agent_mask, target_q_a, max_step_len):
        returns = target_q_a.new_zeros(*target_q_a.shape)
        returns[:, -1] = target_q_a[:, -1] * (1 - terminated.sum(dim=1))
        for t in range(max_step_len - 2, -1, -1):
            returns[:, t] = self.td_lambda * self.gamma * returns[:, t + 1] + (rewards[:, t] + (1 - self.td_lambda) * self.gamma * target_q_a[:, t + 1] * (1 - terminated[:, t])) * agent_mask[:, t]
        return returns[:, 0:-1]

    def update(self, sample):
        self.iterations += 1
        state = torch.Tensor(sample['state']).to(self.device)
        state_repeat = state.unsqueeze(-2).repeat(1, 1, self.n_agents, 1).to(self.device)
        obs = torch.Tensor(sample['obs']).to(self.device)
        actions = torch.Tensor(sample['actions']).to(self.device)
        actions_onehot = torch.Tensor(sample['actions_onehot']).to(self.device)
        rewards = torch.Tensor(sample['rewards'][:, :-1]).mean(dim=-2).to(self.device)
        terminals = torch.Tensor(sample['terminals']).float().to(self.device)
        agent_mask = torch.Tensor(sample['agent_mask']).float().to(self.device)
        batch_size, step_len = obs.shape[0], obs.shape[1]
        IDs = torch.eye(self.n_agents).unsqueeze(0).unsqueeze(0).expand(batch_size, step_len, -1, -1).to(self.device)

        # train critic network
        target_critic_in = self.policy.build_critic_in(state_repeat, obs, actions_onehot, IDs)
        target_q_eval = self.policy.target_critic(target_critic_in)
        target_q_a = target_q_eval.gather(-1, actions.unsqueeze(-1).long()).view(batch_size, step_len, self.n_agents)
        targets = self.build_td_lambda(rewards, terminals, agent_mask, target_q_a, step_len)

        loss_c_item = 0.0
        q_eval = torch.zeros_like(target_q_eval)[:, :-1]
        for t in reversed(range(step_len - 1)):
            agent_mask_t = agent_mask[:, t:t + 1]
            actions_t = actions[:, t].unsqueeze(-2)
            critic_in = self.policy.build_critic_in(state_repeat, obs, actions_onehot, IDs, t)
            q_eval_t = self.policy.critic(critic_in)
            q_eval[:, t:t + 1] = q_eval_t
            q_eval_a_t = q_eval_t.gather(-1, actions_t.unsqueeze(-1).long()).view(batch_size, 1, self.n_agents)
            q_eval_a_t *= agent_mask_t
            target_t = targets[:, t:t + 1]

            self.iterations_critic += 1
            loss_c = self.mse_loss(q_eval_a_t, target_t.detach())
            self.optimizer['critic'].zero_grad()
            loss_c.backward()
            grad_norm_critic = torch.nn.utils.clip_grad_norm_(self.policy.parameters_critic, self.args.clip_grad)
            self.optimizer['critic'].step()
            if self.iterations_critic % self.sync_frequency == 0:
                self.policy.copy_target()
            loss_c_item += loss_c.item()
        loss_c_item /= (step_len - 1)

        if self.scheduler['critic'] is not None:
            self.scheduler['critic'].step()

        # calculate baselines
        _, pi_dist = self.policy(obs, IDs)
        pi = pi_dist.logits.softmax(dim=-1)[:, :-1]
        pi_log_prob = pi_dist.log_prob(actions)[:, :-1]
        baseline = (pi * q_eval).sum(-1).detach()

        q_eval_a = q_eval.gather(-1, actions[:, :-1].unsqueeze(-1).long()).view(batch_size, step_len - 1, self.n_agents)
        advantages = (q_eval_a - baseline).detach()

        self.iterations_actor += 1
        loss_coma = -((advantages * pi_log_prob) * agent_mask[:, :-1]).sum() / agent_mask[:, :-1].sum()
        self.optimizer['actor'].zero_grad()
        loss_coma.backward()
        grad_norm_actor = torch.nn.utils.clip_grad_norm_(self.policy.parameters_actor, self.args.clip_grad)
        self.optimizer['actor'].step()

        if self.scheduler['actor'] is not None:
            self.scheduler['actor'].step()

        # Logger
        lr_a = self.optimizer['actor'].state_dict()['param_groups'][0]['lr']
        lr_c = self.optimizer['critic'].state_dict()['param_groups'][0]['lr']

        info = {
            "learning_rate_actor": lr_a,
            "learning_rate_critic": lr_c,
            "actor_loss": loss_coma.item(),
            "critic_loss": loss_c_item,
            "advantage": advantages.mean().item(),
            "actor_gradient_norm": grad_norm_actor.item(),
            "critic_gradient_norm": grad_norm_critic.item()
        }

        return info
