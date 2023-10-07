"""
COMA: Counterfactual Multi-Agent Policy Gradients
Paper link: https://ojs.aaai.org/index.php/AAAI/article/view/11794
Implementation: Pytorch
"""
import torch

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
        self.use_global_state = config.use_global_state
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

    def update(self, sample, epsilon=0.0):
        self.iterations += 1
        state = torch.Tensor(sample['state']).to(self.device)
        obs = torch.Tensor(sample['obs']).to(self.device)
        actions = torch.Tensor(sample['actions']).to(self.device)
        actions_onehot = torch.Tensor(sample['actions_onehot']).to(self.device)
        rewards = torch.Tensor(sample['rewards']).mean(dim=1).to(self.device)
        targets = torch.Tensor(sample['returns']).squeeze(-1).to(self.device)
        terminals = torch.Tensor(sample['terminals']).float().to(self.device)
        agent_mask = torch.Tensor(sample['agent_mask']).float().to(self.device)
        batch_size = obs.shape[0]
        IDs = torch.eye(self.n_agents).unsqueeze(0).expand(batch_size, -1, -1).to(self.device)

        # build critic input
        actions_in = actions_onehot.unsqueeze(1).view(batch_size, 1, -1).repeat(1, self.n_agents, 1)
        actions_in_mask = 1 - torch.eye(self.n_agents, device=self.device)
        actions_in_mask = actions_in_mask.view(-1, 1).repeat(1, self.dim_act).view(self.n_agents, -1)
        actions_in = actions_in * actions_in_mask.unsqueeze(0)
        if self.use_global_state:
            state = state.unsqueeze(1).repeat(1, self.n_agents, 1)
            critic_in = torch.concat([state, obs, actions_in], dim=-1)
        else:
            critic_in = torch.concat([obs, actions_in])
        # get critic value
        _, q_eval = self.policy.get_values(critic_in)
        q_eval_a = q_eval.gather(-1, actions.unsqueeze(-1).long()).squeeze(-1)
        q_eval_a *= agent_mask
        targets *= agent_mask
        loss_c = self.mse_loss(q_eval_a, targets.detach())
        self.optimizer['critic'].zero_grad()
        loss_c.backward()
        grad_norm_critic = torch.nn.utils.clip_grad_norm_(self.policy.parameters_critic, self.args.clip_grad)
        self.optimizer['critic'].step()
        if self.iterations_critic % self.sync_frequency == 0:
            self.policy.copy_target()
        self.iterations_critic += 1

        if self.scheduler['critic'] is not None:
            self.scheduler['critic'].step()

        # calculate baselines
        _, pi_probs = self.policy(obs, IDs, epsilon=epsilon)
        baseline = (pi_probs * q_eval).sum(-1).detach()

        pi_a = pi_probs.gather(-1, actions.unsqueeze(-1).long()).squeeze(-1)
        log_pi_a = torch.log(pi_a)
        advantages = (q_eval_a - baseline).detach()
        log_pi_a *= agent_mask
        advantages *= agent_mask
        loss_coma = -(advantages * log_pi_a).sum() / agent_mask.sum()

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
            "critic_loss": loss_c.item(),
            "advantage": advantages.mean().item(),
            "actor_gradient_norm": grad_norm_actor.item(),
            "critic_gradient_norm": grad_norm_critic.item()
        }

        return info
