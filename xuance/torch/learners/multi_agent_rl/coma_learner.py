"""
COMA: Counterfactual Multi-Agent Policy Gradients
Paper link: https://ojs.aaai.org/index.php/AAAI/article/view/11794
Implementation: Pytorch
"""
import torch
from torch import nn
from torch.nn.functional import one_hot
from xuance.common import List
from argparse import Namespace
from xuance.torch.learners.multi_agent_rl.iac_learner import IAC_Learner


class COMA_Learner(IAC_Learner):
    def __init__(self,
                 config: Namespace,
                 model_keys: List[str],
                 agent_keys: List[str],
                 policy: nn.Module):
        super(COMA_Learner, self).__init__(config, model_keys, agent_keys, policy)
        self.sync_frequency = config.sync_frequency
        self.n_actions = {k: self.policy.action_space[k].n for k in self.model_keys}
        self.use_global_state = config.use_global_state
        self.mse_loss = nn.MSELoss()
        self.egreedy = 0.0

    def build_optimizer(self):
        self.optimizer = {
            'actor': torch.optim.Adam(self.policy.parameters_actor, self.config.learning_rate_actor, eps=1e-5),
            'critic': torch.optim.Adam(self.policy.parameters_critic, self.config.learning_rate_critic, eps=1e-5)
        }
        self.scheduler = {
            'actor': torch.optim.lr_scheduler.LinearLR(self.optimizer['actor'],
                                                       start_factor=1.0,
                                                       end_factor=self.end_factor_lr_decay,
                                                       total_iters=self.config.running_steps),
            'critic': torch.optim.lr_scheduler.LinearLR(self.optimizer['critic'],
                                                        start_factor=1.0,
                                                        end_factor=self.end_factor_lr_decay,
                                                        total_iters=self.config.running_steps)
        }

    def update(self, sample, epsilon=0.0):
        self.iterations += 1
        info = {}

        # prepare training data
        sample_Tensor = self.build_training_data(sample=sample,
                                                 use_parameter_sharing=self.use_parameter_sharing,
                                                 use_actions_mask=self.use_actions_mask,
                                                 use_global_state=True)
        batch_size = sample_Tensor['batch_size']
        state = sample_Tensor['state']
        obs = sample_Tensor['obs']
        actions = sample_Tensor['actions']
        agent_mask = sample_Tensor['agent_mask']
        avail_actions = sample_Tensor['avail_actions']
        returns = sample_Tensor['returns']
        IDs = sample_Tensor['agent_ids']

        bs = batch_size * self.n_agents if self.use_parameter_sharing else batch_size

        # feedforward
        _, pi_dist_dict = self.policy(observation=obs, agent_ids=IDs, avail_actions=avail_actions, epsilon=self.egreedy)
        if self.use_parameter_sharing:
            key = self.model_keys[0]
            actions_onehot = {key: one_hot(actions[key].long(), self.n_actions[key])}
            _, values_pred = self.policy.get_values(state=state, observation=obs, actions=actions_onehot,
                                                    agent_ids=IDs, target=False)
            values_pred = values_pred.reshape(bs, -1)
        else:
            pass

        values_pred_dict = {k: values_pred for k in self.model_keys}

        # calculate loss
        loss_a, loss_c = [], []
        for key in self.model_keys:
            mask_values = agent_mask[key]

            pi_probs = pi_dist_dict[key].probs
            if self.use_actions_mask:
                pi_probs[avail_actions[key] == 0] = 0
            baseline = (pi_probs * values_pred_dict[key]).sum(-1).reshape(bs)
            pi_taken = pi_probs.gather(-1, actions[key].unsqueeze(-1).long())
            q_taken = values_pred_dict[key].gather(-1, actions[key].unsqueeze(-1).long()).reshape(bs)
            log_pi_taken = torch.log(pi_taken).reshape(bs)
            advantages = (q_taken - baseline).detach()
            loss_a.append(-(advantages * log_pi_taken * mask_values).sum() / mask_values.sum())

            td_error = (q_taken - returns[key]) * mask_values
            loss_c.append((td_error ** 2).sum() / mask_values.sum())

        # update critic
        loss_critic = sum(loss_c)
        self.optimizer['critic'].zero_grad()
        loss_critic.backward()
        if self.use_grad_clip:
            grad_norm = torch.nn.utils.clip_grad_norm_(self.policy.parameters_critic, self.grad_clip_norm)
            info["gradient_norm_actor"] = grad_norm.item()
        self.optimizer['critic'].step()
        if self.scheduler['critic'] is not None:
            self.scheduler['critic'].step()
        if self.iterations % self.sync_frequency == 0:
            self.policy.copy_target()

        # update actor(s)
        loss_coma = sum(loss_a)
        self.optimizer['actor'].zero_grad()
        loss_coma.backward()
        if self.use_grad_clip:
            grad_norm = torch.nn.utils.clip_grad_norm_(self.policy.parameters_actor, self.grad_clip_norm)
            info["gradient_norm_actor"] = grad_norm.item()
        self.optimizer['actor'].step()
        if self.scheduler['actor'] is not None:
            self.scheduler['actor'].step()

        # Logger
        learning_rate_actor = self.optimizer['actor'].state_dict()['param_groups'][0]['lr']
        learning_rate_critic = self.optimizer['critic'].state_dict()['param_groups'][0]['lr']

        info = {
            "learning_rate_actor": learning_rate_actor,
            "learning_rate_critic": learning_rate_critic,
            "actor_loss": loss_coma.item(),
            "critic_loss": loss_critic.item(),
            "advantage": advantages.mean().item(),
        }

        return info

    def update_recurrent(self, sample, epsilon=0.0):
        self.iterations += 1
        state = torch.Tensor(sample['state']).to(self.device)
        obs = torch.Tensor(sample['obs']).to(self.device)
        actions = torch.Tensor(sample['actions']).to(self.device)
        actions_onehot = torch.Tensor(sample['actions_onehot']).to(self.device)
        targets = torch.Tensor(sample['returns']).squeeze(-1).to(self.device)
        avail_actions = torch.Tensor(sample['avail_actions']).float().to(self.device)
        filled = torch.Tensor(sample['filled']).float().to(self.device)
        batch_size = obs.shape[0]
        episode_length = actions.shape[2]
        IDs = torch.eye(self.n_agents).unsqueeze(1).unsqueeze(0).expand(batch_size, -1, episode_length + 1, -1).to(
            self.device)

        # build critic input
        actions_in = actions_onehot.transpose(1, 2).reshape(batch_size, episode_length, -1)
        actions_in = actions_in.unsqueeze(1).repeat(1, self.n_agents, 1, 1)
        actions_in_mask = 1 - torch.eye(self.n_agents, device=self.device)
        actions_in_mask = actions_in_mask.view(-1, 1).repeat(1, self.dim_act).view(self.n_agents, -1)
        actions_in_mask = actions_in_mask.unsqueeze(1).repeat(1, episode_length, 1)
        actions_in = actions_in * actions_in_mask
        if self.use_global_state:
            state = state[:, :-1].unsqueeze(1).repeat(1, self.n_agents, 1, 1)
            critic_in = torch.concat([state, obs[:, :, :-1], actions_in], dim=-1)
        else:
            critic_in = torch.concat([obs[:, :, :-1], actions_in], dim=-1)

        # get critic value
        _, q_eval = self.policy.get_values(critic_in)
        q_eval_a = q_eval.gather(-1, actions.unsqueeze(-1).long()).squeeze(-1)
        filled_n = filled.unsqueeze(1).expand(-1, self.n_agents, -1, -1).squeeze(-1)
        td_errors = q_eval_a - targets.detach()
        td_errors *= filled_n
        loss_c = (td_errors ** 2).sum() / filled_n.sum()
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
        rnn_hidden_actor = self.policy.representation.init_hidden(batch_size * self.n_agents)
        _, pi_probs = self.policy(obs[:, :, :-1].reshape(-1, episode_length, self.dim_obs),
                                  IDs[:, :, :-1].reshape(-1, episode_length, self.n_agents),
                                  *rnn_hidden_actor,
                                  avail_actions=avail_actions[:, :, :-1].reshape(-1, episode_length, self.dim_act),
                                  epsilon=epsilon)
        pi_probs = pi_probs.reshape(batch_size, self.n_agents, episode_length, self.dim_act)
        baseline = (pi_probs * q_eval).sum(-1)

        pi_a = pi_probs.gather(-1, actions.unsqueeze(-1).long()).squeeze(-1)
        log_pi_a = torch.log(pi_a)
        advantages = (q_eval_a - baseline).detach()
        loss_coma = -(advantages * log_pi_a * filled_n).sum() / filled_n.sum()

        self.optimizer['actor'].zero_grad()
        loss_coma.backward()
        grad_norm_actor = torch.nn.utils.clip_grad_norm_(self.policy.parameters_actor, self.args.clip_grad)
        self.optimizer['actor'].step()

        if self.scheduler['actor'] is not None:
            self.scheduler['actor'].step()

        # Logger
        learning_rate_actor = self.optimizer['actor'].state_dict()['param_groups'][0]['lr']
        learning_rate_critic = self.optimizer['critic'].state_dict()['param_groups'][0]['lr']

        info = {
            "learning_rate_actor": learning_rate_actor,
            "learning_rate_critic": learning_rate_critic,
            "actor_loss": loss_coma.item(),
            "critic_loss": loss_c.item(),
            "advantage": advantages.mean().item(),
            "actor_gradient_norm": grad_norm_actor.item(),
            "critic_gradient_norm": grad_norm_critic.item()
        }

        return info
