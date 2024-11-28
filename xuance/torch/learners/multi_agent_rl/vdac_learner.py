"""
Value-Dcomposition Actor-Critic (VDAC)
Paper link:
https://ojs.aaai.org/index.php/AAAI/article/view/17353
Implementation: Pytorch
"""
import numpy as np
import torch
from torch import nn
from argparse import Namespace
from operator import itemgetter
from xuance.common import Optional, Union, List
from xuance.torch import Tensor
from xuance.torch.utils.value_norm import ValueNorm
from xuance.torch.learners import LearnerMAS


class VDAC_Learner(LearnerMAS):
    def __init__(self,
                 config: Namespace,
                 model_keys: List[str],
                 agent_keys: List[str],
                 policy: nn.Module):
        super(VDAC_Learner, self).__init__(config, model_keys, agent_keys, policy)
        self.optimizer = torch.optim.Adam(self.policy.parameters_model, lr=config.learning_rate, eps=1e-5,
                                          weight_decay=config.weight_decay)
        self.scheduler = torch.optim.lr_scheduler.LinearLR(self.optimizer,
                                                           start_factor=1.0, end_factor=config.end_factor_lr_decay,
                                                           total_iters=self.config.running_steps)
        self.lr = config.learning_rate
        self.end_factor_lr_decay = config.end_factor_lr_decay
        self.gamma = config.gamma
        self.clip_range = config.clip_range
        self.use_linear_lr_decay = config.use_linear_lr_decay
        self.use_value_clip, self.value_clip_range = config.use_value_clip, config.value_clip_range
        self.use_huber_loss, self.huber_delta = config.use_huber_loss, config.huber_delta
        self.use_value_norm = config.use_value_norm
        self.vf_coef, self.ent_coef = config.vf_coef, config.ent_coef
        self.mse_loss = nn.MSELoss()
        self.huber_loss = nn.HuberLoss(reduction="none", delta=self.huber_delta)
        if self.use_value_norm:
            self.value_normalizer = {key: ValueNorm(1).to(self.device) for key in self.model_keys}
        else:
            self.value_normalizer = None

    def build_training_data(self, sample: Optional[dict],
                            use_parameter_sharing: Optional[bool] = False,
                            use_actions_mask: Optional[bool] = False,
                            use_global_state: Optional[bool] = False):
        """
        Prepare the training data.

        Parameters:
            sample (dict): The raw sampled data.
            use_parameter_sharing (bool): Whether to use parameter sharing for individual agent models.
            use_actions_mask (bool): Whether to use actions mask for unavailable actions.
            use_global_state (bool): Whether to use global state.

        Returns:
            sample_Tensor (dict): The formatted sampled data.
        """
        batch_size = sample['batch_size']
        seq_length = sample['sequence_length'] if self.use_rnn else 1
        state, avail_actions, filled, IDs = None, None, None, None
        if use_parameter_sharing:
            k = self.model_keys[0]
            bs = batch_size * self.n_agents
            obs_tensor = Tensor(np.stack(itemgetter(*self.agent_keys)(sample['obs']), axis=1)).to(self.device)
            actions_tensor = Tensor(np.stack(itemgetter(*self.agent_keys)(sample['actions']), axis=1)).to(self.device)
            values_tensor = Tensor(np.stack(itemgetter(*self.agent_keys)(sample['values']), axis=1)).to(self.device)
            returns_tensor = Tensor(np.stack(itemgetter(*self.agent_keys)(sample['returns']), axis=1)).to(self.device)
            advantages_tensor = Tensor(np.stack(itemgetter(*self.agent_keys)(sample['advantages']), 1)).to(self.device)
            log_pi_old_tensor = Tensor(np.stack(itemgetter(*self.agent_keys)(sample['log_pi_old']), 1)).to(self.device)
            ter_tensor = Tensor(np.stack(itemgetter(*self.agent_keys)(sample['terminals']), 1)).float().to(self.device)
            msk_tensor = Tensor(np.stack(itemgetter(*self.agent_keys)(sample['agent_mask']), 1)).float().to(self.device)
            if self.use_rnn:
                obs = {k: obs_tensor.reshape(bs, seq_length, -1)}
                if len(actions_tensor.shape) == 3:
                    actions = {k: actions_tensor.reshape(bs, seq_length)}
                elif len(actions_tensor.shape) == 4:
                    actions = {k: actions_tensor.reshape(bs, seq_length, -1)}
                else:
                    raise AttributeError("Wrong actions shape.")
                values = {k: values_tensor.reshape(bs, seq_length)}
                returns = {k: returns_tensor.reshape(bs, seq_length)}
                advantages = {k: advantages_tensor.reshape(bs, seq_length)}
                log_pi_old = {k: log_pi_old_tensor.reshape(bs, seq_length)}
                terminals = {k: ter_tensor.reshape(bs, seq_length)}
                agent_mask = {k: msk_tensor.reshape(bs, seq_length)}
                IDs = torch.eye(self.n_agents).unsqueeze(1).unsqueeze(0).expand(
                    batch_size, -1, seq_length, -1).reshape(bs, seq_length, self.n_agents).to(self.device)
            else:
                obs = {k: obs_tensor.reshape(bs, -1)}
                if len(actions_tensor.shape) == 2:
                    actions = {k: actions_tensor.reshape(bs)}
                elif len(actions_tensor.shape) == 3:
                    actions = {k: actions_tensor.reshape(bs, -1)}
                else:
                    raise AttributeError("Wrong actions shape.")
                values = {k: values_tensor.reshape(bs)}
                returns = {k: returns_tensor.reshape(bs)}
                advantages = {k: advantages_tensor.reshape(bs)}
                log_pi_old = {k: log_pi_old_tensor.reshape(bs)}
                terminals = {k: ter_tensor.reshape(bs)}
                agent_mask = {k: msk_tensor.reshape(bs)}
                IDs = torch.eye(self.n_agents).unsqueeze(0).expand(
                    batch_size, -1, -1).reshape(bs, self.n_agents).to(self.device)

            if use_actions_mask:
                avail_a = np.stack(itemgetter(*self.agent_keys)(sample['avail_actions']), axis=1)
                if self.use_rnn:
                    avail_actions = {k: Tensor(avail_a.reshape([bs, seq_length, -1])).float().to(self.device)}
                else:
                    avail_actions = {k: Tensor(avail_a.reshape([bs, -1])).float().to(self.device)}

        else:
            obs = {k: Tensor(sample['obs'][k]).to(self.device) for k in self.agent_keys}
            actions = {k: Tensor(sample['actions'][k]).to(self.device) for k in self.agent_keys}
            values = {k: Tensor(sample['values'][k]).to(self.device) for k in self.agent_keys}
            returns = {k: Tensor(sample['returns'][k]).to(self.device) for k in self.agent_keys}
            advantages = {k: Tensor(sample['advantages'][k]).to(self.device) for k in self.agent_keys}
            log_pi_old = {k: Tensor(sample['log_pi_old'][k]).to(self.device) for k in self.agent_keys}
            terminals = {k: Tensor(sample['terminals'][k]).float().to(self.device) for k in self.agent_keys}
            agent_mask = {k: Tensor(sample['agent_mask'][k]).float().to(self.device) for k in self.agent_keys}
            if use_actions_mask:
                avail_actions = {k: Tensor(sample['avail_actions'][k]).float().to(self.device) for k in self.agent_keys}

        if use_global_state:
            state = Tensor(sample['state']).to(self.device)

        if self.use_rnn:
            filled = Tensor(sample['filled']).float().to(self.device)

        sample_Tensor = {
            'batch_size': batch_size,
            'state': state,
            'obs': obs,
            'actions': actions,
            'values': values,
            'returns': returns,
            'advantages': advantages,
            'log_pi_old': log_pi_old,
            'terminals': terminals,
            'agent_mask': agent_mask,
            'avail_actions': avail_actions,
            'agent_ids': IDs,
            'filled': filled,
            'seq_length': seq_length,
        }
        return sample_Tensor

    def update(self, sample):
        return {}
        info = {}
        self.iterations += 1
        state = torch.Tensor(sample['state']).to(self.device)
        obs = torch.Tensor(sample['obs']).to(self.device)
        actions = torch.Tensor(sample['actions']).to(self.device)
        returns = torch.Tensor(sample['returns']).to(self.device)
        agent_mask = torch.Tensor(sample['agent_mask']).float().reshape(-1, self.n_agents, 1).to(self.device)
        batch_size = obs.shape[0]
        IDs = torch.eye(self.n_agents).unsqueeze(0).expand(batch_size, -1, -1).to(self.device)

        # actor loss
        _, pi_dist, value_pred = self.policy(obs, IDs)
        log_pi = pi_dist.log_prob(actions).unsqueeze(-1)
        entropy = pi_dist.entropy().reshape(agent_mask.shape) * agent_mask

        targets = returns
        advantages = targets - value_pred
        td_error = value_pred - targets.detach()

        pg_loss = -((advantages.detach() * log_pi) * agent_mask).sum() / agent_mask.sum()
        vf_loss = ((td_error ** 2) * agent_mask).sum() / agent_mask.sum()
        entropy_loss = (entropy * agent_mask).sum() / agent_mask.sum()
        loss = pg_loss + self.vf_coef * vf_loss - self.ent_coef * entropy_loss

        self.optimizer.zero_grad()
        loss.backward()
        if self.use_grad_clip:
            grad_norm = torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.grad_clip_norm)
            info["gradient_norm"] = grad_norm.item()
        self.optimizer.step()
        if self.scheduler is not None:
            self.scheduler.step()

        # Logger
        lr = self.optimizer.state_dict()['param_groups'][0]['lr']

        info.update({
            "learning_rate": lr,
            "pg_loss": pg_loss.item(),
            "vf_loss": vf_loss.item(),
            "entropy_loss": entropy_loss.item(),
            "loss": loss.item(),
            "predict_value": value_pred.mean().item()
        })

        return info

    def update_rnn(self, sample):
        info = {}
        self.iterations += 1
        state = torch.Tensor(sample['state']).to(self.device)
        obs = torch.Tensor(sample['obs']).to(self.device)
        actions = torch.Tensor(sample['actions']).to(self.device)
        returns = torch.Tensor(sample['returns']).to(self.device)
        avail_actions = torch.Tensor(sample['avail_actions']).float().to(self.device)
        filled = torch.Tensor(sample['filled']).float().to(self.device)
        batch_size = obs.shape[0]
        episode_length = actions.shape[2]
        IDs = torch.eye(self.n_agents).unsqueeze(1).unsqueeze(0).expand(batch_size, -1, episode_length + 1, -1).to(
            self.device)

        filled_n = filled.unsqueeze(1).expand(batch_size, self.n_agents, episode_length, 1)

        # actor loss
        rnn_hidden = self.policy.representation.init_hidden(batch_size * self.n_agents)
        _, pi_dist, value_pred = self.policy(obs[:, :, :-1].reshape(-1, episode_length, self.dim_obs),
                                             IDs[:, :, :-1].reshape(-1, episode_length, self.n_agents),
                                             *rnn_hidden,
                                             avail_actions=avail_actions[:, :, :-1].reshape(-1, episode_length, self.dim_act),
                                             state=state[:, :-1])
        log_pi = pi_dist.log_prob(actions.reshape(-1, episode_length)).reshape(batch_size, self.n_agents, episode_length, 1)
        entropy = pi_dist.entropy().reshape(batch_size, self.n_agents, episode_length, 1)

        targets = returns
        advantages = targets - value_pred
        td_error = value_pred - targets.detach()

        pg_loss = -((advantages.detach() * log_pi) * filled_n).sum() / filled_n.sum()
        vf_loss = ((td_error ** 2) * filled_n).sum() / filled_n.sum()
        entropy_loss = (entropy * filled_n).sum() / filled_n.sum()
        loss = pg_loss + self.vf_coef * vf_loss - self.ent_coef * entropy_loss

        self.optimizer.zero_grad()
        loss.backward()
        if self.use_grad_clip:
            grad_norm = torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.grad_clip_norm)
            info["gradient_norm"] = grad_norm.item()
        self.optimizer.step()
        if self.scheduler is not None:
            self.scheduler.step()

        # Logger
        lr = self.optimizer.state_dict()['param_groups'][0]['lr']

        info.update({
            "learning_rate": lr,
            "pg_loss": pg_loss.item(),
            "vf_loss": vf_loss.item(),
            "entropy_loss": entropy_loss.item(),
            "loss": loss.item(),
            "predict_value": value_pred.mean().item()
        })

        return info
