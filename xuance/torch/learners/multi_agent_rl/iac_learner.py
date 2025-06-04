"""
Independent Advantage Actor Critic (IAC)
Paper link: https://ojs.aaai.org/index.php/AAAI/article/view/11794
Implementation: Pytorch
"""
import numpy as np
import torch
from torch import nn
from argparse import Namespace
from operator import itemgetter
from xuance.common import Optional, List
from xuance.torch import Tensor
from xuance.torch.utils import ValueNorm
from xuance.torch.learners import LearnerMAS


class IAC_Learner(LearnerMAS):
    def __init__(self,
                 config: Namespace,
                 model_keys: List[str],
                 agent_keys: List[str],
                 policy: nn.Module,
                 callback):
        super(IAC_Learner, self).__init__(config, model_keys, agent_keys, policy, callback)
        self.build_optimizer()
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

    def build_optimizer(self):
        self.optimizer = torch.optim.Adam(self.policy.parameters_model, lr=self.learning_rate, eps=1e-5,
                                          weight_decay=self.config.weight_decay)
        self.scheduler = torch.optim.lr_scheduler.LinearLR(self.optimizer,
                                                           start_factor=1.0,
                                                           end_factor=self.end_factor_lr_decay,
                                                           total_iters=self.config.running_steps)

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
        self.iterations += 1

        # prepare training data
        sample_Tensor = self.build_training_data(sample=sample,
                                                 use_parameter_sharing=self.use_parameter_sharing,
                                                 use_actions_mask=self.use_actions_mask)
        batch_size = sample_Tensor['batch_size']
        obs = sample_Tensor['obs']
        actions = sample_Tensor['actions']
        agent_mask = sample_Tensor['agent_mask']
        avail_actions = sample_Tensor['avail_actions']
        values = sample_Tensor['values']
        returns = sample_Tensor['returns']
        advantages = sample_Tensor['advantages']
        IDs = sample_Tensor['agent_ids']

        bs = batch_size * self.n_agents if self.use_parameter_sharing else batch_size

        info = self.callback.on_update_start(self.iterations, method="update",
                                             policy=self.policy, sample_Tensor=sample_Tensor, bs=bs)

        # feedforward
        _, pi_dist_dict = self.policy(observation=obs, agent_ids=IDs, avail_actions=avail_actions)
        _, values_pred_dict = self.policy.get_values(observation=obs, agent_ids=IDs)

        loss_a, loss_e, loss_c = [], [], []
        for key in self.model_keys:
            mask_values = agent_mask[key]
            # policy gradient loss
            log_pi = pi_dist_dict[key].log_prob(actions[key])
            pg_loss = -((advantages[key].detach() * log_pi) * mask_values).sum() / mask_values.sum()
            loss_a.append(pg_loss)

            # entropy loss
            entropy = pi_dist_dict[key].entropy()
            entropy_loss = (entropy * mask_values).sum() / mask_values.sum()
            loss_e.append(entropy_loss)

            # value loss
            value_pred_i = values_pred_dict[key].reshape(bs)
            value_target = returns[key].reshape(bs)
            values_i = values[key].reshape(bs)
            if self.use_value_clip:
                value_clipped = values_i + (value_pred_i - values_i).clamp(-self.value_clip_range,
                                                                           self.value_clip_range)
                if self.use_value_norm:
                    self.value_normalizer[key].update(value_target.reshape(bs, 1))
                    value_target = self.value_normalizer[key].normalize(value_target.reshape(bs, 1)).reshape(bs)
                if self.use_huber_loss:
                    loss_v = self.huber_loss(value_pred_i, value_target)
                    loss_v_clipped = self.huber_loss(value_clipped, value_target)
                else:
                    loss_v = (value_pred_i - value_target) ** 2
                    loss_v_clipped = (value_clipped - value_target) ** 2
                loss_c_ = torch.max(loss_v, loss_v_clipped) * mask_values
                loss_c.append(loss_c_.sum() / mask_values.sum())
            else:
                if self.use_value_norm:
                    self.value_normalizer[key].update(value_target)
                    value_target = self.value_normalizer[key].normalize(value_target)
                if self.use_huber_loss:
                    loss_v = self.huber_loss(value_pred_i, value_target) * mask_values
                else:
                    loss_v = ((value_pred_i - value_target) ** 2) * mask_values
                loss_c.append(loss_v.sum() / mask_values.sum())

            info.update({
                f"predict_value/{key}": value_pred_i.mean().item()
            })

            info.update(self.callback.on_update_agent_wise(self.iterations, key, info=info, method="update",
                                                           mask_values=mask_values, log_pi=log_pi, pg_loss=pg_loss,
                                                           entropy=entropy, entropy_loss=entropy_loss,
                                                           value_pred_i=value_pred_i, value_target=value_target,
                                                           values_i=values_i, loss_v=loss_v))

        # Total loss
        loss = sum(loss_a) + self.vf_coef * sum(loss_c) - self.ent_coef * sum(loss_e)
        self.optimizer.zero_grad()
        loss.backward()
        if self.use_grad_clip:
            grad_norm = torch.nn.utils.clip_grad_norm_(self.policy.parameters_model, self.grad_clip_norm)
            info["gradient_norm"] = grad_norm.item()
        self.optimizer.step()
        if self.scheduler is not None:
            self.scheduler.step()

        # Logger
        lr = self.optimizer.state_dict()['param_groups'][0]['lr']

        info.update({
            "learning_rate": lr,
            "pg_loss": sum(loss_a).item(),
            "vf_loss": sum(loss_c).item(),
            "entropy_loss": sum(loss_e).item(),
            "loss": loss.item(),
        })

        info.update(self.callback.on_update_end(self.iterations, method="update", policy=self.policy, info=info))

        return info

    def update_rnn(self, sample):
        self.iterations += 1

        sample_Tensor = self.build_training_data(sample=sample,
                                                 use_parameter_sharing=self.use_parameter_sharing,
                                                 use_actions_mask=self.use_actions_mask)
        batch_size = sample_Tensor['batch_size']
        bs_rnn = batch_size * self.n_agents if self.use_parameter_sharing else batch_size
        obs = sample_Tensor['obs']
        actions = sample_Tensor['actions']
        values = sample_Tensor['values']
        returns = sample_Tensor['returns']
        advantages = sample_Tensor['advantages']
        avail_actions = sample_Tensor['avail_actions']
        agent_mask = sample_Tensor['agent_mask']
        filled = sample_Tensor['filled']
        seq_len = filled.shape[1]
        IDs = sample_Tensor['agent_ids']

        if self.use_parameter_sharing:
            filled = filled.unsqueeze(1).expand(batch_size, self.n_agents, seq_len).reshape(bs_rnn, seq_len)

        info = self.callback.on_update_start(self.iterations, method="update_rnn",
                                             policy=self.policy, sample_Tensor=sample_Tensor, bs_rnn=bs_rnn)

        rnn_hidden_actor = {k: self.policy.actor_representation[k].init_hidden(bs_rnn) for k in self.model_keys}
        rnn_hidden_critic = {k: self.policy.critic_representation[k].init_hidden(bs_rnn) for k in self.model_keys}

        # feedforward
        _, pi_dist_dict = self.policy(obs, agent_ids=IDs, avail_actions=avail_actions, rnn_hidden=rnn_hidden_actor)
        _, values_pred_dict = self.policy.get_values(obs, agent_ids=IDs, rnn_hidden=rnn_hidden_critic)

        # calculate losses for each agent
        loss_a, loss_e, loss_c = [], [], []
        for key in self.model_keys:
            mask_values = agent_mask[key] * filled
            # policy gradient loss
            log_pi = pi_dist_dict[key].log_prob(actions[key]).reshape(bs_rnn, seq_len)
            pg_loss = -((advantages[key].detach() * log_pi) * mask_values).sum() / mask_values.sum()
            loss_a.append(pg_loss)

            # entropy loss
            entropy = pi_dist_dict[key].entropy()
            entropy_loss = (entropy * mask_values).sum() / mask_values.sum()
            loss_e.append(entropy_loss)

            # value loss
            value_pred_i = values_pred_dict[key].reshape(bs_rnn, seq_len)
            value_target = returns[key].reshape(bs_rnn, seq_len)
            values_i = values[key].reshape(bs_rnn, seq_len)
            if self.use_value_clip:
                value_clipped = values_i + (value_pred_i - values_i).clamp(-self.value_clip_range,
                                                                           self.value_clip_range)
                if self.use_value_norm:
                    self.value_normalizer[key].update(value_target.reshape(-1, 1))
                    value_target = self.value_normalizer[key].normalize(value_target.reshape(-1, 1))
                    value_target = value_target.reshape(bs_rnn, seq_len)
                if self.use_huber_loss:
                    loss_v = self.huber_loss(value_pred_i, value_target)
                    loss_v_clipped = self.huber_loss(value_clipped, value_target)
                else:
                    loss_v = (value_pred_i - value_target) ** 2
                    loss_v_clipped = (value_clipped - value_target) ** 2
                loss_c_ = torch.max(loss_v, loss_v_clipped) * mask_values
                loss_c.append(loss_c_.sum() / mask_values.sum())
            else:
                if self.use_value_norm:
                    self.value_normalizer[key].update(value_target)
                    value_target = self.value_normalizer[key].normalize(value_target)
                if self.use_huber_loss:
                    loss_v = self.huber_loss(value_pred_i, value_target)
                else:
                    loss_v = (value_pred_i - value_target) ** 2
                loss_c.append((loss_v * mask_values).sum() / mask_values.sum())

            info.update({
                f"predict_value/{key}": value_pred_i.mean().item()
            })

            info.update(self.callback.on_update_agent_wise(self.iterations, key, info=info, method="update_rnn",
                                                           mask_values=mask_values, log_pi=log_pi,
                                                           pg_loss=pg_loss, entropy=entropy,
                                                           entropy_loss=entropy_loss, value_pred_i=value_pred_i,
                                                           value_target=value_target, values_i=values_i,
                                                           loss_v=loss_v))

        loss = sum(loss_a) + self.vf_coef * sum(loss_c) - self.ent_coef * sum(loss_e)
        self.optimizer.zero_grad()
        loss.backward()
        if self.use_grad_clip:
            grad_norm = torch.nn.utils.clip_grad_norm_(self.policy.parameters_model, self.grad_clip_norm)
            info["gradient_norm"] = grad_norm.item()
        self.optimizer.step()
        if self.scheduler is not None:
            self.scheduler.step()

        # Logger
        lr = self.optimizer.state_dict()['param_groups'][0]['lr']

        info.update({
            "learning_rate": lr,
            "pg_loss": sum(loss_a).item(),
            "vf_loss": sum(loss_c).item(),
            "entropy_loss": sum(loss_e).item(),
            "loss": loss.item(),
        })

        info.update(self.callback.on_update_end(self.iterations, method="update_rnn", policy=self.policy, info=info))

        return info
