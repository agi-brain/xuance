"""
Independent Advantage Actor Critic (IAC)
Paper link: https://ojs.aaai.org/index.php/AAAI/article/view/11794
Implementation: Pytorch
"""
import torch
from torch import nn
from argparse import Namespace
from xuance.common import Optional, AgentGrouping
from xuance.torch.utils import ValueNorm
from xuance.torch.learners import LearnerMAS


class IAC_Learner(LearnerMAS):
    def __init__(self,
                 config: Namespace,
                 agent_grouping: AgentGrouping,
                 model: nn.Module,
                 callback):
        super(IAC_Learner, self).__init__(config, agent_grouping, model, callback)
        self.build_optimizer()
        self.use_value_clip, self.value_clip_range = config.use_value_clip, config.value_clip_range
        self.use_huber_loss, self.huber_delta = config.use_huber_loss, config.huber_delta
        self.use_value_norm = config.use_value_norm
        self.vf_coef, self.ent_coef = config.vf_coef, config.ent_coef
        self.mse_loss = nn.MSELoss()
        self.huber_loss = nn.HuberLoss(reduction="none", delta=self.huber_delta)
        if self.use_value_norm:
            self.value_normalizer = {key: ValueNorm(1).to(self.device) for key in self.group_keys}
        else:
            self.value_normalizer = None

    def estimate_total_iterations(self):
        """Estimated total number of training iterations"""
        buffer_size = self.config.buffer_size
        n_epochs = getattr(self.config, "n_epochs", 1)
        n_minibatch = getattr(self.config, "n_minibatch", 1)
        episode_length = self.episode_length
        if self.use_rnn:
            update_times = (self.config.running_steps // episode_length) // buffer_size
        else:
            update_times = self.config.running_steps // buffer_size
        total_iters = update_times * n_epochs * n_minibatch
        return total_iters

    def build_optimizer(self):
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate, eps=1e-5,
                                          weight_decay=self.config.weight_decay)
        self.scheduler = torch.optim.lr_scheduler.LinearLR(self.optimizer,
                                                           start_factor=1.0,
                                                           end_factor=self.end_factor_lr_decay,
                                                           total_iters=self.total_iters)

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
        state, filled = None, None
        obs, actions, rewards, terminals, agent_mask = {}, {}, {}, {}, {}
        values, returns, advantages, log_pi_old = {}, {}, {}, {}
        avail_actions = {} if self.use_actions_mask else None
        agent_indices = {}

        for agent in self.agent_keys:
            obs[agent] = torch.as_tensor(sample['obs'][agent], device=self.device)
            actions[agent] = torch.as_tensor(sample['actions'][agent], device=self.device)
            rewards[agent] = torch.as_tensor(sample['rewards'][agent], device=self.device)
            terminals[agent] = torch.as_tensor(sample['terminals'][agent], device=self.device, dtype=torch.float32)
            agent_mask[agent] = torch.as_tensor(sample['agent_mask'][agent], device=self.device, dtype=torch.float32)
            values[agent] = torch.as_tensor(sample['values'][agent], device=self.device)
            returns[agent] = torch.as_tensor(sample['returns'][agent], device=self.device)
            advantages[agent] = torch.as_tensor(sample['advantages'][agent], device=self.device)
            log_pi_old[agent] = torch.as_tensor(sample['log_pi_old'][agent], device=self.device)
            if use_actions_mask:
                avail_actions[agent] = torch.as_tensor(sample['avail_actions'][agent],
                                                       device=self.device, dtype=torch.float32)

        if use_global_state:
            state = torch.as_tensor(sample['state'], device=self.device)

        if self.use_rnn:
            filled = torch.as_tensor(sample['filled'], device=self.device, dtype=torch.float32)

        for key in self.group_keys:
            n_agents = self.n_group_agents[key]
            bs = batch_size * n_agents

            if self.use_rnn:
                agents_id = torch.as_tensor(self.agent_grouping.agent_indices(key), dtype=torch.int64).repeat(
                    batch_size, 1).reshape(bs, 1, 1).expand(-1, seq_length, -1).to(self.device)
            else:
                agents_id = torch.as_tensor(self.agent_indices[key],
                                            dtype=torch.int64).repeat(batch_size, 1).reshape([bs, 1]).to(self.device)

            agent_indices[key] = agents_id

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
            'agent_indices': agent_indices,
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
        agent_indices = sample_Tensor['agent_indices']

        if not self.agent_grouping.full_independent:
            obs = self.packed_tensor(obs)
            actions = self.packed_tensor(actions)
            advantages = self.packed_tensor(advantages)
            returns = self.packed_tensor(returns)
            values = self.packed_tensor(values)
            agent_mask = self.packed_tensor(agent_mask)

        info = self.callback.on_update_start(self.iterations, method="update",
                                             model=self.model, sample_Tensor=sample_Tensor)

        # feedforward
        pi_dist_dict = self.model(observations=obs, agent_indices=agent_indices,
                                  avail_actions=avail_actions).distributions
        values_pred_dict = self.model.get_values(observations=obs, agent_indices=agent_indices).values
        if not self.agent_grouping.full_independent:
            values_pred_dict = self.packed_tensor(values_pred_dict)

        loss_a, loss_e, loss_c = [], [], []
        for group, agent_keys in self.groups.items():
            bs = batch_size * self.n_group_agents[group]
            mask_values = agent_mask[group]
            # policy gradient loss
            log_pi = pi_dist_dict[group].log_prob(actions[group])
            pg_loss = -((advantages[group].detach() * log_pi) * mask_values).sum() / mask_values.sum()
            loss_a.append(pg_loss)

            # entropy loss
            entropy = pi_dist_dict[group].entropy()
            entropy_loss = (entropy * mask_values).sum() / mask_values.sum()
            loss_e.append(entropy_loss)

            # value loss
            value_pred_i = values_pred_dict[group].reshape(bs)
            value_target = returns[group].reshape(bs)
            values_i = values[group].reshape(bs)
            if self.use_value_clip:
                value_clipped = values_i + (value_pred_i - values_i).clamp(-self.value_clip_range,
                                                                           self.value_clip_range)
                if self.use_value_norm:
                    self.value_normalizer[group].update(value_target.reshape(bs, 1))
                    value_target = self.value_normalizer[group].normalize(value_target.reshape(bs, 1)).reshape(bs)
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
                    self.value_normalizer[group].update(value_target)
                    value_target = self.value_normalizer[group].normalize(value_target)
                if self.use_huber_loss:
                    loss_v = self.huber_loss(value_pred_i, value_target) * mask_values
                else:
                    loss_v = ((value_pred_i - value_target) ** 2) * mask_values
                loss_c.append(loss_v.sum() / mask_values.sum())

            info.update({
                f"predict_value/{group}": value_pred_i.mean().item()
            })

            info.update(self.callback.on_update_agent_wise(self.iterations, group, info=info, method="update",
                                                           mask_values=mask_values, log_pi=log_pi, pg_loss=pg_loss,
                                                           entropy=entropy, entropy_loss=entropy_loss,
                                                           value_pred_i=value_pred_i, value_target=value_target,
                                                           values_i=values_i, loss_v=loss_v))

        # Total loss
        loss = sum(loss_a) + self.vf_coef * sum(loss_c) - self.ent_coef * sum(loss_e)
        self.optimizer.zero_grad()
        loss.backward()
        if self.use_grad_clip:
            grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)
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

        info.update(self.callback.on_update_end(self.iterations, method="update", model=self.model, info=info))

        return info

    def update_rnn(self, sample):
        self.iterations += 1

        sample_Tensor = self.build_training_data(sample=sample,
                                                 use_parameter_sharing=self.use_parameter_sharing,
                                                 use_actions_mask=self.use_actions_mask)
        batch_size = sample_Tensor['batch_size']
        obs = sample_Tensor['obs']
        actions = sample_Tensor['actions']
        values = sample_Tensor['values']
        returns = sample_Tensor['returns']
        advantages = sample_Tensor['advantages']
        avail_actions = sample_Tensor['avail_actions']
        agent_mask = sample_Tensor['agent_mask']
        filled = sample_Tensor['filled']
        seq_len = filled.shape[1]
        agent_indices = sample_Tensor['agent_indices']

        if not self.agent_grouping.full_independent:
            obs = self.packed_tensor(obs)
            actions = self.packed_tensor(actions)
            advantages = self.packed_tensor(advantages)
            returns = self.packed_tensor(returns)
            values = self.packed_tensor(values)
            agent_mask = self.packed_tensor(agent_mask)

        info = self.callback.on_update_start(self.iterations, method="update_rnn",
                                             model=self.model, sample_Tensor=sample_Tensor)

        # initial hidden states for rnn
        rnn_states_actor = self.model.init_actor_rnn_states(batch_size)
        rnn_states_critic = self.model.init_critic_rnn_states(batch_size)

        # feedforward
        pi_dist_dict = self.model(observations=obs, agent_indices=agent_indices,
                                  avail_actions=avail_actions, rnn_states=rnn_states_actor).distributions
        values_pred_dict = self.model.get_values(observations=obs, agent_indices=agent_indices,
                                                 rnn_states=rnn_states_critic).values
        if not self.agent_grouping.full_independent:
            values_pred_dict = self.packed_tensor(values_pred_dict)

        # calculate losses for each agent
        loss_a, loss_e, loss_c = [], [], []
        for group, agent_keys in self.groups.items():
            n_agents = len(agent_keys)
            bs = batch_size * n_agents
            group_filled = filled.unsqueeze(1).expand(batch_size, n_agents, seq_len).reshape([bs, seq_len])
            mask_values = agent_mask[group] * group_filled
            # model gradient loss
            log_pi = pi_dist_dict[group].log_prob(actions[group]).reshape([bs, seq_len])
            pg_loss = -((advantages[group].detach() * log_pi) * mask_values).sum() / mask_values.sum()
            loss_a.append(pg_loss)

            # entropy loss
            entropy = pi_dist_dict[group].entropy()
            entropy_loss = (entropy * mask_values).sum() / mask_values.sum()
            loss_e.append(entropy_loss)

            # value loss
            value_pred_i = values_pred_dict[group].reshape(bs, seq_len)
            value_target = returns[group].reshape(bs, seq_len)
            values_i = values[group].reshape(bs, seq_len)
            if self.use_value_clip:
                value_clipped = values_i + (value_pred_i - values_i).clamp(-self.value_clip_range,
                                                                           self.value_clip_range)
                if self.use_value_norm:
                    self.value_normalizer[group].update(value_target.reshape(-1, 1))
                    value_target = self.value_normalizer[group].normalize(value_target.reshape(-1, 1))
                    value_target = value_target.reshape(bs, seq_len)
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
                    self.value_normalizer[group].update(value_target)
                    value_target = self.value_normalizer[group].normalize(value_target)
                if self.use_huber_loss:
                    loss_v = self.huber_loss(value_pred_i, value_target)
                else:
                    loss_v = (value_pred_i - value_target) ** 2
                loss_c.append((loss_v * mask_values).sum() / mask_values.sum())

            info.update({
                f"predict_value/{group}": value_pred_i.mean().item()
            })

            info.update(self.callback.on_update_agent_wise(self.iterations, group, info=info, method="update_rnn",
                                                           mask_values=mask_values, log_pi=log_pi,
                                                           pg_loss=pg_loss, entropy=entropy,
                                                           entropy_loss=entropy_loss, value_pred_i=value_pred_i,
                                                           value_target=value_target, values_i=values_i,
                                                           loss_v=loss_v))

        loss = sum(loss_a) + self.vf_coef * sum(loss_c) - self.ent_coef * sum(loss_e)
        self.optimizer.zero_grad()
        loss.backward()
        if self.use_grad_clip:
            grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)
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

        info.update(self.callback.on_update_end(self.iterations, method="update_rnn", model=self.model, info=info))

        return info
