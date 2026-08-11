"""
Value-Dcomposition Actor-Critic (VDAC)
Paper link:
https://ojs.aaai.org/index.php/AAAI/article/view/17353
Implementation: Pytorch
"""
import torch
from torch import nn
from argparse import Namespace
from operator import itemgetter
from xuance.common import AgentGrouping
from xuance.torch.learners.multi_agent_rl.iac_learner import IAC_Learner


class VDAC_Learner(IAC_Learner):
    def __init__(self,
                 config: Namespace,
                 agent_grouping: AgentGrouping,
                 model: nn.Module,
                 callback):
        super(VDAC_Learner, self).__init__(config, agent_grouping, model, callback)
        self.use_global_state = True if config.mixer == "QMIX" else getattr(config, "use_global_state", False)

    def update(self, sample):
        self.iterations += 1

        # prepare training data
        sample_Tensor = self.build_training_data(sample=sample,
                                                 use_parameter_sharing=self.use_parameter_sharing,
                                                 use_actions_mask=self.use_actions_mask,
                                                 use_global_state=self.use_global_state)
        batch_size = sample_Tensor['batch_size']
        state = sample_Tensor['state']
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
        values_pred_individual = self.model.get_values(observations=obs, agent_indices=agent_indices).values
        values_tot = self.model.values_tot(values_pred_individual, state).reshape(batch_size, 1)
        values_pred_dict = {k: values_tot for k in self.agent_keys}
        if not self.agent_grouping.full_independent:
            values_pred_dict = self.packed_tensor(values_pred_dict)

        loss_a, loss_e, loss_c = [], [], []
        for group, agent_keys in self.groups.items():
            bs = batch_size * self.n_group_agents[group]
            mask_values = agent_mask[group]
            # model gradient loss
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
            "predict_value": values_tot.mean().item()
        })

        info.update(self.callback.on_update_end(self.iterations, method="update", model=self.model, info=info))

        return info

    def update_rnn(self, sample):
        self.iterations += 1

        sample_Tensor = self.build_training_data(sample=sample,
                                                 use_parameter_sharing=self.use_parameter_sharing,
                                                 use_actions_mask=self.use_actions_mask,
                                                 use_global_state=self.use_global_state)
        batch_size = sample_Tensor['batch_size']
        state = sample_Tensor['state']
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
        agent_indices = sample_Tensor['agent_indices']

        if not self.agent_grouping.full_independent:
            obs = self.packed_tensor(obs)
            actions = self.packed_tensor(actions)
            advantages = self.packed_tensor(advantages)
            returns = self.packed_tensor(returns)
            values = self.packed_tensor(values)
            agent_mask = self.packed_tensor(agent_mask)

        info = self.callback.on_update_start(self.iterations, method="update_rnn",
                                             model=self.model, sample_Tensor=sample_Tensor, bs_rnn=bs_rnn)

        # initial hidden states for rnn
        rnn_states_actor = self.model.init_actor_rnn_states(batch_size)
        rnn_states_critic = self.model.init_critic_rnn_states(batch_size)

        # feedforward
        pi_dist_dict = self.model(observations=obs, agent_indices=agent_indices,
                                  avail_actions=avail_actions, rnn_states=rnn_states_actor).distributions
        values_pred_individual = self.model.get_values(observations=obs, agent_indices=agent_indices,
                                                       rnn_states=rnn_states_critic).values
        values_tot = self.model.values_tot(values_pred_individual, state).reshape(batch_size, seq_len)
        values_pred_dict = {k: values_tot for k in self.agent_keys}
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
            log_pi = pi_dist_dict[group].log_prob(actions[group]).reshape(bs_rnn, seq_len)
            pg_loss = -((advantages[group].detach() * log_pi) * mask_values).sum() / mask_values.sum()
            loss_a.append(pg_loss)

            # entropy loss
            entropy = pi_dist_dict[group].entropy()
            entropy_loss = (entropy * mask_values).sum() / mask_values.sum()
            loss_e.append(entropy_loss)

            # value loss
            value_pred_i = values_pred_dict[group].reshape(bs_rnn, seq_len)
            value_target = returns[group].reshape(bs_rnn, seq_len)
            values_i = values[group].reshape(bs_rnn, seq_len)
            if self.use_value_clip:
                value_clipped = values_i + (value_pred_i - values_i).clamp(-self.value_clip_range,
                                                                           self.value_clip_range)
                if self.use_value_norm:
                    self.value_normalizer[group].update(value_target.reshape(-1, 1))
                    value_target = self.value_normalizer[group].normalize(value_target.reshape(-1, 1))
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
                    self.value_normalizer[group].update(value_target)
                    value_target = self.value_normalizer[group].normalize(value_target)
                if self.use_huber_loss:
                    loss_v = self.huber_loss(value_pred_i, value_target)
                else:
                    loss_v = (value_pred_i - value_target) ** 2
                loss_c.append((loss_v * mask_values).sum() / mask_values.sum())

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
            "predict_value": values_tot.mean().item()
        })

        info.update(self.callback.on_update_end(self.iterations, method="update_rnn", model=self.model, info=info))

        return info
