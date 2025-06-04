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
from xuance.common import List
from xuance.torch.learners.multi_agent_rl.iac_learner import IAC_Learner


class VDAC_Learner(IAC_Learner):
    def __init__(self,
                 config: Namespace,
                 model_keys: List[str],
                 agent_keys: List[str],
                 policy: nn.Module,
                 callback):
        super(VDAC_Learner, self).__init__(config, model_keys, agent_keys, policy, callback)
        self.use_global_state = True if self.policy.mixer == "QMIX" else False

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
        IDs = sample_Tensor['agent_ids']

        bs = batch_size * self.n_agents if self.use_parameter_sharing else batch_size

        info = self.callback.on_update_start(self.iterations, method="update",
                                             policy=self.policy, sample_Tensor=sample_Tensor, bs=bs)

        # feedforward
        _, pi_dist_dict = self.policy(observation=obs, agent_ids=IDs, avail_actions=avail_actions)
        _, values_pred_individual = self.policy.get_values(observation=obs, agent_ids=IDs)
        if self.use_parameter_sharing:
            values_n = values_pred_individual[self.model_keys[0]].reshape(batch_size, self.n_agents)
        else:
            values_n = self.get_joint_input(values_pred_individual)
        if self.config.mixer == "VDN":
            values_tot = self.policy.value_tot(values_n)
        elif self.config.mixer == "QMIX":
            values_tot = self.policy.value_tot(values_n, state)
        else:
            raise NotImplementedError("Mixer not implemented.")
        if self.use_parameter_sharing:
            values_tot = values_tot.reshape(batch_size, 1).repeat(1, self.n_agents).reshape(bs)
        values_pred_dict = {k: values_tot for k in self.model_keys}

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
            "predict_value": values_tot.mean().item()
        })

        info.update(self.callback.on_update_end(self.iterations, method="update", policy=self.policy, info=info))

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
        IDs = sample_Tensor['agent_ids']

        if self.use_parameter_sharing:
            filled = filled.unsqueeze(1).expand(batch_size, self.n_agents, seq_len).reshape(bs_rnn, seq_len)

        info = self.callback.on_update_start(self.iterations, method="update_rnn",
                                             policy=self.policy, sample_Tensor=sample_Tensor, bs_rnn=bs_rnn)

        rnn_hidden_actor = {k: self.policy.actor_representation[k].init_hidden(bs_rnn) for k in self.model_keys}
        rnn_hidden_critic = {k: self.policy.critic_representation[k].init_hidden(bs_rnn) for k in self.model_keys}

        # feedforward
        _, pi_dist_dict = self.policy(obs, agent_ids=IDs, avail_actions=avail_actions, rnn_hidden=rnn_hidden_actor)
        _, values_pred_individual = self.policy.get_values(obs, agent_ids=IDs, rnn_hidden=rnn_hidden_critic)
        if self.use_parameter_sharing:
            values_n = values_pred_individual[self.model_keys[0]].reshape(
                batch_size, self.n_agents, seq_len).transpose(1, 2).reshape(-1, self.n_agents)
        else:
            if self.n_agents == 1:
                values_n = values_pred_individual[self.agent_keys[0]].reshape(-1, self.n_agents)
            else:
                values_n = torch.stack(itemgetter(*self.agent_keys)(values_pred_individual),
                                       dim=2).reshape(-1, self.n_agents)
        if self.config.mixer == "VDN":
            values_tot = self.policy.value_tot(values_n)
        elif self.config.mixer == "QMIX":
            values_tot = self.policy.value_tot(values_n, state)
        else:
            raise NotImplementedError("Mixer not implemented.")
        if self.use_parameter_sharing:
            values_tot = values_tot.reshape(batch_size, 1, seq_len).repeat(1, self.n_agents, 1)
        else:
            values_tot = values_tot.reshape(batch_size, seq_len)
        values_pred_dict = {k: values_tot for k in self.model_keys}

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
            "predict_value": values_tot.mean().item()
        })

        info.update(self.callback.on_update_end(self.iterations, method="update_rnn", policy=self.policy, info=info))

        return info
