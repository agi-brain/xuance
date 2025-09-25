"""
MFAC: Mean Field Actor-Critic
Paper link:
http://proceedings.mlr.press/v80/yang18d/yang18d.pdf
Implementation: Pytorch
"""
import numpy as np
from argparse import Namespace
from operator import itemgetter
from xuance.common import Optional, List
from xuance.mindspore import ops, Tensor, Module
from xuance.mindspore.utils import clip_grads
from xuance.mindspore.learners.multi_agent_rl.ippo_learner import IPPO_Learner


class MFAC_Learner(IPPO_Learner):
    def __init__(self,
                 config: Namespace,
                 model_keys: List[str],
                 agent_keys: List[str],
                 policy: Module):
        super(MFAC_Learner, self).__init__(config, model_keys, agent_keys, policy)

    def build_actions_mean_input(self, sample: Optional[dict], use_parameter_sharing: Optional[bool] = False):
        batch_size = sample['batch_size']
        seq_length = sample['sequence_length'] if self.use_rnn else 1
        if use_parameter_sharing:
            k = self.model_keys[0]
            bs = batch_size * self.n_agents
            if self.n_agents == 1:
                actions_mean_tensor = Tensor(sample['actions_mean'][k]).unsqueeze(1)
            else:
                actions_mean_tensor = Tensor(np.stack(itemgetter(*self.agent_keys)(sample['actions_mean']), axis=1))
            if self.use_rnn:
                actions_mean = {k: actions_mean_tensor.reshape(bs, seq_length, -1)}
            else:
                actions_mean = {k: actions_mean_tensor.reshape(bs, -1)}
        else:
            actions_mean = {k: Tensor(sample['actions_mean'][k]) for k in self.agent_keys}

        return actions_mean

    def forward_fn(self, *args):
        bs, obs, actions, act_mean, values, returns, advantages, log_pi_old, agent_mask, avail_actions, IDs = args
        value_pred = {}
        pi_dist_mu, pi_dist_std, pi_dist_logits = {}, {}, {}

        # feedforward
        if self.is_continuous:
            _, pi_dist_mu, pi_dist_std = self.policy(observation=obs, agent_ids=IDs, avail_actions=avail_actions)
        else:
            _, pi_dist_logits = self.policy(observation=obs, agent_ids=IDs, avail_actions=avail_actions)

        _, value_pred_dict = self.policy.get_values(observation=obs, actions_mean=act_mean, agent_ids=IDs)

        # calculate losses for each agent
        loss_a, loss_e, loss_c = [], [], []
        for key in self.model_keys:
            mask_values = agent_mask[key]
            # actor loss
            if self.is_continuous:
                log_pi = self.pi_dist[key]._log_prob(value=actions[key], mean=pi_dist_mu[key], sd=pi_dist_std[key])
                log_pi = ops.reduce_sum(x=log_pi, axis=-1)
                entropy = self.pi_dist[key]._entropy(mean=pi_dist_mu[key], sd=pi_dist_std[key])
                entropy = ops.reduce_sum(x=entropy, axis=-1)
            else:
                probs = self.softmax(pi_dist_logits[key])
                log_pi = self.pi_dist[key]._log_prob(value=actions[key], probs=probs)
                entropy = self.pi_dist[key].entropy(probs=probs)
            ratio = ops.exp(log_pi - log_pi_old[key]).reshape(bs)
            advantages_mask = ops.stop_gradient(advantages[key]) * mask_values
            surrogate1 = ratio * advantages_mask
            surrogate2 = ops.clip_by_value(ratio, Tensor(1 - self.clip_range), Tensor(1 + self.clip_range))
            surrogate2 *= advantages_mask
            loss_a.append(-ops.minimum(surrogate1, surrogate2).mean())

            # entropy loss
            entropy_loss = (entropy * mask_values).sum() / mask_values.sum()
            loss_e.append(entropy_loss)

            # critic loss
            value_pred_i = value_pred_dict[key].reshape(bs)
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
                loss_c_ = ops.maximum(loss_v, loss_v_clipped) * mask_values
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

            value_pred.update({
                f"predict_value/{key}": value_pred_i.mean().asnumpy()
            })

        loss = sum(loss_a) + self.vf_coef * sum(loss_c) - self.ent_coef * sum(loss_e)
        return loss, sum(loss_a), sum(loss_c), sum(loss_e), value_pred

    def update(self, sample):
        self.iterations += 1
        info = {}

        # prepare training data
        act_mean = self.build_actions_mean_input(sample=sample,
                                                 use_parameter_sharing=self.use_parameter_sharing)
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
        log_pi_old = sample_Tensor['log_pi_old']
        IDs = sample_Tensor['agent_ids']

        bs = batch_size * self.n_agents if self.use_parameter_sharing else batch_size

        # info = self.callback.on_update_start(self.iterations, method="update", actions_mean=act_mean,
        #                                      policy=self.policy, sample_Tensor=sample_Tensor, bs=bs)

        (loss, loss_a, loss_c, loss_e, value_pred), grads = self.grad_fn(bs, obs, actions, act_mean,
                                                                         values, returns, advantages, log_pi_old,
                                                                         agent_mask, avail_actions, IDs)
        if self.use_grad_clip:
            grads = clip_grads(grads, Tensor(-self.grad_clip_norm), Tensor(self.grad_clip_norm))
        self.optimizer(grads)

        self.scheduler.step()
        lr = self.scheduler.get_last_lr()[0]

        info.update({
            "learning_rate": lr.asnumpy(),
            "loss": loss.asnumpy(),
            "loss_a": loss_a.asnumpy(),
            "loss_c": loss_c.asnumpy(),
            "loss_e": loss_e.asnumpy()
        })
        info.update(value_pred)

        return info
