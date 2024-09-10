"""
Multi-Agent Proximal Policy Optimization (MAPPO)
Paper link:
https://arxiv.org/pdf/2103.01955.pdf
Implementation: MindSpore
"""
import numpy as np
from mindspore.nn import MSELoss, HuberLoss
from xuance.mindspore import ms, Module, Tensor, optim, ops
from xuance.mindspore.learners import LearnerMAS
from xuance.mindspore.utils import clip_grads
from xuance.common import List, Optional
from xuance.mindspore.utils import ValueNorm
from argparse import Namespace
from operator import itemgetter
from xuance.mindspore.learners.multi_agent_rl.ippo_learner import IPPO_Learner


class MAPPO_Learner(IPPO_Learner):
    def __init__(self,
                 config: Namespace,
                 model_keys: List[str],
                 agent_keys: List[str],
                 policy: Module):
        super(MAPPO_Learner, self).__init__(config, model_keys, agent_keys, policy)

    def forward_fn(self, *args):
        bs, obs, actions, avail_actions, log_pi_old, values, returns, advantages, agt_mask, ids, critic_input = args
        # feedforward
        _, pi_dists_dict = self.policy(observation=obs, agent_ids=ids, avail_actions=avail_actions)
        _, value_pred_dict = self.policy.get_values(observation=critic_input, agent_ids=ids)

        # calculate losses for each agent
        loss_a, loss_e, loss_c = [], [], []
        for key in self.model_keys:
            mask_values = agt_mask[key]
            # actor loss
            log_pi = pi_dists_dict[key].log_prob(actions[key]).reshape(bs)
            ratio = ops.exp(log_pi - log_pi_old[key]).reshape(bs)
            advantages_mask = ops.stop_gradient(advantages[key]) * mask_values
            surrogate1 = ratio * advantages_mask
            surrogate2 = ops.clip(ratio, Tensor(1 - self.clip_range), Tensor(1 + self.clip_range)) * advantages_mask
            loss_a.append(-ops.minimum(surrogate1, surrogate2).mean())

            # entropy loss
            entropy = pi_dists_dict[key].entropy().reshape(bs) * mask_values
            loss_e.append(entropy.mean())

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
        loss = sum(loss_a) + self.vf_coef * sum(loss_c) - self.ent_coef * sum(loss_e)
        return loss, sum(loss_a), sum(loss_c), sum(loss_e)

    def update(self, sample):
        self.iterations += 1
        info = {}

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
        log_pi_old = sample_Tensor['log_pi_old']
        IDs = sample_Tensor['agent_ids']

        # prepare critic inputs
        if self.use_parameter_sharing:
            key = self.model_keys[0]
            bs = batch_size * self.n_agents
            if self.use_global_state:
                critic_input = {key: state.reshape(batch_size, 1, -1).broadcast_to(
                    (batch_size, self.n_agents, -1)).reshape(bs, -1)}
            else:
                critic_input = {key: obs[key].reshape(batch_size, 1, -1).broadcast_to(
                    (batch_size, self.n_agents, -1)).reshape(bs, -1)}
        else:
            bs = batch_size
            if self.use_global_state:
                critic_input = {k: state.reshape(batch_size, -1) for k in self.agent_keys}
            else:
                joint_obs = ops.cat(itemgetter(*self.agent_keys)(obs), axis=-1)
                critic_input = {k: joint_obs for k in self.agent_keys}

        (loss, loss_a, loss_c, loss_e), grads = self.grad_fn(bs, obs, actions, avail_actions, log_pi_old,
                                                             values, returns, advantages, agent_mask, IDs, critic_input)
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

        return info
