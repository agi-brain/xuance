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


class IPPO_Learner(LearnerMAS):
    def __init__(self,
                 config: Namespace,
                 model_keys: List[str],
                 agent_keys: List[str],
                 policy: Module):
        super(IPPO_Learner, self).__init__(config, model_keys, agent_keys, policy)
        self.optimizer = optim.Adam(params=self.policy.trainable_params(), lr=config.learning_rate, eps=1e-5,
                                    weight_decay=config.weight_decay)
        self.scheduler = optim.lr_scheduler.LinearLR(self.optimizer, start_factor=1.0, end_factor=0.5,
                                                     total_iters=self.config.running_steps)
        self.lr = config.learning_rate
        self.end_factor_lr_decay = config.end_factor_lr_decay
        self.gamma = config.gamma
        self.clip_range = config.clip_range
        self.use_linear_lr_decay = config.use_linear_lr_decay
        self.use_value_clip, self.value_clip_range = config.use_value_clip, config.value_clip_range
        self.use_huber_loss, self.huber_delta = config.use_huber_loss, config.huber_delta
        self.use_value_norm = config.use_value_norm
        self.use_global_state = config.use_global_state
        self.vf_coef, self.ent_coef = config.vf_coef, config.ent_coef
        self.mse_loss = MSELoss()
        self.huber_loss = HuberLoss(reduction="none", delta=self.huber_delta)
        if self.use_value_norm:
            self.value_normalizer = {key: ValueNorm(1) for key in self.model_keys}
        else:
            self.value_normalizer = None
        # Get gradient function
        self.grad_fn = ms.value_and_grad(self.forward_fn, None, self.optimizer.parameters, has_aux=True)
        self.policy.set_train()

    def forward_fn(self, *args):
        bs, obs, actions, avail_actions, log_pi_old, values, returns, advantages, agt_mask, ids = args
        # feedforward
        _, pi_dists_dict = self.policy(observation=obs, agent_ids=ids, avail_actions=avail_actions)
        _, value_pred_dict = self.policy.get_values(observation=obs, agent_ids=ids)

        # calculate losses for each agent
        loss_a, loss_e, loss_c = [], [], []
        info = {}
        for key in self.model_keys:
            mask_values = agt_mask[key]
            # actor loss
            log_pi = pi_dists_dict[key].log_prob(actions[key]).reshape(bs)
            ratio = ops.exp(log_pi - log_pi_old[key]).reshape(bs)
            advantages_mask = ops.stop_gradient(advantages[key]) * mask_values
            surrogate1 = ratio * advantages_mask
            surrogate2 = ops.clip_by_value(ratio, Tensor(1 - self.clip_range), Tensor(1 + self.clip_range))
            surrogate2 *= advantages_mask
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

            info.update({
                f"{key}/actor_loss": loss_a[-1].asnumpy(),
                f"{key}/critic_loss": loss_c[-1].asnumpy(),
                f"{key}/entropy": loss_e[-1].asnumpy(),
                f"{key}/predict_value": value_pred_i.mean().asnumpy()
            })

        loss = sum(loss_a) + self.vf_coef * sum(loss_c) - self.ent_coef * sum(loss_e)
        return loss, sum(loss_a), sum(loss_c), sum(loss_e)

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
            obs_tensor = Tensor(np.stack(itemgetter(*self.agent_keys)(sample['obs']), axis=1))
            actions_tensor = Tensor(np.stack(itemgetter(*self.agent_keys)(sample['actions']), axis=1))
            values_tensor = Tensor(np.stack(itemgetter(*self.agent_keys)(sample['values']), axis=1))
            returns_tensor = Tensor(np.stack(itemgetter(*self.agent_keys)(sample['returns']), axis=1))
            advantages_tensor = Tensor(np.stack(itemgetter(*self.agent_keys)(sample['advantages']), 1))
            log_pi_old_tensor = Tensor(np.stack(itemgetter(*self.agent_keys)(sample['log_pi_old']), 1))
            ter_tensor = Tensor(np.stack(itemgetter(*self.agent_keys)(sample['terminals']), 1)).float()
            msk_tensor = Tensor(np.stack(itemgetter(*self.agent_keys)(sample['agent_mask']), 1)).float()
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
                IDs = self.eye(self.n_agents, self.n_agents, ms.float32).unsqueeze(1).unsqueeze(0).broadcast_to(
                    (batch_size, -1, seq_length, -1)).reshape(bs, seq_length, self.n_agents)
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
                IDs = self.eye(self.n_agents, self.n_agents, ms.float32).unsqueeze(0).broadcast_to(
                    (batch_size, -1, -1)).reshape(bs, self.n_agents)

            if use_actions_mask:
                avail_a = np.stack(itemgetter(*self.agent_keys)(sample['avail_actions']), axis=1)
                if self.use_rnn:
                    avail_actions = {k: Tensor(avail_a.reshape([bs, seq_length, -1])).float()}
                else:
                    avail_actions = {k: Tensor(avail_a.reshape([bs, -1])).float()}

        else:
            obs = {k: Tensor(sample['obs'][k]) for k in self.agent_keys}
            actions = {k: Tensor(sample['actions'][k]) for k in self.agent_keys}
            values = {k: Tensor(sample['values'][k]) for k in self.agent_keys}
            returns = {k: Tensor(sample['returns'][k]) for k in self.agent_keys}
            advantages = {k: Tensor(sample['advantages'][k]) for k in self.agent_keys}
            log_pi_old = {k: Tensor(sample['log_pi_old'][k]) for k in self.agent_keys}
            terminals = {k: Tensor(sample['terminals'][k]).float() for k in self.agent_keys}
            agent_mask = {k: Tensor(sample['agent_mask'][k]).float() for k in self.agent_keys}
            if use_actions_mask:
                avail_actions = {k: Tensor(sample['avail_actions'][k]).float() for k in self.agent_keys}

        if use_global_state:
            state = Tensor(sample['state'])

        if self.use_rnn:
            filled = Tensor(sample['filled']).float()

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
        info = {}

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
        log_pi_old = sample_Tensor['log_pi_old']
        IDs = sample_Tensor['agent_ids']

        bs = batch_size * self.n_agents if self.use_parameter_sharing else batch_size

        (loss, loss_a, loss_c, loss_e), grads = self.grad_fn(bs, obs, actions, avail_actions, log_pi_old,
                                                             values, returns, advantages, agent_mask, IDs)
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
