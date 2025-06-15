"""
MFAC: Mean Field Actor-Critic
Paper link:
http://proceedings.mlr.press/v80/yang18d/yang18d.pdf
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
from xuance.torch.learners.multi_agent_rl.iac_learner import IAC_Learner


class MFAC_Learner(IAC_Learner):
    def __init__(self,
                 config: Namespace,
                 model_keys: List[str],
                 agent_keys: List[str],
                 policy: nn.Module,
                 callback):
        super(MFAC_Learner, self).__init__(config, model_keys, agent_keys, policy, callback)

    def build_actions_mean_input(self, sample: Optional[dict], use_parameter_sharing: Optional[bool] = False):
        batch_size = sample['batch_size']
        seq_length = sample['sequence_length'] if self.use_rnn else 1
        if use_parameter_sharing:
            k = self.model_keys[0]
            bs = batch_size * self.n_agents
            if self.n_agents == 1:
                actions_mean_tensor = Tensor(sample['actions_mean'][k]).to(self.device).unsqueeze(1)
            else:
                actions_mean_tensor = Tensor(np.stack(itemgetter(*self.agent_keys)(sample['actions_mean']),
                                                      axis=1)).to(self.device)
            if self.use_rnn:
                actions_mean = {k: actions_mean_tensor.reshape(bs, seq_length + 1, -1)}
            else:
                actions_mean = {k: actions_mean_tensor.reshape(bs, -1)}
        else:
            actions_mean = {k: Tensor(sample['actions_mean'][k]).to(self.device) for k in self.agent_keys}

        return actions_mean

    def update(self, sample):
        self.iterations += 1

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
        IDs = sample_Tensor['agent_ids']

        bs = batch_size * self.n_agents if self.use_parameter_sharing else batch_size

        info = self.callback.on_update_start(self.iterations, method="update", actions_mean=act_mean,
                                             policy=self.policy, sample_Tensor=sample_Tensor, bs=bs)

        # feedforward
        _, pi_dist_dict = self.policy(observation=obs, agent_ids=IDs, avail_actions=avail_actions)
        _, values_pred_dict = self.policy.get_values(observation=obs, actions_mean=act_mean, agent_ids=IDs)

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

        info.update(self.callback.on_update_end(self.iterations, method="update", policy=self.policy, info=info))

        return info
