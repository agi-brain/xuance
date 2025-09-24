"""
Independent Advantage Actor Critic (IAC)
Paper link: https://ojs.aaai.org/index.php/AAAI/article/view/11794
Implementation: Pytorch
"""
import numpy as np
from argparse import Namespace
from operator import itemgetter
from mindspore.nn import MSELoss, HuberLoss
from xuance.common import Optional, List
from xuance.mindspore import ms, nn, msd, ops, Module, Tensor, optim
from xuance.mindspore.utils import ValueNorm, clip_grads
from xuance.mindspore.learners import LearnerMAS


class IAC_Learner(LearnerMAS):
    def __init__(self,
                 config: Namespace,
                 model_keys: List[str],
                 agent_keys: List[str],
                 policy: Module):
        super(IAC_Learner, self).__init__(config, model_keys, agent_keys, policy)
        self.build_optimizer()
        self.use_value_clip, self.value_clip_range = config.use_value_clip, config.value_clip_range
        self.use_huber_loss, self.huber_delta = config.use_huber_loss, config.huber_delta
        self.use_value_norm = config.use_value_norm
        self.vf_coef, self.ent_coef = config.vf_coef, config.ent_coef
        self.mse_loss = MSELoss()
        self.huber_loss = HuberLoss(reduction="none", delta=self.huber_delta)
        self.softmax = nn.Softmax(axis=-1)
        self.is_continuous = self.policy.is_continuous
        if self.use_value_norm:
            self.value_normalizer = {key: ValueNorm(1).to(self.device) for key in self.model_keys}
        else:
            self.value_normalizer = None

        if self.is_continuous:
            self.pi_dist = {k: msd.Normal(dtype=ms.float32) for k in self.model_keys}
        else:
            self.pi_dist = {k: msd.Categorical() for k in self.model_keys}

        # Get gradient function
        self.grad_fn = ms.value_and_grad(self.forward_fn, None, self.optimizer.parameters, has_aux=True)
        self.policy.set_train()

    def build_optimizer(self):
        self.optimizer = optim.Adam(params=self.policy.parameters_model, lr=self.config.learning_rate, eps=1e-5)
        self.scheduler = optim.lr_scheduler.LinearLR(self.optimizer,
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
                IDs = ops.eye(self.n_agents).unsqueeze(1).unsqueeze(0).expand(
                    batch_size, -1, seq_length, -1).reshape(bs, seq_length, self.n_agents)
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
                IDs = Tensor(np.eye(self.n_agents, dtype=np.float32)[None].repeat(batch_size, axis=0).reshape(bs, -1))

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

    def forward_fn(self, *args):
        bs, obs, actions, agent_mask, avail_actions, values, returns, advantages, IDs = args
        value_pred = {}
        pi_dist_mu, pi_dist_std, pi_dist_logits = {}, {}, {}

        # feedforward
        if self.is_continuous:
            _, pi_dist_mu, pi_dist_std = self.policy(observation=obs, agent_ids=IDs, avail_actions=avail_actions)
        else:
            _, pi_dist_logits = self.policy(observation=obs, agent_ids=IDs, avail_actions=avail_actions)

        _, values_pred_dict = self.policy.get_values(observation=obs, agent_ids=IDs)

        loss_a, loss_e, loss_c = [], [], []
        for key in self.model_keys:
            mask_values = agent_mask[key]
            # policy gradient loss
            if self.is_continuous:
                log_pi = self.pi_dist[key]._log_prob(value=actions[key], mean=pi_dist_mu[key], sd=pi_dist_std[key])
                log_pi = ops.reduce_sum(x=log_pi, axis=-1)
                entropy = self.pi_dist[key]._entropy(mean=pi_dist_mu[key], sd=pi_dist_std[key])
                entropy = ops.reduce_sum(x=entropy, axis=-1)
            else:
                probs = self.softmax(pi_dist_logits[key])
                log_pi = self.pi_dist[key]._log_prob(value=actions[key], probs=probs)
                entropy = self.pi_dist[key].entropy(probs=probs)

            pg_loss = -(ops.stop_gradient(advantages[key]) * log_pi * mask_values).sum() / mask_values.sum()
            loss_a.append(pg_loss)

            # entropy loss
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

        # Total loss
        loss = sum(loss_a) + self.vf_coef * sum(loss_c) - self.ent_coef * sum(loss_e)

        return loss, loss_a, loss_e, loss_c, value_pred

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
        IDs = sample_Tensor['agent_ids']

        bs = batch_size * self.n_agents if self.use_parameter_sharing else batch_size

        # feedforward
        (loss, loss_a, loss_e, loss_c, value_pred), grads = self.grad_fn(bs, obs, actions, agent_mask, avail_actions,
                                                                         values, returns, advantages, IDs)
        if self.use_grad_clip:
            grads = clip_grads(grads, Tensor(-self.grad_clip_norm), Tensor(self.grad_clip_norm))
        self.optimizer(grads)  # backpropagation

        self.scheduler.step()  # update learning rate
        lr = self.scheduler.get_last_lr()[0]

        info.update({
            "learning_rate": lr.asnumpy(),
            "pg_loss": sum(loss_a).asnumpy(),
            "vf_loss": sum(loss_c).asnumpy(),
            "entropy_loss": sum(loss_e).asnumpy(),
            "loss": loss.asnumpy(),
        })
        info.update(value_pred)

        return info
