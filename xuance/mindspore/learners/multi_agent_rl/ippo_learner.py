"""
Multi-Agent Proximal Policy Optimization (MAPPO)
Paper link:
https://arxiv.org/pdf/2103.01955.pdf
Implementation: MindSpore
"""
from argparse import Namespace
from xuance.common import List
from xuance.mindspore import ms, nn, msd, ops, Module, Tensor
from xuance.mindspore.utils import ValueNorm, clip_grads
from xuance.mindspore.learners.multi_agent_rl.iac_learner import IAC_Learner


class IPPO_Learner(IAC_Learner):
    def __init__(self,
                 config: Namespace,
                 model_keys: List[str],
                 agent_keys: List[str],
                 policy: Module,
                 callback):
        super(IPPO_Learner, self).__init__(config, model_keys, agent_keys, policy, callback)
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
        self.mse_loss = nn.MSELoss()
        self.huber_loss = nn.HuberLoss(reduction="none", delta=self.huber_delta)
        self.softmax = nn.Softmax(axis=-1)
        self.is_continuous = self.policy.is_continuous
        if self.use_value_norm:
            self.value_normalizer = {key: ValueNorm(1) for key in self.model_keys}
        else:
            self.value_normalizer = None

        if self.is_continuous:
            self.pi_dist = {k: msd.Normal(dtype=ms.float32) for k in self.model_keys}
        else:
            self.pi_dist = {k: msd.Categorical() for k in self.model_keys}

        # Get gradient function
        self.grad_fn = ms.value_and_grad(self.forward_fn, None, self.optimizer.parameters, has_aux=True)
        self.policy.set_train()

    def forward_fn(self, *args):
        bs, obs, actions, avail_actions, log_pi_old, values, returns, advantages, agt_mask, ids = args
        value_pred = {}
        pi_dist_mu, pi_dist_std, pi_dist_logits = {}, {}, {}

        # feedforward
        if self.is_continuous:
            _, pi_dist_mu, pi_dist_std = self.policy(observation=obs, agent_ids=ids, avail_actions=avail_actions)
        else:
            _, pi_dist_logits = self.policy(observation=obs, agent_ids=ids, avail_actions=avail_actions)

        _, value_pred_dict = self.policy.get_values(observation=obs, agent_ids=ids)

        # calculate losses for each agent
        loss_a, loss_e, loss_c = [], [], []
        for key in self.model_keys:
            mask_values = agt_mask[key]
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

        (loss, loss_a, loss_c, loss_e, value_pred), grads = self.grad_fn(bs, obs, actions, avail_actions, log_pi_old,
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
        info.update(value_pred)

        return info
