"""
Value Decomposition Actor-Critic (VDAC)
Paper link: https://ojs.aaai.org/index.php/AAAI/article/view/17353
Implementation: MindSpore
"""
from argparse import Namespace
from xuance.common import List
from xuance.mindspore import ops, Module, Tensor
from xuance.mindspore.utils import clip_grads
from xuance.mindspore.learners.multi_agent_rl.iac_learner import IAC_Learner


class VDAC_Learner(IAC_Learner):
    def __init__(self,
                 config: Namespace,
                 model_keys: List[str],
                 agent_keys: List[str],
                 policy: Module):
        super(VDAC_Learner, self).__init__(config, model_keys, agent_keys, policy)
        self.use_global_state = True if config.mixer == "QMIX" else getattr(config, "use_global_state", False)

    def forward_fn(self, *args):
        bs, batch_size, state, obs, actions, agent_mask, avail_actions, values, returns, advantages, IDs = args
        value_pred = {}
        pi_dist_mu, pi_dist_std, pi_dist_logits = {}, {}, {}

        # feedforward
        if self.is_continuous:
            _, pi_dist_mu, pi_dist_std = self.policy(observation=obs, agent_ids=IDs, avail_actions=avail_actions)
        else:
            _, pi_dist_logits = self.policy(observation=obs, agent_ids=IDs, avail_actions=avail_actions)

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
            values_tot = ops.repeat_elements(values_tot.reshape(batch_size, 1), rep=self.n_agents, axis=1).reshape(bs)
        values_pred_dict = {k: values_tot for k in self.model_keys}

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
                value_target = ops.stop_gradient(value_target)
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
                value_target = ops.stop_gradient(value_target)
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

        return loss, sum(loss_a), sum(loss_e), sum(loss_c), value_pred

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
        IDs = sample_Tensor['agent_ids']

        bs = batch_size * self.n_agents if self.use_parameter_sharing else batch_size

        # feedforward
        (loss, loss_a, loss_e, loss_c, value_pred), grads = self.grad_fn(bs, batch_size, state, obs, actions,
                                                                         agent_mask, avail_actions, values, returns,
                                                                         advantages, IDs)
        if self.use_grad_clip:
            grads = clip_grads(grads, Tensor(-self.grad_clip_norm), Tensor(self.grad_clip_norm))
        self.optimizer(grads)  # backpropagation

        self.scheduler.step()  # update learning rate
        lr = self.scheduler.get_last_lr()[0]

        info.update({
            "learning_rate": lr.asnumpy(),
            "pg_loss": loss_a.asnumpy(),
            "vf_loss": loss_c.asnumpy(),
            "entropy_loss": loss_e.asnumpy(),
            "loss": loss.asnumpy(),
        })
        info.update(value_pred)

        return info
