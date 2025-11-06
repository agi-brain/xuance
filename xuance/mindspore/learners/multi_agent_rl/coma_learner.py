"""
COMA: Counterfactual Multi-Agent Policy Gradients
Paper link: https://ojs.aaai.org/index.php/AAAI/article/view/11794
Implementation: MindSpore
"""
import numpy as np
from argparse import Namespace
from operator import itemgetter
from xuance.common import List, Optional
from xuance.mindspore import ms, msd, Tensor, Module, ops, nn, optim
from xuance.mindspore.utils import clip_grads
from xuance.mindspore.learners import LearnerMAS  # Different from torch


class COMA_Learner(LearnerMAS):
    def __init__(self,
                 config: Namespace,
                 model_keys: List[str],
                 agent_keys: List[str],
                 policy: Module,
                 callback):
        config.use_value_clip, config.value_clip_range = False, None
        config.use_huber_loss, config.huber_delta = False, None
        config.use_value_norm = False
        config.vf_coef, config.ent_coef = None, None
        super(COMA_Learner, self).__init__(config, model_keys, agent_keys, policy, callback)
        self.build_optimizer()
        self.use_value_clip, self.value_clip_range = config.use_value_clip, config.value_clip_range
        self.use_huber_loss, self.huber_delta = config.use_huber_loss, config.huber_delta
        self.use_value_norm = config.use_value_norm
        self.vf_coef, self.ent_coef = config.vf_coef, config.ent_coef
        self.mse_loss = nn.MSELoss()
        self.huber_loss = nn.HuberLoss(reduction="none", delta=self.huber_delta)

        self.sync_frequency = config.sync_frequency
        self.n_actions = {k: int(self.policy.action_space[k].n) for k in self.model_keys}
        self.mse_loss = nn.MSELoss()

        self.softmax = nn.Softmax(axis=-1)
        self.is_continuous = self.policy.is_continuous
        if self.is_continuous:
            self.pi_dist = {k: msd.Normal(dtype=ms.float32) for k in self.model_keys}
        else:
            self.pi_dist = {k: msd.Categorical() for k in self.model_keys}

        # Get gradient function
        self.actor_grad_fn = ms.value_and_grad(self.actor_forward_fn, None,
                                               self.optimizer['actor'].parameters, has_aux=True)
        self.critic_grad_fn = ms.value_and_grad(self.critic_forward_fn, None,
                                                self.optimizer['critic'].parameters, has_aux=True)
        self.policy.set_train()

    def build_optimizer(self):
        self.optimizer = {
            'actor': optim.Adam(params=self.policy.parameters_actor, lr=self.config.learning_rate_actor, eps=1e-5),
            'critic': optim.Adam(params=self.policy.parameters_critic, lr=self.config.learning_rate_critic, eps=1e-5)
        }
        self.scheduler = {
            'actor': optim.lr_scheduler.LinearLR(self.optimizer['actor'],
                                                 start_factor=1.0,
                                                 end_factor=self.end_factor_lr_decay,
                                                 total_iters=self.config.running_steps),
            'critic': optim.lr_scheduler.LinearLR(self.optimizer['critic'],
                                                  start_factor=1.0,
                                                  end_factor=self.end_factor_lr_decay,
                                                  total_iters=self.config.running_steps)
        }

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

    def actor_forward_fn(self, bs, obs, actions, values_pred_dict, agent_mask, avail_actions, IDs, epsilon):
        # feedforward
        _, pi_logits = self.policy(observation=obs, agent_ids=IDs, avail_actions=avail_actions, epsilon=epsilon)

        # calculate loss
        loss_a, advantages_list = [], []
        for key in self.model_keys:
            mask_values = agent_mask[key]
            mask_values_sum = ops.reduce_sum(mask_values)

            pi_probs = self.softmax(pi_logits[key])
            pi_probs = (1 - epsilon) * pi_probs + epsilon * 1 / self.n_actions[key]
            baseline = ops.reduce_sum(pi_probs * values_pred_dict[key], axis=-1).reshape(bs)
            pi_taken = pi_probs.gather(-1, actions[key].unsqueeze(-1).long())
            q_taken = values_pred_dict[key].gather(-1, actions[key].unsqueeze(-1).long()).reshape(bs)
            log_pi_taken = ops.log(pi_taken).reshape(bs)
            advantages = ops.stop_gradient(q_taken - baseline)
            loss_a.append(-(advantages * log_pi_taken * mask_values).sum() / mask_values_sum)

            # info.update(self.callback.on_update_agent_wise(self.iterations, key, info=info, method="update",
            #                                                mask_values=mask_values, pi_probs=pi_probs,
            #                                                baseline=baseline, pi_taken=pi_taken,
            #                                                q_taken=q_taken, log_pi_taken=log_pi_taken,
            #                                                advantages=advantages, loss_a=loss_a,
            #                                                td_error=td_error))

            advantages_list.append(advantages)

        loss_coma = sum(loss_a)
        advantages_mean = sum(advantages_list) / len(advantages_list)
        return loss_coma, advantages_mean

    def critic_forward_fn(self, bs, batch_size, obs, state, actions, agent_mask, returns, IDs):
        # forward
        if self.use_parameter_sharing:
            key = self.model_keys[0]
            actions_onehot = {key: ops.one_hot(actions[key].long(), depth=self.n_actions[key])}
        else:
            IDs = ops.repeat_elements(ops.eye(self.n_agents).unsqueeze(0), rep=batch_size, axis=0).reshape(bs, -1)
            actions_onehot = {k: ops.one_hot(actions[k].long(), depth=self.n_actions[k]) for k in self.agent_keys}

        _, values_pred = self.policy.get_values(state=state, observation=obs, actions=actions_onehot,
                                                agent_ids=IDs, target=False)

        if self.use_parameter_sharing:
            values_pred_dict = {k: values_pred.reshape(bs, -1) for k in self.model_keys}
        else:
            values_pred_dict = {k: values_pred[:, i] for i, k in enumerate(self.model_keys)}

        # calculate loss
        loss_c = []
        for key in self.model_keys:
            mask_values = agent_mask[key]
            q_taken = values_pred_dict[key].gather(-1, actions[key].unsqueeze(-1).long()).reshape(bs)
            td_error = (q_taken - ops.stop_gradient(returns[key])) * mask_values
            loss_c.append((td_error ** 2).sum() / mask_values.sum())

        loss_critic = sum(loss_c)
        return loss_critic, values_pred_dict

    def update(self, sample, epsilon=0.0):
        self.iterations += 1

        # prepare training data
        sample_Tensor = self.build_training_data(sample=sample,
                                                 use_parameter_sharing=self.use_parameter_sharing,
                                                 use_actions_mask=self.use_actions_mask,
                                                 use_global_state=True)
        batch_size = sample_Tensor['batch_size']
        state = sample_Tensor['state']
        obs = sample_Tensor['obs']
        actions = sample_Tensor['actions']
        agent_mask = sample_Tensor['agent_mask']
        avail_actions = sample_Tensor['avail_actions']
        returns = sample_Tensor['returns']
        IDs = sample_Tensor['agent_ids']

        bs = batch_size * self.n_agents if self.use_parameter_sharing else batch_size

        info = self.callback.on_update_start(self.iterations, method="update",
                                             policy=self.policy, sample_Tensor=sample_Tensor, bs=bs)

        # update critic
        (loss_critic, values_pred_dict), grad_critic = self.critic_grad_fn(bs, batch_size, obs, state, actions,
                                                                           agent_mask, returns, IDs)
        if self.use_grad_clip:
            grad_critic = clip_grads(grad_critic, Tensor(-self.grad_clip_norm), Tensor(self.grad_clip_norm))
        self.optimizer['critic'](grad_critic)
        self.scheduler['critic'].step()  # update learning rate

        # update actor
        (loss_coma, advantages_mean), grad_actor = self.actor_grad_fn(bs, obs, actions, values_pred_dict,
                                                                      agent_mask, avail_actions, IDs, epsilon)
        if self.use_grad_clip:
            grad_actor = clip_grads(grad_actor, Tensor(-self.grad_clip_norm), Tensor(self.grad_clip_norm))
        self.optimizer['actor'](grad_actor)
        self.scheduler['actor'].step()  # update learning rate

        # Logger
        learning_rate_actor = self.scheduler['actor'].get_last_lr()[0]
        learning_rate_critic = self.scheduler['critic'].get_last_lr()[0]

        info.update({
            "learning_rate_actor": learning_rate_actor.asnumpy(),
            "learning_rate_critic": learning_rate_critic.asnumpy(),
            "actor_loss": loss_coma.asnumpy(),
            "critic_loss": loss_critic.asnumpy(),
            "advantage": advantages_mean.mean().asnumpy(),
        })

        info.update(self.callback.on_update_end(self.iterations, method="update", policy=self.policy, info=info))

        return info
