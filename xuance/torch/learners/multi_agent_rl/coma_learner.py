"""
COMA: Counterfactual Multi-Agent Policy Gradients
Paper link: https://ojs.aaai.org/index.php/AAAI/article/view/11794
Implementation: Pytorch
"""
import torch
from torch import nn
from torch.nn.functional import one_hot
from xuance.common import List
from argparse import Namespace
from xuance.torch.learners.multi_agent_rl.iac_learner import IAC_Learner


class COMA_Learner(IAC_Learner):
    def __init__(self,
                 config: Namespace,
                 model_keys: List[str],
                 agent_keys: List[str],
                 policy: nn.Module,
                 callback):
        config.use_value_clip, config.value_clip_range = False, None
        config.use_huber_loss, config.huber_delta = False, None
        config.use_value_norm = False
        config.vf_coef, config.ent_coef = None, None
        super(COMA_Learner, self).__init__(config, model_keys, agent_keys, policy, callback)
        self.sync_frequency = config.sync_frequency
        self.n_actions = {k: self.policy.action_space[k].n for k in self.model_keys}
        self.mse_loss = nn.MSELoss()

    def build_optimizer(self):
        self.optimizer = {
            'actor': torch.optim.Adam(self.policy.parameters_actor, self.config.learning_rate_actor, eps=1e-5),
            'critic': torch.optim.Adam(self.policy.parameters_critic, self.config.learning_rate_critic, eps=1e-5)
        }
        self.scheduler = {
            'actor': torch.optim.lr_scheduler.LinearLR(self.optimizer['actor'],
                                                       start_factor=1.0,
                                                       end_factor=self.end_factor_lr_decay,
                                                       total_iters=self.config.running_steps),
            'critic': torch.optim.lr_scheduler.LinearLR(self.optimizer['critic'],
                                                        start_factor=1.0,
                                                        end_factor=self.end_factor_lr_decay,
                                                        total_iters=self.config.running_steps)
        }

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

        # feedforward
        _, pi_probs = self.policy(observation=obs, agent_ids=IDs, avail_actions=avail_actions, epsilon=epsilon)

        if self.use_parameter_sharing:
            key = self.model_keys[0]
            actions_onehot = {key: one_hot(actions[key].long(), self.n_actions[key])}
        else:
            IDs = torch.eye(self.n_agents).unsqueeze(0).repeat(batch_size, 1, 1).reshape(bs, -1).to(self.device)
            actions_onehot = {k: one_hot(actions[k].long(), self.n_actions[k]) for k in self.agent_keys}

        _, values_pred = self.policy.get_values(state=state, observation=obs, actions=actions_onehot,
                                                agent_ids=IDs, target=False)

        if self.use_parameter_sharing:
            values_pred_dict = {k: values_pred.reshape(bs, -1) for k in self.model_keys}
        else:
            values_pred_dict = {k: values_pred[:, i] for i, k in enumerate(self.model_keys)}

        # calculate loss
        loss_a, loss_c = [], []
        for key in self.model_keys:
            mask_values = agent_mask[key]

            if self.use_actions_mask:
                pi_probs[key][avail_actions[key] == 0] = 0.0  # mask out the unavailable actions.
                pi_probs[key] = pi_probs[key] / pi_probs[key].sum(dim=-1, keepdim=True)  # re-normalize the actions.
                pi_probs[key][avail_actions[key] == 0] = 0.0
            baseline = (pi_probs[key] * values_pred_dict[key]).sum(-1).reshape(bs)
            pi_taken = pi_probs[key].gather(-1, actions[key].unsqueeze(-1).long())
            q_taken = values_pred_dict[key].gather(-1, actions[key].unsqueeze(-1).long()).reshape(bs)
            log_pi_taken = torch.log(pi_taken).reshape(bs)
            advantages = (q_taken - baseline).detach()
            loss_a.append(-(advantages * log_pi_taken * mask_values).sum() / mask_values.sum())

            td_error = (q_taken - returns[key].detach()) * mask_values
            loss_c.append((td_error ** 2).sum() / mask_values.sum())

            info.update(self.callback.on_update_agent_wise(self.iterations, key, info=info, method="update",
                                                           mask_values=mask_values, pi_probs=pi_probs,
                                                           baseline=baseline, pi_taken=pi_taken,
                                                           q_taken=q_taken, log_pi_taken=log_pi_taken,
                                                           advantages=advantages, loss_a=loss_a,
                                                           td_error=td_error))

        # update critic
        loss_critic = sum(loss_c)
        self.optimizer['critic'].zero_grad()
        loss_critic.backward()
        if self.use_grad_clip:
            grad_norm = torch.nn.utils.clip_grad_norm_(self.policy.parameters_critic, self.grad_clip_norm)
            info["gradient_norm_actor"] = grad_norm.item()
        self.optimizer['critic'].step()
        if self.scheduler['critic'] is not None:
            self.scheduler['critic'].step()
        if self.iterations % self.sync_frequency == 0:
            self.policy.copy_target()

        # update actor(s)
        loss_coma = sum(loss_a)
        self.optimizer['actor'].zero_grad()
        loss_coma.backward()
        if self.use_grad_clip:
            grad_norm = torch.nn.utils.clip_grad_norm_(self.policy.parameters_actor, self.grad_clip_norm)
            info["gradient_norm_actor"] = grad_norm.item()
        self.optimizer['actor'].step()
        if self.scheduler['actor'] is not None:
            self.scheduler['actor'].step()

        # Logger
        learning_rate_actor = self.optimizer['actor'].state_dict()['param_groups'][0]['lr']
        learning_rate_critic = self.optimizer['critic'].state_dict()['param_groups'][0]['lr']

        info.update({
            "learning_rate_actor": learning_rate_actor,
            "learning_rate_critic": learning_rate_critic,
            "actor_loss": loss_coma.item(),
            "critic_loss": loss_critic.item(),
            "advantage": advantages.mean().item(),
        })

        info.update(self.callback.on_update_end(self.iterations, method="update", policy=self.policy, info=info))

        return info

    def update_rnn(self, sample, epsilon=0.0):
        self.iterations += 1

        sample_Tensor = self.build_training_data(sample=sample,
                                                 use_parameter_sharing=self.use_parameter_sharing,
                                                 use_actions_mask=self.use_actions_mask,
                                                 use_global_state=True)
        batch_size = sample_Tensor['batch_size']
        state = sample_Tensor['state']
        bs_rnn = batch_size * self.n_agents if self.use_parameter_sharing else batch_size
        obs = sample_Tensor['obs']
        actions = sample_Tensor['actions']
        returns = sample_Tensor['returns']
        avail_actions = sample_Tensor['avail_actions']
        agent_mask = sample_Tensor['agent_mask']
        filled = sample_Tensor['filled']
        seq_len = filled.shape[1]
        IDs = sample_Tensor['agent_ids']

        if self.use_parameter_sharing:
            filled = filled.unsqueeze(1).expand(batch_size, self.n_agents, seq_len).reshape(bs_rnn, seq_len)
        else:
            IDs = torch.eye(self.n_agents).unsqueeze(0).unsqueeze(0).repeat(batch_size, seq_len, 1, 1).to(self.device)

        info = self.callback.on_update_start(self.iterations, method="update_rnn",
                                             policy=self.policy, sample_Tensor=sample_Tensor,
                                             bs_rnn=bs_rnn, filled=filled, IDs=IDs)

        rnn_hidden_actor = {k: self.policy.actor_representation[k].init_hidden(bs_rnn) for k in self.model_keys}
        rnn_hidden_critic = {k: self.policy.critic_representation[k].init_hidden(bs_rnn) for k in self.model_keys}

        # feedforward
        _, pi_probs = self.policy(observation=obs, agent_ids=IDs, avail_actions=avail_actions,
                                  rnn_hidden=rnn_hidden_actor, epsilon=epsilon)
        actions_onehot = {k: one_hot(actions[k].long(), self.n_actions[k]) for k in self.model_keys}
        _, values_pred = self.policy.get_values(state=state, observation=obs, actions=actions_onehot,
                                                agent_ids=IDs, rnn_hidden=rnn_hidden_critic, target=False)

        if self.use_parameter_sharing:
            values_pred_dict = {self.model_keys[0]: values_pred.transpose(1, 2).reshape(bs_rnn, seq_len, -1)}
        else:
            values_pred_dict = {k: values_pred[:, :, i] for i, k in enumerate(self.model_keys)}

        # calculate loss
        loss_a, loss_c = [], []
        for key in self.model_keys:
            mask_values = agent_mask[key] * filled

            if self.use_actions_mask:
                pi_probs[key][avail_actions[key] == 0] = 0.0  # mask out the unavailable actions.
                pi_probs[key] = pi_probs[key] / pi_probs[key].sum(dim=-1, keepdim=True)  # re-normalize the actions.
                pi_probs[key][avail_actions[key] == 0] = 0.0
            baseline = (pi_probs[key] * values_pred_dict[key]).sum(-1).reshape(bs_rnn, seq_len)
            pi_taken = pi_probs[key].gather(-1, actions[key].unsqueeze(-1).long())
            q_taken = values_pred_dict[key].gather(-1, actions[key].unsqueeze(-1).long()).reshape(bs_rnn, seq_len)
            log_pi_taken = torch.log(pi_taken).reshape(bs_rnn, seq_len)
            advantages = (q_taken - baseline).detach()
            loss_a.append(-(advantages * log_pi_taken * mask_values).sum() / mask_values.sum())

            td_error = (q_taken - returns[key].detach()) * mask_values
            loss_c.append((td_error ** 2).sum() / mask_values.sum())

            info.update(self.callback.on_update_agent_wise(self.iterations, key, info=info, method="update_rnn",
                                                           mask_values=mask_values, pi_probs=pi_probs,
                                                           baseline=baseline, pi_taken=pi_taken,
                                                           q_taken=q_taken, log_pi_taken=log_pi_taken,
                                                           advantages=advantages, loss_a=loss_a,
                                                           td_error=td_error))

        # update critic
        loss_critic = sum(loss_c)
        self.optimizer['critic'].zero_grad()
        loss_critic.backward()
        if self.use_grad_clip:
            grad_norm = torch.nn.utils.clip_grad_norm_(self.policy.parameters_critic, self.grad_clip_norm)
            info["gradient_norm_actor"] = grad_norm.item()
        self.optimizer['critic'].step()
        if self.scheduler['critic'] is not None:
            self.scheduler['critic'].step()
        if self.iterations % self.sync_frequency == 0:
            self.policy.copy_target()

        # update actor(s)
        loss_coma = sum(loss_a)
        self.optimizer['actor'].zero_grad()
        loss_coma.backward()
        if self.use_grad_clip:
            grad_norm = torch.nn.utils.clip_grad_norm_(self.policy.parameters_actor, self.grad_clip_norm)
            info["gradient_norm_actor"] = grad_norm.item()
        self.optimizer['actor'].step()
        if self.scheduler['actor'] is not None:
            self.scheduler['actor'].step()

        # Logger
        learning_rate_actor = self.optimizer['actor'].state_dict()['param_groups'][0]['lr']
        learning_rate_critic = self.optimizer['critic'].state_dict()['param_groups'][0]['lr']

        info.update({
            "learning_rate_actor": learning_rate_actor,
            "learning_rate_critic": learning_rate_critic,
            "actor_loss": loss_coma.item(),
            "critic_loss": loss_critic.item(),
            "advantage": advantages.mean().item(),
        })

        info.update(self.callback.on_update_end(self.iterations, method="update_rnn", policy=self.policy, info=info))

        return info
