"""
COMA: Counterfactual Multi-Agent Policy Gradients
Paper link: https://ojs.aaai.org/index.php/AAAI/article/view/11794
Implementation: Pytorch
"""
import torch
from argparse import Namespace
from torch import nn
from torch.nn.functional import one_hot
from xuance.common import AgentGrouping
from xuance.torch.learners.multi_agent_rl.iac_learner import IAC_Learner


class COMA_Learner(IAC_Learner):
    def __init__(self,
                 config: Namespace,
                 agent_grouping: AgentGrouping,
                 model: nn.Module,
                 callback):
        config.use_value_clip, config.value_clip_range = False, None
        config.use_huber_loss, config.huber_delta = False, None
        config.use_value_norm = False
        config.vf_coef, config.ent_coef = None, None
        super(COMA_Learner, self).__init__(config, agent_grouping, model, callback)
        self.sync_frequency = config.sync_frequency
        self.n_actions = {k: self.model.critics.action_space[k].n for k in self.agent_keys}
        self.mse_loss = nn.MSELoss()

    def build_optimizer(self):
        self.optimizer = {
            'actor': torch.optim.Adam(self.model.actors.parameters(), self.config.learning_rate_actor, eps=1e-5),
            'critic': torch.optim.Adam(self.model.critics.parameters(), self.config.learning_rate_critic, eps=1e-5)
        }
        self.scheduler = {
            'actor': torch.optim.lr_scheduler.LinearLR(self.optimizer['actor'],
                                                       start_factor=1.0,
                                                       end_factor=self.end_factor_lr_decay,
                                                       total_iters=self.total_iters),
            'critic': torch.optim.lr_scheduler.LinearLR(self.optimizer['critic'],
                                                        start_factor=1.0,
                                                        end_factor=self.end_factor_lr_decay,
                                                        total_iters=self.total_iters)
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
        agent_indices = sample_Tensor['agent_indices']

        if not self.agent_grouping.full_independent:
            obs = self.packed_tensor(obs)
            packed_actions = self.packed_tensor(actions)
            packed_avail_actions = self.packed_tensor(avail_actions)
            returns = self.packed_tensor(returns)
            agent_mask = self.packed_tensor(agent_mask)
        else:
            packed_actions = actions
            packed_avail_actions = avail_actions

        info = self.callback.on_update_start(self.iterations, method="update",
                                             model=self.model, sample_Tensor=sample_Tensor)

        # feedforward
        pi_dist_dict = self.model(observations=obs, agent_indices=agent_indices,
                                  avail_actions=avail_actions,
                                  epsilon=epsilon).distributions

        joint_actions = torch.concat([one_hot(v.long(), self.n_actions[k]) for k, v in actions.items()],
                                     dim=1).reshape([batch_size, -1])

        values_pred = self.model.get_values(states=state, observations=obs, joint_actions=joint_actions,
                                            agent_indices=agent_indices, target=False).values
        if not self.agent_grouping.full_independent:
            values_pred = self.packed_tensor(values_pred)

        # calculate loss
        loss_a, loss_c = [], []
        for group, agent_keys in self.groups.items():
            bs = batch_size * self.n_group_agents[group]
            mask_values = agent_mask[group]
            pi_probs = pi_dist_dict[group].probs
            if self.use_actions_mask:
                pi_probs[packed_avail_actions[group] == 0] = 0.0  # mask out the unavailable actions.
                pi_probs = pi_probs[group] / pi_probs.sum(dim=-1, keepdim=True)  # re-normalize the actions.
                pi_probs[packed_avail_actions[group] == 0] = 0.0
            baseline = (pi_probs * values_pred[group]).sum(-1).reshape(bs)
            pi_taken = pi_probs.gather(-1, packed_actions[group].unsqueeze(-1).long())
            q_taken = values_pred[group].gather(-1, packed_actions[group].unsqueeze(-1).long()).reshape(bs)
            log_pi_taken = torch.log(pi_taken).reshape(bs)
            advantages = (q_taken - baseline).detach()
            loss_a.append(-(advantages * log_pi_taken * mask_values).sum() / mask_values.sum())

            td_error = (q_taken - returns[group].detach()) * mask_values
            loss_c.append((td_error ** 2).sum() / mask_values.sum())

            info.update(self.callback.on_update_agent_wise(self.iterations, group, info=info, method="update",
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
            grad_norm = torch.nn.utils.clip_grad_norm_(self.model.critics.parameters(), self.grad_clip_norm)
            info["gradient_norm_actor"] = grad_norm.item()
        self.optimizer['critic'].step()
        if self.scheduler['critic'] is not None:
            self.scheduler['critic'].step()
        if self.iterations % self.sync_frequency == 0:
            self.model.copy_target()

        # update actor(s)
        loss_coma = sum(loss_a)
        self.optimizer['actor'].zero_grad()
        loss_coma.backward()
        if self.use_grad_clip:
            grad_norm = torch.nn.utils.clip_grad_norm_(self.model.actors.parameters(), self.grad_clip_norm)
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

        info.update(self.callback.on_update_end(self.iterations, method="update", model=self.model, info=info))

        return info

    def update_rnn(self, sample, epsilon=0.0):
        self.iterations += 1

        sample_Tensor = self.build_training_data(sample=sample,
                                                 use_parameter_sharing=self.use_parameter_sharing,
                                                 use_actions_mask=self.use_actions_mask,
                                                 use_global_state=True)
        batch_size = sample_Tensor['batch_size']
        state = sample_Tensor['state']
        obs = sample_Tensor['obs']
        actions = sample_Tensor['actions']
        returns = sample_Tensor['returns']
        avail_actions = sample_Tensor['avail_actions']
        agent_mask = sample_Tensor['agent_mask']
        filled = sample_Tensor['filled']
        seq_len = filled.shape[1]
        agent_indices = sample_Tensor['agent_indices']

        if not self.agent_grouping.full_independent:
            obs = self.packed_tensor(obs)
            packed_actions = self.packed_tensor(actions)
            packed_avail_actions = self.packed_tensor(avail_actions)
            returns = self.packed_tensor(returns)
            agent_mask = self.packed_tensor(agent_mask)
        else:
            packed_actions = actions
            packed_avail_actions = avail_actions

        info = self.callback.on_update_start(self.iterations, method="update_rnn",
                                             model=self.model, sample_Tensor=sample_Tensor, filled=filled)

        # initial hidden states for rnn
        rnn_states_actor = self.model.init_actor_rnn_states(batch_size)
        rnn_states_critic = self.model.init_critic_rnn_states(batch_size)

        # feedforward
        pi_dist_dict = self.model(observations=obs, agent_indices=agent_indices,
                                  avail_actions=avail_actions,
                                  rnn_states=rnn_states_actor, epsilon=epsilon).distributions

        joint_actions = torch.concat([one_hot(v.long(), self.n_actions[k]) for k, v in actions.items()],
                                     dim=2).reshape([batch_size, seq_len, -1])

        values_pred = self.model.get_values(states=state, observations=obs, joint_actions=joint_actions,
                                            agent_indices=agent_indices, rnn_states=rnn_states_critic,
                                            target=False).values
        if not self.agent_grouping.full_independent:
            values_pred = self.packed_tensor(values_pred)

        # calculate loss
        loss_a, loss_c = [], []
        for group, agent_keys in self.groups.items():
            bs_rnn = batch_size * self.n_group_agents[group]
            mask_values = agent_mask[group]
            pi_probs = pi_dist_dict[group].probs

            if self.use_actions_mask:
                pi_probs[packed_avail_actions[group] == 0] = 0.0  # mask out the unavailable actions.
                pi_probs = pi_probs[group] / pi_probs.sum(dim=-1, keepdim=True)  # re-normalize the actions.
                pi_probs[packed_avail_actions[group] == 0] = 0.0
            baseline = (pi_probs * values_pred[group]).sum(-1).reshape(bs_rnn, seq_len)
            pi_taken = pi_probs.gather(-1, packed_actions[group].unsqueeze(-1).long())
            q_taken = values_pred[group].gather(-1, packed_actions[group].unsqueeze(-1).long()).reshape(bs_rnn, seq_len)
            log_pi_taken = torch.log(pi_taken).reshape(bs_rnn, seq_len)
            advantages = (q_taken - baseline).detach()
            loss_a.append(-(advantages * log_pi_taken * mask_values).sum() / mask_values.sum())

            td_error = (q_taken - returns[group].detach()) * mask_values
            loss_c.append((td_error ** 2).sum() / mask_values.sum())

            info.update(self.callback.on_update_agent_wise(self.iterations, group, info=info, method="update_rnn",
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
            grad_norm = torch.nn.utils.clip_grad_norm_(self.model.critics.parameters(), self.grad_clip_norm)
            info["gradient_norm_actor"] = grad_norm.item()
        self.optimizer['critic'].step()
        if self.scheduler['critic'] is not None:
            self.scheduler['critic'].step()
        if self.iterations % self.sync_frequency == 0:
            self.model.copy_target()

        # update actor(s)
        loss_coma = sum(loss_a)
        self.optimizer['actor'].zero_grad()
        loss_coma.backward()
        if self.use_grad_clip:
            grad_norm = torch.nn.utils.clip_grad_norm_(self.model.actors.parameters(), self.grad_clip_norm)
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

        info.update(self.callback.on_update_end(self.iterations, method="update_rnn", model=self.model, info=info))

        return info
