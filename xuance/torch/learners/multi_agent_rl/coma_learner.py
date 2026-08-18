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
        batch = self.build_training_data(sample=sample,
                                         use_actions_mask=self.use_actions_mask,
                                         use_global_state=True)

        joint_actions = torch.concat([one_hot(torch.as_tensor(v, device=self.device, dtype=torch.int64),
                                              num_classes=self.n_actions[k])
                                      for k, v in sample['actions'].items()], dim=-1)
        if self.use_rnn:
            joint_actions = joint_actions.reshape([batch.batch_size, batch.seq_length, -1])
        else:
            joint_actions = joint_actions.reshape([batch.batch_size, -1])

        info = self.callback.on_update_start(self.iterations, model=self.model, batch=batch)

        # initial hidden states for rnn
        rnn_states_actor = self.model.init_actor_rnn_states(batch.batch_size)
        rnn_states_critic = self.model.init_critic_rnn_states(batch.batch_size)

        # feedforward
        policy_outputs = self.model(
            observations=batch.observations,
            agent_indices=batch.agent_indices,
            avail_actions=batch.avail_actions,
            rnn_states=rnn_states_actor,
            epsilon=epsilon
        )
        value_outputs = self.model.get_values(
            states=batch.global_states,
            observations=batch.observations,
            joint_actions=joint_actions,
            agent_indices=batch.agent_indices,
            rnn_states=rnn_states_critic,
            target=False
        )
        values_pred = value_outputs.values

        # calculate actor and critic losses
        loss_a, loss_c = [], []
        for group, n_agents in self.n_group_agents.items():
            mask_values = batch.valid_mask(group, n_agents).reshape(-1)

            dist = policy_outputs.distributions[group]
            pi_probs = dist.probs
            returns = batch.returns.packed(group).detach().reshape(-1)

            if self.use_actions_mask:
                pi_probs[batch.avail_actions.packed(group) == 0] = 0.0  # mask out the unavailable actions.
                pi_probs = pi_probs / pi_probs.sum(dim=-1, keepdim=True)  # re-normalize the actions.
            baseline = (pi_probs * values_pred.packed(group)).sum(-1).reshape(-1)
            pi_taken = pi_probs.gather(-1, batch.actions.packed(group).unsqueeze(-1).long())
            q_taken = values_pred.packed(group).gather(-1, batch.actions.packed(group).unsqueeze(-1).long()).reshape(-1)
            log_pi_taken = torch.log(pi_taken).reshape(-1)
            advantages = (q_taken - baseline).detach()

            loss_a.append(-(advantages * log_pi_taken * mask_values).sum() / mask_values.sum())

            td_error = (q_taken - returns) * mask_values
            loss_c.append((td_error ** 2).sum() / mask_values.sum())

            info.update(self.callback.on_update_agent_wise(self.iterations, group, info=info,
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

        info.update(self.callback.on_update_end(self.iterations, model=self.model, info=info))

        return info

