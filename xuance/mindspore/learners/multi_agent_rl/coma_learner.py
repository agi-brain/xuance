"""
COMA: Counterfactual Multi-Agent Policy Gradients
Paper link: https://ojs.aaai.org/index.php/AAAI/article/view/11794
Implementation: MindSpore
"""
from argparse import Namespace
from xuance.common import AgentGrouping

from xuance.mindspore import ms, ops, nn, Tensor, Module, optim
from xuance.mindspore.utils import ValueNorm
from xuance.mindspore.learners.multi_agent_rl.iac_learner import IAC_Learner
from xuance.mindspore.learners.base.marl_learner import OnPolicyMultiAgentLearner
from xuance.mindspore.rl_models.modules import OnPolicyMARLBatch
from xuance.mindspore.utils import clip_grads


class COMA_Learner(IAC_Learner):
    def __init__(self,
                 config: Namespace,
                 agent_grouping: AgentGrouping,
                 model: Module,
                 callback):
        config.use_value_clip, config.value_clip_range = False, None
        config.use_huber_loss, config.huber_delta = False, None
        config.use_value_norm = False
        config.vf_coef, config.ent_coef = None, None
        super(OnPolicyMultiAgentLearner, self).__init__(config, agent_grouping, model, callback)
        self.build_optimizer()

        self.use_value_clip = config.use_value_clip
        self.value_clip_range = config.value_clip_range
        self.use_huber_loss = config.use_huber_loss
        self.huber_delta = config.huber_delta
        self.use_value_norm = config.use_value_norm
        self.vf_coef = config.vf_coef
        self.ent_coef = config.ent_coef

        self.mse_loss = nn.MSELoss()
        self.huber_loss = nn.HuberLoss(reduction="none", delta=self.huber_delta)
        self.softmax = nn.Softmax(axis=-1)

        if self.use_value_norm:
            self.value_normalizer = {key: ValueNorm(1) for key in self.group_keys}
        else:
            self.value_normalizer = None

        self.sync_frequency = config.sync_frequency
        self.n_actions = {k: self.model.critics.action_space[k].n for k in self.agent_keys}

        # Get gradient function
        self.grad_fn_actor = ms.value_and_grad(self.forward_fn_actor, None,
                                               self.optimizer['actor'].parameters, has_aux=True)
        self.grad_fn_critic = ms.value_and_grad(self.forward_fn_critic, None,
                                                self.optimizer['critic'].parameters, has_aux=True)
        self.model.set_train()

    def build_optimizer(self):
        self.optimizer = {
            'actor': optim.Adam(params=self.model.actors.trainable_params(), lr=self.config.learning_rate_actor,
                                eps=1e-5),
            'critic': optim.Adam(params=self.model.critics.trainable_params(), lr=self.config.learning_rate_critic,
                                 eps=1e-5)
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

    def forward_fn_actor(self, batch: OnPolicyMARLBatch, *args, **kwargs):
        joint_actions, rnn_states_critic = args
        info = {}
        # feedforward
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
        loss_c = []
        for group, n_agents in self.n_group_agents.items():
            mask_values = batch.valid_mask(group, n_agents).reshape(-1)

            returns = ops.stop_gradient(batch.returns.packed(group).reshape(-1))
            q_taken = values_pred.packed(group).gather(-1, batch.actions.packed(group).unsqueeze(-1).long()).reshape(-1)
            td_error = (q_taken - returns) * mask_values
            loss_c.append((td_error ** 2).sum() / mask_values.sum())

            info.update(self.callback.on_update_agent_wise(self.iterations, group, info=info,
                                                           mask_values=mask_values, q_taken=q_taken,
                                                           td_error=td_error))

        # update critic
        loss_critic = sum(loss_c)

        return loss_critic, info

    def forward_fn_critic(self, batch: OnPolicyMARLBatch, *args, **kwargs):
        joint_actions, epsilon, rnn_states_actor, rnn_states_critic = args
        info = {}
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
        loss_a = []
        for group, n_agents in self.n_group_agents.items():
            mask_values = batch.valid_mask(group, n_agents).reshape(-1)

            dist = policy_outputs.distributions[group]
            pi_probs = dist.probs

            if self.use_actions_mask:
                pi_probs[batch.avail_actions.packed(group) == 0] = 0.0  # mask out the unavailable actions.
                pi_probs = pi_probs / pi_probs.sum(dim=-1, keepdim=True)  # re-normalize the actions.
            baseline = (pi_probs * values_pred.packed(group)).sum(-1).reshape(-1)
            pi_taken = pi_probs.gather(-1, batch.actions.packed(group).unsqueeze(-1).long())
            q_taken = values_pred.packed(group).gather(-1, batch.actions.packed(group).unsqueeze(-1).long()).reshape(-1)
            log_pi_taken = ops.log(pi_taken).reshape(-1)
            advantages = ops.stop_gradient(q_taken - baseline)

            loss_a.append(-(advantages * log_pi_taken * mask_values).sum() / mask_values.sum())

            info.update(self.callback.on_update_agent_wise(self.iterations, group, info=info,
                                                           mask_values=mask_values, pi_probs=pi_probs,
                                                           baseline=baseline, pi_taken=pi_taken,
                                                           q_taken=q_taken, log_pi_taken=log_pi_taken,
                                                           advantages=advantages, loss_a=loss_a))

        # update actor(s)
        loss_coma = sum(loss_a)
        return loss_coma, info

    def update(self, sample, epsilon=0.0):
        self.iterations += 1

        # prepare training data
        batch = self.build_training_data(sample=sample,
                                         use_actions_mask=self.use_actions_mask,
                                         use_global_state=True)

        joint_actions = ops.concat([ops.one_hot(Tensor(v, device=self.device, dtype=ms.int64), self.n_actions[k])
                                    for k, v in sample['actions'].items()], axis=-1)
        if self.use_rnn:
            joint_actions = joint_actions.reshape([batch.batch_size, batch.seq_length, -1])
        else:
            joint_actions = joint_actions.reshape([batch.batch_size, -1])

        info = self.callback.on_update_start(self.iterations, model=self.model, batch=batch)

        # initial hidden states for rnn
        rnn_states_actor = self.model.init_actor_rnn_states(batch.batch_size)
        rnn_states_critic = self.model.init_critic_rnn_states(batch.batch_size)

        (loss_critic, info_critic), grads_critic = self.grad_fn_critic(
            batch, joint_actions, epsilon, rnn_states_actor, rnn_states_critic)
        if self.use_grad_clip:
            grads_critic = clip_grads(grads_critic, Tensor(-self.grad_clip_norm), Tensor(self.grad_clip_norm))
        self.optimizer['critic'](grads_critic)

        if self.iterations % self.sync_frequency == 0:
            self.model.copy_target()

        # update actor(s)
        (loss_coma, info_actor), grads_actor = self.grad_fn_actor(batch, joint_actions, rnn_states_critic)
        if self.use_grad_clip:
            grads_actor = clip_grads(grads_actor, Tensor(-self.grad_clip_norm), Tensor(self.grad_clip_norm))
        self.optimizer['actor'](grads_actor)

        # Logger
        self.scheduler['actor'].step()
        self.scheduler['critic'].step()
        learning_rate_actor = self.scheduler['actor'].get_last_lr()[0]
        learning_rate_critic = self.scheduler['critic'].get_last_lr()[0]

        info.update(info_actor)
        info.update(info_critic)

        info.update({
            "learning_rate_actor": learning_rate_actor.asnumpy(),
            "learning_rate_critic": learning_rate_critic.asnumpy(),
            "actor_loss": loss_coma.asnumpy(),
            "critic_loss": loss_critic.asnumpy(),
        })

        info.update(self.callback.on_update_end(self.iterations, model=self.model, info=info))

        return info
