"""
Independent Advantage Actor Critic (IAC)
Paper link: https://ojs.aaai.org/index.php/AAAI/article/view/11794
Implementation: Pytorch
"""
import torch
from torch import nn
from argparse import Namespace
from xuance.common import Optional, AgentGrouping
from xuance.torch.utils import ValueNorm
from xuance.torch.learners import LearnerMAS
from xuance.torch.rl_models.modules import OnPolicyMARL_Batch


class IAC_Learner(LearnerMAS):
    def __init__(self,
                 config: Namespace,
                 agent_grouping: AgentGrouping,
                 model: nn.Module,
                 callback):
        super(IAC_Learner, self).__init__(config, agent_grouping, model, callback)
        self.build_optimizer()
        self.use_value_clip, self.value_clip_range = config.use_value_clip, config.value_clip_range
        self.use_huber_loss, self.huber_delta = config.use_huber_loss, config.huber_delta
        self.use_value_norm = config.use_value_norm
        self.vf_coef, self.ent_coef = config.vf_coef, config.ent_coef
        self.mse_loss = nn.MSELoss()
        self.huber_loss = nn.HuberLoss(reduction="none", delta=self.huber_delta)
        if self.use_value_norm:
            self.value_normalizer = {key: ValueNorm(1).to(self.device) for key in self.group_keys}
        else:
            self.value_normalizer = None

    def estimate_total_iterations(self):
        """Estimated total number of training iterations"""
        buffer_size = self.config.buffer_size
        n_epochs = getattr(self.config, "n_epochs", 1)
        n_minibatch = getattr(self.config, "n_minibatch", 1)
        episode_length = self.episode_length
        if self.use_rnn:
            update_times = (self.config.running_steps // episode_length) // buffer_size
        else:
            update_times = self.config.running_steps // buffer_size
        total_iters = update_times * n_epochs * n_minibatch
        return total_iters

    def build_optimizer(self):
        self.optimizer = {
            group: torch.optim.Adam(
                self.model.parameters_model[group],
                lr=self.learning_rate,
                eps=1e-5,
                weight_decay=self.config.weight_decay
            )
            for group in self.group_keys
        }
        self.scheduler = {
            group: torch.optim.lr_scheduler.LinearLR(
                self.optimizer[group],
                start_factor=1.0,
                end_factor=self.end_factor_lr_decay,
                total_iters=self.total_iters
            )
            for group in self.group_keys
        }

    def build_training_data(
            self,
            sample: Optional[dict],
            use_parameter_sharing: Optional[bool] = False,
            use_actions_mask: Optional[bool] = False,
            use_global_state: Optional[bool] = False
    ) -> OnPolicyMARL_Batch:
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
        state, filled = None, None
        obs, actions, rewards, terminals, agent_mask = {}, {}, {}, {}, {}
        values, returns, advantages, log_pi_old = {}, {}, {}, {}
        avail_actions = {} if self.use_actions_mask else None
        agent_indices = {}

        for agent in self.agent_keys:
            obs[agent] = torch.as_tensor(sample['obs'][agent], device=self.device)
            actions[agent] = torch.as_tensor(sample['actions'][agent], device=self.device)
            rewards[agent] = torch.as_tensor(sample['rewards'][agent], device=self.device)
            agent_mask[agent] = torch.as_tensor(sample['agent_mask'][agent], device=self.device, dtype=torch.float32)
            values[agent] = torch.as_tensor(sample['values'][agent], device=self.device)
            returns[agent] = torch.as_tensor(sample['returns'][agent], device=self.device)
            advantages[agent] = torch.as_tensor(sample['advantages'][agent], device=self.device)
            log_pi_old[agent] = torch.as_tensor(sample['log_pi_old'][agent], device=self.device)
            if use_actions_mask:
                avail_actions[agent] = torch.as_tensor(sample['avail_actions'][agent],
                                                       device=self.device, dtype=torch.float32)

        if use_global_state:
            state = torch.as_tensor(sample['state'], device=self.device)

        if self.use_rnn:
            filled = torch.as_tensor(sample['filled'], device=self.device, dtype=torch.float32)

        for key, n_agents in self.n_group_agents.items():
            bs = batch_size * n_agents

            if self.use_rnn:
                agents_id = torch.as_tensor(self.agent_indices[key], dtype=torch.int64).repeat(
                    batch_size, 1).reshape(bs, 1, 1).expand(-1, seq_length, -1).to(self.device)
            else:
                agents_id = torch.as_tensor(self.agent_indices[key], dtype=torch.int64).repeat(
                    batch_size, 1).reshape([bs, 1]).to(self.device)

            agent_indices[key] = agents_id

        # from agent-wise to group-wise
        if not self.agent_grouping.full_independent:
            obs = self.packed_tensor(obs)
            actions = self.packed_tensor(actions)
            values = self.packed_tensor(values)
            returns = self.packed_tensor(returns)
            advantages = self.packed_tensor(advantages)
            log_pi_old = self.packed_tensor(log_pi_old)
            agent_mask = self.packed_tensor(agent_mask)
            avail_actions = self.packed_tensor(avail_actions)

        return OnPolicyMARL_Batch(
            batch_size=batch_size,
            global_states=state,
            observations=obs,
            actions=actions,
            values=values,
            returns=returns,
            advantages=advantages,
            old_log_probs=log_pi_old,
            agent_masks=agent_mask,
            avail_actions=avail_actions,
            agent_indices=agent_indices,
            filled_masks=filled,
            seq_length=seq_length
        )

    def update(self, sample):
        self.iterations += 1

        # prepare training data
        batch = self.build_training_data(sample=sample,
                                         use_parameter_sharing=self.use_parameter_sharing,
                                         use_actions_mask=self.use_actions_mask)

        info = self.callback.on_update_start(self.iterations, model=self.model, batch=batch)

        # initial hidden states for rnn
        rnn_states_actor = self.model.init_actor_rnn_states(batch.batch_size)
        rnn_states_critic = self.model.init_critic_rnn_states(batch.batch_size)

        # feedforward
        policy_outputs = self.model(
            observations=batch.observations,
            agent_indices=batch.agent_indices,
            avail_actions=batch.avail_actions,
            rnn_states=rnn_states_actor
        )
        value_outputs = self.model.get_values(
            observations=batch.observations,
            agent_indices=batch.agent_indices,
            rnn_states=rnn_states_critic
        )
        values_pred = value_outputs.values
        if not self.agent_grouping.full_independent:
            values_pred = self.packed_tensor(values_pred)

        # calculate losses and update networks for each group of agents
        for group, n_agents in self.n_group_agents.items():
            mask_values = batch.valid_mask(group, n_agents).reshape(-1)

            # actor loss
            dist = policy_outputs.distributions[group]
            log_pi = dist.log_prob(batch.actions[group]).reshape(-1)
            advantages = batch.advantages[group].reshape(-1)

            actor_loss = -((advantages.detach() * log_pi) * mask_values).sum() / mask_values.sum()

            # entropy loss
            entropy = dist.entropy().reshape(-1)
            entropy_loss = (entropy * mask_values).sum() / mask_values.sum()

            # value loss
            value_pred_i = values_pred[group].reshape(-1)
            value_target = batch.returns[group].reshape(-1)
            values_i = batch.values[group].reshape(-1)
            if self.use_value_clip:
                value_clipped = values_i + (value_pred_i - values_i).clamp(
                    -self.value_clip_range,
                    self.value_clip_range
                )
                if self.use_value_norm:
                    self.value_normalizer[group].update(value_target.reshape(-1, 1))
                    value_target = self.value_normalizer[group].normalize(value_target.reshape(-1, 1))
                    value_target = value_target.reshape(-1)
                if self.use_huber_loss:
                    loss_v = self.huber_loss(value_pred_i, value_target)
                    loss_v_clipped = self.huber_loss(value_clipped, value_target)
                else:
                    loss_v = (value_pred_i - value_target) ** 2
                    loss_v_clipped = (value_clipped - value_target) ** 2
                loss_c_ = torch.max(loss_v, loss_v_clipped)
                critic_loss = (loss_c_ * mask_values).sum() / mask_values.sum()
            else:
                if self.use_value_norm:
                    self.value_normalizer[group].update(value_target)
                    value_target = self.value_normalizer[group].normalize(value_target)
                if self.use_huber_loss:
                    loss_v = self.huber_loss(value_pred_i, value_target)
                else:
                    loss_v = (value_pred_i - value_target) ** 2
                critic_loss = (loss_v * mask_values).sum() / mask_values.sum()

            loss = actor_loss + self.vf_coef * critic_loss - self.ent_coef * entropy_loss

            self.optimizer[group].zero_grad()
            loss.backward()
            if self.use_grad_clip:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters_model[group],
                    self.grad_clip_norm
                )
                info["gradient_norm"] = grad_norm.item()
            self.optimizer[group].step()
            if self.scheduler is not None:
                self.scheduler[group].step()

            # Logger
            lr = self.optimizer[group].state_dict()['param_groups'][0]['lr']

            info.update({
                f"{group}/actor_loss": actor_loss.item(),
                f"{group}/critic_loss": critic_loss.item(),
                f"{group}/entropy": entropy_loss.item(),
                f"{group}/total_loss": loss.item(),
                f"{group}/predict_value": value_pred_i.mean().item(),
                f"{group}/learning_rate": lr,
            })

            info.update(self.callback.on_update_agent_wise(self.iterations, group, info=info,
                                                           mask_values=mask_values, log_pi=log_pi,
                                                           actor_loss=actor_loss,
                                                           entropy=entropy, entropy_loss=entropy_loss,
                                                           value_pred_i=value_pred_i, value_target=value_target,
                                                           values_i=values_i, loss_v=loss_v))

        info.update(self.callback.on_update_end(self.iterations, method="update", model=self.model, info=info))

        return info
