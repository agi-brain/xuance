"""
Multi-Agent Deep Deterministic Policy Gradient
Paper link:
https://proceedings.neurips.cc/paper/2017/file/68a9750337a418a86fe06c1991a1d64c-Paper.pdf
Implementation: MindSpore
Trick: Parameter sharing for all agents, with agents' one-hot IDs as actor-critic's inputs.
"""
from xuance.mindspore import Tensor, ops
from xuance.mindspore.utils import clip_grads
from xuance.mindspore.utils import AgentGroupedTensor
from xuance.mindspore.learners.multi_agent_rl.iddpg_learner import IDDPG_Learner
from xuance.mindspore.rl_models.modules import OffPolicyMARLBatch


class MADDPG_Learner(IDDPG_Learner):
    def forward_fn_actor(self, batch: OffPolicyMARLBatch, *args, **kwargs):
        observations_t, agent_indices_t, rnn_states_actor, rnn_states_critic, group_key = args
        obs_joint_t = kwargs["joint_observations_t"]

        # feedforward
        actions_eval = self.model(observations=observations_t,
                                  agent_indices=agent_indices_t,
                                  rnn_states=rnn_states_actor).actions
        actions_eval_agent_wise = actions_eval.agent_wise

        n_agents = self.n_group_agents[group_key]
        mask_values = batch.valid_mask(group_key, n_agents).reshape(-1)
        agent_keys = self.groups[group_key]

        # calculate the objective of actor
        actions_all = batch.actions.agent_wise.copy()
        for key in agent_keys:
            actions_all[key] = actions_eval_agent_wise[key]
        if self.use_rnn:
            actions_joint_eval = self.get_joint_input(actions_all, (batch.batch_size, batch.seq_length, -1))
        else:
            actions_joint_eval = self.get_joint_input(actions_all, (batch.batch_size, -1))
        q_policy = self.model.Qpolicy(joint_observations=obs_joint_t,
                                      joint_actions=actions_joint_eval,
                                      agent_indices=agent_indices_t,
                                      group_key=group_key,
                                      rnn_states=rnn_states_critic)
        q_policy_i = q_policy.packed(group_key).reshape(-1)
        loss_actor = -(q_policy_i * mask_values).sum() / mask_values.sum()

        return loss_actor, q_policy_i, actions_joint_eval

    def forward_fn_critic(self, batch: OffPolicyMARLBatch, *args, **kwargs):
        observations_t, agent_indices_t, rnn_states_actor, rnn_states_critic, group_key = args
        obs_joint_t = kwargs["joint_observations_t"]
        actions_joint_t = kwargs["joint_actions_t"]

        q_eval = self.model.Qpolicy(joint_observations=obs_joint_t,
                                    joint_actions=actions_joint_t,
                                    agent_indices=agent_indices_t,
                                    group_key=group_key,
                                    rnn_states=rnn_states_critic)

        if self.use_rnn:
            actions_next = self.model.Atarget(observations=batch.observations,
                                              agent_indices=batch.agent_indices,
                                              rnn_states=rnn_states_actor)
            actions_joint_next = self.get_joint_input(actions_next.agent_wise,
                                                      (batch.batch_size, batch.seq_length + 1, -1))
            q_next = self.model.Qtarget(joint_observations=obs_joint_t,
                                        joint_actions=actions_joint_next,
                                        agent_indices=batch.agent_indices,
                                        rnn_states=rnn_states_critic)
            q_next.grouped_tensor = {k: v[:, :, 1:] for k, v in q_next.grouped_tensor.items()}
        else:
            actions_next = self.model.Atarget(observations=batch.next_observations,
                                              agent_indices=batch.agent_indices)
            actions_joint_next = self.get_joint_input(actions_next.agent_wise, (batch.batch_size, -1))
            obs_joint_next = self.get_joint_input(batch.next_observations.agent_wise,
                                                  output_shape=(batch.batch_size, -1))
            q_next = self.model.Qtarget(joint_observations=obs_joint_next,
                                        joint_actions=actions_joint_next,
                                        agent_indices=batch.agent_indices,
                                        rnn_states=rnn_states_critic)

        n_agents = self.n_group_agents[group_key]
        mask_values = batch.valid_mask(group_key, n_agents).reshape(-1)

        # update critic
        q_eval_a = q_eval.packed(group_key).reshape(-1)
        q_next_i = q_next.packed(group_key).reshape(-1)
        rewards = batch.rewards.packed(group_key).reshape(-1)
        terminals = batch.terminals.packed(group_key).reshape(-1)

        q_target = rewards + (1 - terminals) * self.gamma * q_next_i
        q_target = ops.stop_gradient(q_target)
        td_error = (q_eval_a - q_target) * mask_values

        loss_critic = (td_error ** 2).sum() / mask_values.sum()

        return loss_critic, q_eval_a, td_error

    def update(self, sample):
        self.iterations += 1

        # prepare training data
        batch = self.build_training_data(
            sample,
            use_actions_mask=False
        )

        info = self.callback.on_update_start(self.iterations, model=self.model, batch=batch)

        # initial hidden states for rnn
        rnn_states_actor = self.model.init_actor_rnn_states(batch.batch_size)
        rnn_states_critic = self.model.init_critic_rnn_states(batch.batch_size)

        if self.use_rnn:
            obs_joint = self.get_joint_input(batch.observations.agent_wise,
                                             output_shape=(batch.batch_size, batch.seq_length + 1, -1))
            obs_joint_t = obs_joint[:, :-1]
            actions_joint_t = self.get_joint_input(batch.actions.agent_wise,
                                                   output_shape=(batch.batch_size, batch.seq_length, -1))
            observations_t = AgentGroupedTensor(
                {k: v[:, :, :-1] for k, v in batch.observations.grouped_tensor.items()}, self.agent_grouping
            )
            agent_indices_t = AgentGroupedTensor(
                {k: v[:, :, :-1] for k, v in batch.agent_indices.grouped_tensor.items()}, self.agent_grouping
            )
        else:
            obs_joint_t = self.get_joint_input(batch.observations.agent_wise,
                                               output_shape=(batch.batch_size, -1))
            actions_joint_t = self.get_joint_input(batch.actions.agent_wise, output_shape=(batch.batch_size, -1))
            observations_t = batch.observations
            agent_indices_t = batch.agent_indices

        for group_key in self.group_keys:
            # update critics
            (loss_c, q_eval, td_error), grads_critic = self.grad_fn_critic[group_key](
                batch, observations_t, agent_indices_t, rnn_states_actor, rnn_states_critic, group_key,
                joint_observations_t=obs_joint_t, joint_actions_t=actions_joint_t
            )
            if self.use_grad_clip:
                grads_critic = clip_grads(grads_critic, Tensor(-self.grad_clip_norm), Tensor(self.grad_clip_norm))
            self.optimizer[group_key]['critic'](grads_critic)

            # update actors
            (loss_a, q_eval_i, actions_joint_eval), grads_actor = self.grad_fn_actor[group_key](
                batch, observations_t, agent_indices_t, rnn_states_actor, rnn_states_critic, group_key,
                joint_observations_t=obs_joint_t
            )
            if self.use_grad_clip:
                grads_actor = clip_grads(grads_actor, Tensor(-self.grad_clip_norm), Tensor(self.grad_clip_norm))
            self.optimizer[group_key]['actor'](grads_actor)

            self.scheduler[group_key]['actor'].step()
            self.scheduler[group_key]['critic'].step()
            learning_rate_actor = self.scheduler[group_key]['actor'].get_last_lr()[0]
            learning_rate_critic = self.scheduler[group_key]['critic'].get_last_lr()[0]

            info.update({
                f"{group_key}/learning_rate_actor": learning_rate_actor.asnumpy(),
                f"{group_key}/learning_rate_critic": learning_rate_critic.asnumpy(),
                f"{group_key}/loss_actor": loss_a.asnumpy(),
                f"{group_key}/loss_critic": loss_c.asnumpy(),
                f"{group_key}/predictQ": q_eval_i.mean().asnumpy()
            })

            info.update(self.callback.on_update_agent_wise(self.iterations, group_key, info=info,
                                                           actions_joint_eval=actions_joint_eval,
                                                           q_eval_i=q_eval_i, q_eval=q_eval,
                                                           td_error=td_error))

        self.model.soft_update(self.tau)
        info.update(self.callback.on_update_end(self.iterations, model=self.model, info=info))
        return info
