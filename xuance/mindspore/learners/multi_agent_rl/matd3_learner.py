"""
Multi-Agent TD3
"""
from argparse import Namespace
from xuance.common import AgentGrouping

from xuance.mindspore import ops, Module, Tensor
from xuance.mindspore.utils import clip_grads, AgentGroupedTensor
from xuance.mindspore.learners.multi_agent_rl.itd3_learner import ITD3_Learner
from xuance.mindspore.rl_models.modules import OffPolicyMARLBatch


class MATD3_Learner(ITD3_Learner):
    def __init__(self,
                 config: Namespace,
                 agent_grouping: AgentGrouping,
                 model: Module,
                 callback):
        ITD3_Learner.__init__(self, config, agent_grouping, model, callback)

    def forward_fn_actor(self, batch: OffPolicyMARLBatch, *args, **kwargs):
        observations_t, agent_indices_t, rnn_states_actor, rnn_states_critic_1, rnn_states_critic_2, group_key = args
        obs_joint_t = kwargs["joint_observations_t"]

        actions_eval = self.model(observations=observations_t,
                                  agent_indices=agent_indices_t,
                                  group_key=group_key,
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
        _, _, q_policy = self.model.Qpolicy(joint_observations=obs_joint_t,
                                            joint_actions=actions_joint_eval,
                                            agent_indices=agent_indices_t,
                                            group_key=group_key,
                                            rnn_states_1=rnn_states_critic_1,
                                            rnn_states_2=rnn_states_critic_2)
        q_policy_i = q_policy.packed(group_key).reshape(-1)
        loss_actor = -(q_policy_i * mask_values).sum() / mask_values.sum()

        return loss_actor, q_policy_i

    def forward_fn_critic(self, batch: OffPolicyMARLBatch, *args, **kwargs):
        observations_t, agent_indices_t, rnn_states_actor, rnn_states_critic_1, rnn_states_critic_2, group_key = args
        obs_joint_t = kwargs["joint_observations_t"]
        actions_joint_t = kwargs["joint_actions_t"]

        # get values
        q_eval_A, q_eval_B, _ = self.model.Qpolicy(joint_observations=obs_joint_t,
                                                   joint_actions=actions_joint_t,
                                                   agent_indices=agent_indices_t,
                                                   group_key=group_key,
                                                   rnn_states_1=rnn_states_critic_1,
                                                   rnn_states_2=rnn_states_critic_2)

        if self.use_rnn:
            actions_next = self.model.Atarget(observations=batch.observations,
                                              agent_indices=batch.agent_indices,
                                              group_key=group_key,
                                              rnn_states=rnn_states_actor)
            actions_joint_next = self.get_joint_input(actions_next.agent_wise,
                                                      (batch.batch_size, batch.seq_length + 1, -1))
            q_next = self.model.Qtarget(joint_observations=obs_joint_t,
                                        joint_actions=actions_joint_next,
                                        agent_indices=batch.agent_indices,
                                        group_key=group_key,
                                        rnn_states_1=rnn_states_critic_1,
                                        rnn_states_2=rnn_states_critic_2)
            q_next.grouped_tensor = {k: v[:, :, 1:] for k, v in q_next.grouped_tensor.items()}
        else:
            actions_next = self.model.Atarget(observations=batch.next_observations,
                                              agent_indices=batch.agent_indices,
                                              group_key=group_key)
            actions_joint_next = self.get_joint_input(actions_next.agent_wise, (batch.batch_size, -1))
            obs_joint_next = self.get_joint_input(batch.next_observations.agent_wise,
                                                  output_shape=(batch.batch_size, -1))
            q_next = self.model.Qtarget(joint_observations=obs_joint_next,
                                        joint_actions=actions_joint_next,
                                        agent_indices=batch.agent_indices,
                                        group_key=group_key)

        # Loss for critic(s)
        n_agents = self.n_group_agents[group_key]
        mask_values = batch.valid_mask(group_key, n_agents).reshape(-1)

        q_eval_A_i = q_eval_A.packed(group_key).reshape(-1)
        q_eval_B_i = q_eval_B.packed(group_key).reshape(-1)
        q_next_i = q_next.packed(group_key).reshape(-1)
        rewards = batch.rewards.packed(group_key).reshape(-1)
        terminals = batch.terminals.packed(group_key).reshape(-1)

        q_target = rewards + (1 - terminals) * self.gamma * q_next_i
        q_target = ops.stop_gradient(q_target)
        td_error_A = (q_eval_A_i - q_target) * mask_values
        td_error_B = (q_eval_B_i - q_target) * mask_values

        loss_critics = ((td_error_A ** 2).sum() + (td_error_B ** 2).sum()) / mask_values.sum()

        return loss_critics, q_eval_A_i, q_eval_B_i, td_error_A, td_error_B, q_target

    def update(self, sample):
        self.iterations += 1

        # prepare training data
        batch = self.build_training_data(
            sample,
            use_actions_mask=False
        )

        info = self.callback.on_update_start(self.iterations, model=self.model, batch=batch)

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
            obs_joint_t = obs_joint = self.get_joint_input(batch.observations.agent_wise,
                                                           output_shape=(batch.batch_size, -1))
            actions_joint_t = self.get_joint_input(batch.actions.agent_wise, output_shape=(batch.batch_size, -1))
            observations_t = batch.observations
            agent_indices_t = batch.agent_indices

        # initial hidden states for rnn
        rnn_states_actor = self.model.init_actor_rnn_states(batch.batch_size)
        rnn_states_critic_1, rnn_states_critic_2 = self.model.init_critic_rnn_states(batch.batch_size)

        # update critic(s)
        for group_key in self.group_keys:
            (loss_c, q_eval_A_i, q_eval_B_i, td_error_A, td_error_B, q_target), grads_critic = self.grad_fn_critic[
                group_key](batch, observations_t, agent_indices_t, rnn_states_actor, rnn_states_critic_1,
                           rnn_states_critic_2, group_key,
                           joint_observations_t=obs_joint_t, joint_actions_t=actions_joint_t)
            if self.use_grad_clip:
                grads_critic = clip_grads(grads_critic, Tensor(-self.grad_clip_norm), Tensor(self.grad_clip_norm))
            self.optimizer[group_key]['critic'](grads_critic)

            self.scheduler[group_key]['critic'].step()
            learning_rate_critic = self.scheduler[group_key]['critic'].get_last_lr()[0]

            info.update({
                f"{group_key}/learning_rate_critic": learning_rate_critic.asnumpy(),
                f"{group_key}/loss_critic": loss_c.asnumpy(),
                f"{group_key}/predictQ_A": q_eval_A_i.mean().asnumpy(),
                f"{group_key}/predictQ_B": q_eval_B_i.mean().asnumpy()
            })

            info.update(self.callback.on_update_agent_wise(self.iterations, group_key, info=info,
                                                           q_eval_A_i=q_eval_A_i, q_eval_B_i=q_eval_B_i,
                                                           q_target=q_target,
                                                           td_error_A=td_error_A, td_error_B=td_error_B))

        # update actor(s)
        if self.iterations % self.actor_update_delay == 0:
            for group_key in self.group_keys:
                (loss_a, q_eval_i), grads_actor = self.grad_fn_actor[group_key](
                    batch, observations_t, agent_indices_t, rnn_states_actor, rnn_states_critic_1, rnn_states_critic_2,
                    group_key, joint_observations_t=obs_joint_t
                )
                if self.use_grad_clip:
                    grads_actor = clip_grads(grads_actor, Tensor(-self.grad_clip_norm), Tensor(self.grad_clip_norm))
                self.optimizer[group_key]['actor'](grads_actor)

                self.scheduler[group_key]['actor'].step()
                learning_rate_actor = self.scheduler[group_key]['actor'].get_last_lr()[0]

                info.update({
                    f"{group_key}/learning_rate_actor": learning_rate_actor.asnumpy(),
                    f"{group_key}/loss_actor": loss_a.asnumpy(),
                    f"{group_key}/q_policy": q_eval_i.mean().asnumpy(),
                })
                info.update(self.callback.on_update_agent_wise(self.iterations, group_key, info=info,
                                                               q_eval_i=q_eval_i))

            self.model.soft_update(self.tau)

        info.update(self.callback.on_update_end(self.iterations, model=self.model, info=info))
        return info
