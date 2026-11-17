"""
QTRAN: Learning to Factorize with Transformation for Cooperative Multi-Agent Reinforcement Learning
Paper link:
http://proceedings.mlr.press/v97/son19a/son19a.pdf
Implementation: MindSpore
"""
from argparse import Namespace
from operator import itemgetter
from xuance.common import AgentGrouping

from xuance.mindspore import ms, Module, Tensor, ops
from xuance.mindspore.rl_models.modules import OffPolicyMARLBatch, AgentGroupedTensor
from xuance.mindspore.learners import OffPolicyMultiAgentLearner
from xuance.mindspore.utils import clip_grads


class QTRAN_Learner(OffPolicyMultiAgentLearner):
    def __init__(self,
                 config: Namespace,
                 agent_grouping: AgentGrouping,
                 model: Module,
                 callback):
        super(QTRAN_Learner, self).__init__(config, agent_grouping, model, callback)
        self.sync_frequency = config.sync_frequency
        self.n_actions = {k: self.model.individual_q_networks[k].action_space.n for k in self.group_keys}
        # Get gradient function
        self.grad_fn = ms.value_and_grad(self.forward_fn, None, self.optimizer.parameters, has_aux=True)
        self.model.set_train()

    def forward_fn(self, batch: OffPolicyMARLBatch):
        info_train = {}

        batch_size = batch.batch_size
        state = batch.global_states
        state_next = batch.next_global_states
        obs = batch.observations
        actions = batch.actions
        obs_next = batch.next_observations
        rewards = batch.rewards.agent_wise
        terminals = batch.terminals.agent_wise
        agent_mask = batch.agent_masks
        avail_actions = batch.avail_actions
        avail_actions_next = batch.next_avail_actions
        agent_indices = batch.agent_indices

        rewards_tot = ops.stack(itemgetter(*self.agent_keys)(rewards), axis=1).mean(dim=-1, keepdim=True)
        terminals_tot = ops.stack(itemgetter(*self.agent_keys)(terminals), axis=1).all(dim=1, keepdim=True).float()

        info = self.callback.on_update_start(self.iterations, method="update",
                                             model=self.model, batch=batch,
                                             rewards_tot=rewards_tot, terminals_tot=terminals_tot)

        model_output = self.model(observations=obs,
                                  agent_indices=agent_indices,
                                  avail_actions=avail_actions)
        actions_greedy, q_eval = model_output.actions, model_output.values
        hidden_state = {k: v.embeddings for k, v in model_output.rep_out.items()}

        target_model_output = self.model.Qtarget(observations=obs_next,
                                                 agent_indices=agent_indices)
        q_next = target_model_output.values
        hidden_state_next = {k: v.embeddings for k, v in target_model_output.rep_out.items()}

        if self.config.double_q:
            a_next_greedy = self.model(observations=obs_next,
                                       agent_indices=agent_indices,
                                       avail_actions=avail_actions_next).actions
        else:
            a_next_greedy = {}

        q_eval_a, q_eval_greedy_a, q_next_a = {}, {}, {}
        for key in self.agent_keys:
            mask_values = agent_mask.agent_wise[key]
            q_eval_a[key] = q_eval.agent_wise[key].gather(-1, actions.agent_wise[key].long().unsqueeze(-1)).reshape(
                batch_size)
            q_eval_greedy_a[key] = q_eval.agent_wise[key].gather(-1, actions_greedy.agent_wise[key].long().unsqueeze(
                -1)).reshape(batch_size)

            if self.use_actions_mask:
                q_next.agent_wise[key][avail_actions_next[key] == 0] = -1e10

            if self.config.double_q:
                q_next_a[key] = q_next.agent_wise[key].gather(-1, a_next_greedy.agent_wise[key].long().unsqueeze(
                    -1)).reshape(batch_size)
            else:
                a_next_greedy[key] = q_next.agent_wise[key].argmax(dim=-1, keepdim=False)
                q_next_a[key] = q_next.agent_wise[key].max(dim=-1, keepdim=True).values.reshape(batch_size)

            q_eval_a[key] *= mask_values
            q_eval_greedy_a[key] *= mask_values
            q_next_a[key] *= mask_values

            info.update(self.callback.on_update_agent_wise(self.iterations, key, info=info, method="update",
                                                           mask_values=mask_values, q_eval_a=q_eval_a,
                                                           q_eval_greedy_a=q_eval_greedy_a))

        if self.config.agent == "QTRAN_base":
            # -- TD Loss --
            q_joint, v_joint = self.model.Q_tran(state, hidden_state, actions, agent_mask)
            a_next_greedy = {k: v.reshape(batch_size, -1) for k, v in a_next_greedy.agent_wise.items()}
            q_joint_next, _ = self.model.Q_tran_target(state_next, hidden_state_next,
                                                       AgentGroupedTensor(a_next_greedy, self.agent_grouping),
                                                       agent_mask)

            y_dqn = rewards_tot + (1 - terminals_tot) * self.gamma * q_joint_next
            loss_td = self.mse_loss(q_joint, ops.stop_gradient(y_dqn))  # TD loss

            # -- Opt Loss --
            # Argmax across the current agents' actions
            q_tot_greedy = self.model.Q_tot(q_eval_greedy_a)
            actions_greedy = {k: v.reshape(batch_size, -1) for k, v in actions_greedy.agent_wise.items()}
            q_joint_greedy_hat, _ = self.model.Q_tran(state, hidden_state,
                                                      AgentGroupedTensor(actions_greedy, self.agent_grouping),
                                                      agent_mask)
            error_opt = q_tot_greedy - ops.stop_gradient(q_joint_greedy_hat) + v_joint
            loss_opt = ops.reduce_mean(error_opt ** 2)  # Opt loss

            # -- Nopt Loss --
            q_tot = self.model.Q_tot(q_eval_a)
            q_joint_hat = ops.stop_gradient(q_joint)
            error_nopt = q_tot - q_joint_hat + v_joint
            error_nopt = error_nopt.clamp(max=0)
            loss_nopt = ops.reduce_mean(error_nopt ** 2)  # NOPT loss

            info["Q_joint"] = q_joint.mean().item()

        elif self.config.agent == "QTRAN_alt":
            # -- TD Loss -- (Computed for all agents)
            q_count, v_joint = self.model.Q_tran(state, hidden_state, actions, agent_mask)
            actions_choosen = ops.stack([actions.agent_wise[k] for k in self.agent_keys], axis=1)
            actions_choosen = actions_choosen.reshape(-1, self.n_agents, 1)
            q_joint_choosen = q_count.gather(-1, actions_choosen.long()).reshape(-1, self.n_agents)
            q_next_count, _ = self.model.Q_tran_target(state_next, hidden_state_next, a_next_greedy, agent_mask)
            actions_next_choosen = ops.stack([a_next_greedy.agent_wise[k] for k in self.agent_keys], axis=1)
            actions_next_choosen = actions_next_choosen.reshape(-1, self.n_agents, 1)
            q_joint_next_choosen = q_next_count.gather(-1, actions_next_choosen.long()).reshape(-1, self.n_agents)

            y_dqn = rewards_tot + (1 - terminals_tot) * self.gamma * q_joint_next_choosen
            loss_td = self.mse_loss(q_joint_choosen, ops.stop_gradient(y_dqn))  # TD loss

            # -- Opt Loss -- (Computed for all agents)
            q_tot_greedy = self.model.Q_tot(q_eval_greedy_a)
            q_joint_greedy_hat, _ = self.model.Q_tran(state, hidden_state, actions_greedy, agent_mask)
            actions_greedy_current = ops.stack([actions_greedy.agent_wise[k] for k in self.agent_keys], axis=1)
            actions_greedy_current = actions_greedy_current.reshape(-1, self.n_agents, 1)
            q_joint_greedy_hat_all = q_joint_greedy_hat.gather(
                -1, actions_greedy_current.long()).reshape(-1, self.n_agents)
            error_opt = q_tot_greedy.reshape(-1, 1) - ops.stop_gradient(q_joint_greedy_hat_all) + v_joint
            loss_opt = ops.reduce_mean(error_opt ** 2)  # Opt loss

            # -- Nopt Loss --
            q_eval_count = ops.stack([q_eval.agent_wise[k] for k in self.agent_keys], axis=1)
            q_eval_count = q_eval_count.reshape(batch_size * self.n_agents, -1)
            q_sums = ops.stack([q_eval_a[k] for k in self.agent_keys], axis=1).reshape(-1, self.n_agents)
            q_sums_repeat = ops.repeat_elements(q_sums.unsqueeze(dim=1), self.n_agents, axis=1)
            agent_mask_diag = ops.repeat_elements((1 - ops.eye(self.n_agents, dtype=ms.float32)).unsqueeze(0),
                                                  batch_size, axis=0)
            q_sum_mask = (q_sums_repeat * agent_mask_diag).sum(dim=-1)
            q_count_for_nopt = q_count.view(batch_size * self.n_agents, -1)
            v_joint_repeated = ops.repeat_elements(v_joint, self.n_agents, axis=1).reshape(-1, 1)
            error_nopt = q_eval_count + q_sum_mask.view(-1, 1) - ops.stop_gradient(q_count_for_nopt) + v_joint_repeated
            error_nopt_min, _ = ops.min(error_nopt, axis=-1)
            loss_nopt = ops.reduce_mean(error_nopt_min ** 2)  # NOPT loss

            info["Q_joint"] = q_joint_choosen.mean().item()

        else:
            raise ValueError("Mixer {} not recognised.".format(self.config.agent))

        # calculate the loss function
        loss = loss_td + self.config.lambda_opt * loss_opt + self.config.lambda_nopt * loss_nopt

        return loss, info_train

    def update(self, sample):
        self.iterations += 1

        # prepare training data
        batch = self.build_training_data(
            sample=sample,
            use_actions_mask=self.use_actions_mask,
            use_global_state=True
        )

        info = self.callback.on_update_start(self.iterations, model=self.model, batch=batch)

        (loss, info_train), grads = self.grad_fn(batch)

        if self.use_grad_clip:
            grads = clip_grads(grads, Tensor(-self.grad_clip_norm), Tensor(self.grad_clip_norm))
        self.optimizer(grads)

        self.scheduler.step()
        lr = self.scheduler.get_last_lr()[0]

        info.update(info_train)
        info["learning_rate"] = lr.numpy()

        if self.iterations % self.sync_frequency == 0:
            self.model.copy_target()

        info.update(self.callback.on_update_end(self.iterations, model=self.model, info=info))

        return info
