"""
QTRAN: Learning to Factorize with Transformation for Cooperative Multi-Agent Reinforcement Learning
Paper link:
http://proceedings.mlr.press/v97/son19a/son19a.pdf
Implementation: MindSpore
"""
from argparse import Namespace
from operator import itemgetter
from xuance.common import List
from xuance.mindspore import ms, Module, nn, Tensor, optim, ops
from xuance.mindspore.learners import LearnerMAS
from xuance.mindspore.utils import clip_grads


class QTRAN_Learner(LearnerMAS):
    def __init__(self,
                 config: Namespace,
                 model_keys: List[str],
                 agent_keys: List[str],
                 policy: Module):
        self.sync_frequency = config.sync_frequency
        self.mse_loss = nn.MSELoss()
        super(QTRAN_Learner, self).__init__(config, model_keys, agent_keys, policy)
        self.optimizer = optim.Adam(params=self.policy.trainable_params(), lr=config.learning_rate, eps=1e-5)
        self.scheduler = optim.lr_scheduler.LinearLR(self.optimizer, start_factor=1.0,
                                                     end_factor=self.end_factor_lr_decay,
                                                     total_iters=self.config.running_steps)
        self.n_actions = {k: self.policy.action_space[k].n for k in self.model_keys}
        # Get gradient function
        self.grad_fn = ms.value_and_grad(self.forward_fn, None, self.optimizer.parameters, has_aux=True)
        self.policy.set_train()

    def forward_fn(self, bs, batch_size, state, obs, actions, state_next, obs_next,
                   agent_mask, avail_actions, avail_actions_next,
                   rewards_tot, terminals_tot, IDs):
        info = {}
        _, hidden_state, actions_greedy, q_eval = self.policy(obs, agent_ids=IDs, avail_actions=avail_actions)
        _, hidden_state_next, q_next = self.policy.Qtarget(obs_next, agent_ids=IDs)

        q_eval_a, q_eval_greedy_a, q_next_a = {}, {}, {}
        actions_next_greedy = {}
        for key in self.model_keys:
            mask_values = agent_mask[key]
            q_eval_a[key] = q_eval[key].gather(-1, actions[key].long().unsqueeze(-1)).reshape(bs)
            q_eval_greedy_a[key] = q_eval[key].gather(-1, actions_greedy[key].long().unsqueeze(-1)).reshape(bs)

            if self.use_actions_mask:
                q_next[key][avail_actions_next[key] == 0] = -1e10

            if self.config.double_q:
                _, _, act_next, _ = self.policy(observation=obs_next, agent_ids=IDs,
                                                avail_actions=avail_actions, agent_key=key)
                actions_next_greedy[key] = act_next[key]
                q_next_a[key] = q_next[key].gather(-1, act_next[key].long().unsqueeze(-1)).reshape(bs)
            else:
                actions_next_greedy[key] = q_next[key].argmax(dim=-1, keepdim=False)
                q_next_a[key] = q_next[key].max(dim=-1, keepdim=True).values.reshape(bs)

            q_eval_a[key] *= mask_values
            q_eval_greedy_a[key] *= mask_values
            q_next_a[key] *= mask_values

            # info.update(self.callback.on_update_agent_wise(self.iterations, key, info=info, method="update",
            #                                                mask_values=mask_values, q_eval_a=q_eval_a,
            #                                                q_eval_greedy_a=q_eval_greedy_a))

        if self.config.agent == "QTRAN_base":
            # -- TD Loss --
            q_joint, v_joint = self.policy.Q_tran(state, hidden_state, actions, agent_mask)
            q_joint_next, _ = self.policy.Q_tran_target(state_next, hidden_state_next, actions_next_greedy, agent_mask)

            y_dqn = rewards_tot + (1 - terminals_tot) * self.gamma * q_joint_next
            loss_td = self.mse_loss(q_joint, ops.stop_gradient(y_dqn))  # TD loss

            # -- Opt Loss --
            # Argmax across the current agents' actions
            q_tot_greedy = self.policy.Q_tot(q_eval_greedy_a)
            q_joint_greedy_hat, _ = self.policy.Q_tran(state, hidden_state, actions_greedy, agent_mask)
            error_opt = q_tot_greedy - ops.stop_gradient(q_joint_greedy_hat) + v_joint
            loss_opt = ops.mean(error_opt ** 2)  # Opt loss

            # -- Nopt Loss --
            q_tot = self.policy.Q_tot(q_eval_a)
            q_joint_hat = q_joint
            error_nopt = q_tot - ops.stop_gradient(q_joint_hat) + v_joint
            error_nopt = error_nopt.clamp(max=0)
            loss_nopt = ops.mean(error_nopt ** 2)  # NOPT loss

            info["Q_joint"] = q_joint.mean().asnumpy()

        elif self.config.agent == "QTRAN_alt":
            # -- TD Loss -- (Computed for all agents)
            q_count, v_joint = self.policy.Q_tran(state, hidden_state, actions, agent_mask)
            actions_choosen = itemgetter(*self.model_keys)(actions)
            actions_choosen = actions_choosen.reshape(-1, self.n_agents, 1)
            q_joint_choosen = q_count.gather(-1, actions_choosen.long()).reshape(-1, self.n_agents)
            q_next_count, _ = self.policy.Q_tran_target(state_next, hidden_state_next, actions_next_greedy, agent_mask)
            actions_next_choosen = itemgetter(*self.model_keys)(actions_next_greedy)
            actions_next_choosen = actions_next_choosen.reshape(-1, self.n_agents, 1)
            q_joint_next_choosen = q_next_count.gather(-1, actions_next_choosen.long()).reshape(-1, self.n_agents)

            y_dqn = rewards_tot + (1 - terminals_tot) * self.gamma * q_joint_next_choosen
            loss_td = self.mse_loss(q_joint_choosen, ops.stop_gradient(y_dqn))  # TD loss

            # -- Opt Loss -- (Computed for all agents)
            q_tot_greedy = self.policy.Q_tot(q_eval_greedy_a)
            q_joint_greedy_hat, _ = self.policy.Q_tran(state, hidden_state, actions_greedy, agent_mask)
            actions_greedy_current = itemgetter(*self.model_keys)(actions_greedy)
            actions_greedy_current = actions_greedy_current.reshape(-1, self.n_agents, 1)
            q_joint_greedy_hat_all = q_joint_greedy_hat.gather(
                -1, actions_greedy_current.long()).reshape(-1, self.n_agents)
            error_opt = q_tot_greedy - ops.stop_gradient(q_joint_greedy_hat_all) + v_joint
            loss_opt = ops.mean(error_opt ** 2)  # Opt loss

            # -- Nopt Loss --
            q_eval_count = itemgetter(*self.model_keys)(q_eval).reshape(batch_size * self.n_agents, -1)
            q_sums = itemgetter(*self.model_keys)(q_eval_a).reshape(-1, self.n_agents)
            q_sums_repeat = q_sums.unsqueeze(dim=1).repeat(1, self.n_agents, 1)
            agent_mask_diag = ops.repeat_elements((1 - ops.eye(self.n_agents, dtype=ms.float32)).unsqueeze(0),
                                                  rep=batch_size, axis=0)
            q_sum_mask = (q_sums_repeat * agent_mask_diag).sum(axis=-1)
            q_count_for_nopt = q_count.view(batch_size * self.n_agents, -1)
            v_joint_repeated = v_joint.repeat(1, self.n_agents).view(-1, 1)
            error_nopt = q_eval_count + q_sum_mask.view(-1, 1) - ops.stop_gradient(q_count_for_nopt) + v_joint_repeated
            error_nopt_min = ops.min(error_nopt, axis=-1)
            loss_nopt = ops.mean(error_nopt_min ** 2)  # NOPT loss

            info["Q_joint"] = q_joint_choosen.mean().asnumpy()

        else:
            raise ValueError("Mixer {} not recognised.".format(self.config.agent))

        # calculate the loss function
        loss = loss_td + self.config.lambda_opt * loss_opt + self.config.lambda_nopt * loss_nopt

        return loss, loss_td, loss_opt, loss_nopt

    def update(self, sample):
        self.iterations += 1
        info = {}

        # prepare training data
        sample_Tensor = self.build_training_data(sample=sample,
                                                 use_parameter_sharing=self.use_parameter_sharing,
                                                 use_actions_mask=self.use_actions_mask,
                                                 use_global_state=True)
        batch_size = sample_Tensor['batch_size']
        state = sample_Tensor['state']
        state_next = sample_Tensor['state_next']
        obs = sample_Tensor['obs']
        actions = sample_Tensor['actions']
        obs_next = sample_Tensor['obs_next']
        rewards = sample_Tensor['rewards']
        terminals = sample_Tensor['terminals']
        agent_mask = sample_Tensor['agent_mask']
        avail_actions = sample_Tensor['avail_actions']
        avail_actions_next = sample_Tensor['avail_actions_next']
        IDs = sample_Tensor['agent_ids']

        if self.use_parameter_sharing:
            key = self.model_keys[0]
            bs = batch_size * self.n_agents
            rewards_tot = rewards[key].mean(dim=1).reshape(batch_size, 1)
            terminals_tot = terminals[key].all(dim=1, keepdim=False).float().reshape(batch_size, 1)
        else:
            bs = batch_size
            rewards_tot = ops.stack(itemgetter(*self.agent_keys)(rewards), axis=1).mean(dim=-1, keepdim=True)
            terminals_tot = ops.stack(itemgetter(*self.agent_keys)(terminals), axis=1).all(dim=1, keepdim=True).float()

        # info = self.callback.on_update_start(self.iterations, method="update",
        #                                      policy=self.policy, sample_Tensor=sample_Tensor, bs=bs,
        #                                      rewards_tot=rewards_tot, terminals_tot=terminals_tot)

        (loss, loss_td, loss_opt, loss_nopt), grads = self.grad_fn(bs, batch_size, state, obs, actions,
                                                                   state_next, obs_next, agent_mask, avail_actions,
                                                                   avail_actions_next, rewards_tot, terminals_tot, IDs)
        if self.use_grad_clip:
            grads = clip_grads(grads, Tensor(-self.grad_clip_norm), Tensor(self.grad_clip_norm))
        self.optimizer(grads)

        if self.iterations % self.sync_frequency == 0:
            self.policy.copy_target()

        self.scheduler.step()
        lr = self.scheduler.get_last_lr()[0]

        info.update({
            "learning_rate": lr.asnumpy(),
            "loss_td": loss_td.asnumpy(),
            "loss_opt": loss_opt.asnumpy(),
            "loss_nopt": loss_nopt.asnumpy(),
            "loss": loss.asnumpy()
        })

        # info.update(self.callback.on_update_end(self.iterations, method="update", policy=self.policy, info=info,
        #                                         v_joint=v_joint, y_dqn=y_dqn, q_tot_greedy=q_tot_greedy,
        #                                         q_joint_greedy_hat=q_joint_greedy_hat, error_opt=error_opt,
        #                                         error_nopt=error_nopt))

        return info
