"""
QTRAN: Learning to Factorize with Transformation for Cooperative Multi-Agent Reinforcement Learning
Paper link:
http://proceedings.mlr.press/v97/son19a/son19a.pdf
Implementation: Pytorch
"""
import torch
from torch import nn
from xuance.torch.learners import LearnerMAS
from xuance.common import List
from argparse import Namespace
from operator import itemgetter


class QTRAN_Learner(LearnerMAS):
    def __init__(self,
                 config: Namespace,
                 model_keys: List[str],
                 agent_keys: List[str],
                 policy: nn.Module):
        self.gamma = config.gamma
        self.sync_frequency = config.sync_frequency
        self.mse_loss = nn.MSELoss()
        super(QTRAN_Learner, self).__init__(config, model_keys, agent_keys, policy)
        self.optimizer = torch.optim.Adam(self.policy.parameters_model, config.learning_rate, eps=1e-5)
        self.scheduler = torch.optim.lr_scheduler.LinearLR(self.optimizer, start_factor=1.0, end_factor=0.5,
                                                           total_iters=self.config.running_steps)
        self.n_actions = {k: self.policy.action_space[k].n for k in self.model_keys}

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
            rewards_tot = torch.stack(itemgetter(*self.agent_keys)(rewards), dim=1).mean(dim=-1, keepdim=True)
            terminals_tot = torch.stack(itemgetter(*self.agent_keys)(terminals), dim=1).all(dim=1, keepdim=True).float()

        _, hidden_state, _, q_eval = self.policy(observation=obs, agent_ids=IDs, avail_actions=avail_actions)
        _, hidden_state_next, q_next = self.policy.Qtarget(observation=obs_next, agent_ids=IDs)

        q_eval_a, q_eval_greedy_a, q_next_a = {}, {}, {}
        actions_greedy, actions_next_greedy = {}, {}
        for key in self.model_keys:
            q_eval_a[key] = q_eval[key].gather(-1, actions[key].long().unsqueeze(-1)).reshape(bs)
            actions_greedy[key] = q_eval[key].argmax(dim=-1, keepdim=False)  # \bar{u}
            q_eval_greedy_a[key] = q_eval[key].gather(-1, actions_greedy[key].long().unsqueeze(-1)).reshape(bs)

            if self.use_actions_mask:
                q_next[key][avail_actions_next[key] == 0] = -9999999

            if self.config.double_q:
                _, _, act_next, _ = self.policy(observation=obs_next, agent_ids=IDs,
                                                avail_actions=avail_actions, agent_key=key)
                actions_next_greedy[key] = act_next[key]
                q_next_a[key] = q_next[key].gather(-1, act_next[key].long().unsqueeze(-1)).reshape(bs)
            else:
                actions_next_greedy[key] = q_next[key].argmax(dim=-1, keepdim=False)
                q_next_a[key] = q_next[key].max(dim=-1, keepdim=True).values.reshape(bs)

            q_eval_a[key] *= agent_mask[key]
            q_eval_greedy_a[key] *= agent_mask[key]
            q_next_a[key] *= agent_mask[key]

        # -- TD Loss --
        q_joint, v_joint = self.policy.Q_tran(hidden_state, actions, agent_mask)
        q_joint_next, _ = self.policy.Q_tran_target(hidden_state_next, actions_next_greedy, agent_mask)

        y_dqn = rewards_tot + (1 - terminals_tot) * self.gamma * q_joint_next
        loss_td = self.mse_loss(q_joint, y_dqn.detach())
        # -- TD Loss --

        # -- Opt Loss --
        q_tot_greedy = self.policy.Q_tot(q_eval_greedy_a)
        q_joint_greedy_hat, _ = self.policy.Q_tran(hidden_state, actions_greedy, agent_mask)
        error_opt = q_tot_greedy - q_joint_greedy_hat.detach() + v_joint
        loss_opt = torch.mean(error_opt ** 2)
        # -- Opt Loss --

        # -- Nopt Loss --
        if self.config.agent == "QTRAN_base":
            q_tot = self.policy.Q_tot(q_eval_a)
            q_joint_hat = q_joint
            error_nopt = q_tot - q_joint_hat.detach() + v_joint
            error_nopt = error_nopt.clamp(max=0)
            loss_nopt = torch.mean(error_nopt ** 2)  # NOPT loss
        elif self.config.agent == "QTRAN_alt":
            q_tot_counterfactual = self.policy.qtran_net.counterfactual_values(q_eval, q_eval_a) * actions_mask
            q_joint_hat_counterfactual = self.policy.qtran_net.counterfactual_values_hat(hidden_state * hidden_mask,
                                                                                         actions_onehot * actions_mask)
            error_nopt = q_tot_counterfactual - q_joint_hat_counterfactual.detach() + v_joint.unsqueeze(dim=-1).repeat(
                1, self.n_agents, self.dim_act)
            error_nopt_min = torch.min(error_nopt, dim=-1).values
            loss_nopt = torch.mean(error_nopt_min ** 2)  # NOPT loss
        else:
            raise ValueError("Mixer {} not recognised.".format(self.config.agent))
        # -- Nopt loss --

        # calculate the loss function
        loss = loss_td + self.config.lambda_opt * loss_opt + self.config.lambda_nopt * loss_nopt
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        if self.scheduler is not None:
            self.scheduler.step()

        if self.iterations % self.sync_frequency == 0:
            self.policy.copy_target()
        lr = self.optimizer.state_dict()['param_groups'][0]['lr']

        info = {
            "learning_rate": lr,
            "loss_td": loss_td.item(),
            "loss_opt": loss_opt.item(),
            "loss_nopt": loss_nopt.item(),
            "loss": loss.item(),
            "Q_joint": q_joint.mean().item()
        }

        return info
