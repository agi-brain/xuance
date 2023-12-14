"""
QTRAN: Learning to Factorize with Transformation for Cooperative Multi-Agent Reinforcement Learning
Paper link:
http://proceedings.mlr.press/v97/son19a/son19a.pdf
Implementation: MindSpore
"""
from xuance.mindspore.learners import *


class QTRAN_Learner(LearnerMAS):
    class PolicyNetWithLossCell(nn.Cell):
        def __init__(self, backbone, dim_act, n_agents, agent_name, lambda_opt, lambda_nopt):
            super(QTRAN_Learner.PolicyNetWithLossCell, self).__init__(auto_prefix=False)
            self._backbone = backbone
            self.dim_act = dim_act
            self.n_agents = n_agents
            self.agent = agent_name
            self._lambda_opt = lambda_opt
            self._lambda_nopt = lambda_nopt

            self._expand_dims = ops.ExpandDims()
            self._onehot = ms.ops.OneHot()

        def construct(self, o, ids, a, a_onehot, agt_mask, act_mask, hidden_mask, y_dqn):
            _, hidden_state, _, q_eval = self._backbone(o, ids)
            q_joint, v_joint = self._backbone.qtran_net(hidden_state * hidden_mask,
                                                        a_onehot * act_mask)
            loss_td = ((q_joint - y_dqn) ** 2).sum() / agt_mask.sum()

            action_greedy = q_eval.argmax(axis=-1).astype(ms.int32)  # \bar{u}
            q_eval_greedy_a = GatherD()(q_eval, -1, action_greedy.view(-1, self.n_agents, 1))
            q_tot_greedy = self._backbone.q_tot(q_eval_greedy_a * agt_mask)
            q_joint_greedy_hat, _ = self._backbone.qtran_net(hidden_state * hidden_mask,
                                                             self._onehot(action_greedy, self.dim_act,
                                                                          ms.Tensor(1.0, ms.float32),
                                                                          ms.Tensor(0.0, ms.float32)) * act_mask)
            error_opt = q_tot_greedy - q_joint_greedy_hat + v_joint
            loss_opt = (error_opt ** 2).mean()

            q_eval_a = GatherD()(q_eval, -1, a)
            if self.agent == "QTRAN_base":
                q_tot = self._backbone.q_tot(q_eval_a * agt_mask)
                q_joint_hat, _ = self._backbone.qtran_net(hidden_state * hidden_mask, a_onehot * act_mask)
                error_nopt = q_tot - q_joint_hat + v_joint
                error_nopt = ops.clip_by_value(error_nopt, clip_value_max=ms.Tensor(0.0, ms.float32))
                loss_nopt = (error_nopt ** 2).mean()
            elif self.agent == "QTRAN_alt":
                q_tot_counterfactual = self._backbone.qtran_net.counterfactual_values(q_eval, q_eval_a) * act_mask
                q_joint_hat_counterfactual = self._backbone.qtran_net.counterfactual_values_hat(
                    hidden_state * hidden_mask, a_onehot * act_mask)
                error_nopt = q_tot_counterfactual - q_joint_hat_counterfactual + ops.broadcast_to(
                    self._expand_dims(v_joint, -1), (-1, -1, self.dim_act))
                error_nopt_min = error_nopt.min(axis=-1)
                loss_nopt = (error_nopt_min ** 2).mean()
            else:
                raise ValueError("Mixer {} not recognised.".format(self.args.agent))

            loss = loss_td + self._lambda_opt * loss_opt + self._lambda_nopt * loss_nopt
            return loss

    def __init__(self,
                 config: Namespace,
                 policy: nn.Cell,
                 optimizer: nn.Optimizer,
                 scheduler: Optional[nn.exponential_decay_lr] = None,
                 model_dir: str = "./",
                 gamma: float = 0.99,
                 sync_frequency: int = 100
                 ):
        self.gamma = gamma
        self.sync_frequency = sync_frequency
        self.mse_loss = nn.MSELoss()
        super(QTRAN_Learner, self).__init__(config, policy, optimizer, scheduler, model_dir)
        self._mean = ops.ReduceMean(keep_dims=False)
        self.loss_net = self.PolicyNetWithLossCell(policy, self.dim_act, self.n_agents, self.args.agent,
                                                   self.args.lambda_opt, self.args.lambda_nopt)
        self.policy_train = nn.TrainOneStepCell(self.loss_net, optimizer)
        self.policy_train.set_train()

    def update(self, sample):
        self.iterations += 1
        obs = Tensor(sample['obs'])
        actions = Tensor(sample['actions'])
        actions_onehot = self.onehot_action(actions, self.dim_act)
        actions = actions.view(-1, self.n_agents, 1).astype(ms.int32)
        obs_next = Tensor(sample['obs_next'])
        rewards = self._mean(Tensor(sample['rewards']), 1)
        terminals = Tensor(sample['terminals']).view(-1, self.n_agents, 1).all(axis=1, keep_dims=True).astype(ms.float32)
        agent_mask = Tensor(sample['agent_mask']).view(-1, self.n_agents, 1)
        batch_size = obs.shape[0]
        IDs = ops.broadcast_to(self.expand_dims(self.eye(self.n_agents, self.n_agents, ms.float32), 0),
                               (batch_size, -1, -1))

        actions_mask = ops.broadcast_to(agent_mask, (-1, -1, int(self.dim_act)))
        hidden_mask = ops.broadcast_to(agent_mask, (-1, -1, self.policy.representation_info_shape['state'][0]))

        _, hidden_state_next, q_next_eval = self.policy.target_Q(obs_next.view(batch_size, self.n_agents, -1), IDs)
        if self.args.double_q:
            _, _, actions_next_greedy, _ = self.policy(obs_next, IDs)
        else:
            actions_next_greedy = q_next_eval.argmax(axis=-1, keepdim=False)
        q_joint_next, _ = self.policy.target_qtran_net(hidden_state_next * hidden_mask,
                                                       self.onehot_action(actions_next_greedy,
                                                                          self.dim_act) * actions_mask)
        y_dqn = rewards + (1 - terminals) * self.args.gamma * q_joint_next

        # calculate the loss function
        loss = self.policy_train(obs, IDs, actions, actions_onehot, agent_mask, actions_mask, hidden_mask, y_dqn)
        if self.iterations % self.sync_frequency == 0:
            self.policy.copy_target()

        lr = self.scheduler(self.iterations).asnumpy()

        info = {
            "learning_rate": lr,
            "loss": loss.asnumpy()
        }

        return info
