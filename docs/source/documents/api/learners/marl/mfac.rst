MFAC_Learner
======================

xxxxxx.

.. raw:: html

    <br><hr>

**PyTorch:**

.. py:class::
  xuance.torch.learners.multi_agent_rl.mfac_learner.MFAC_Learner(config, policy, optimizer, scheduler, device, model_dir, gamma)

  :param config: xxxxxx.
  :type config: xxxxxx
  :param policy: xxxxxx.
  :type policy: xxxxxx
  :param optimizer: xxxxxx.
  :type optimizer: xxxxxx
  :param scheduler: xxxxxx.
  :type scheduler: xxxxxx
  :param device: xxxxxx.
  :type device: xxxxxx
  :param model_dir: xxxxxx.
  :type model_dir: xxxxxx
  :param gamma: xxxxxx.
  :type gamma: xxxxxx

.. py:function::
  xuance.torch.learners.multi_agent_rl.mfac_learner.MFAC_Learner.update(sample)

  xxxxxx.

  :param sample: xxxxxx.
  :type sample: xxxxxx
  :return: xxxxxx.
  :rtype: xxxxxx

.. raw:: html

    <br><hr>

**TensorFlow:**

.. raw:: html

    <br><hr>

**MindSpore:**

.. py:class::
  xuance.mindspore.learners.multi_agent_rl.mfac_learner.MFAC_Learner(config, policy, optimizer, scheduler, model_dir, gamma)

  :param config: xxxxxx.
  :type config: xxxxxx
  :param policy: xxxxxx.
  :type policy: xxxxxx
  :param optimizer: xxxxxx.
  :type optimizer: xxxxxx
  :param scheduler: xxxxxx.
  :type scheduler: xxxxxx
  :param model_dir: xxxxxx.
  :type model_dir: xxxxxx
  :param gamma: xxxxxx.
  :type gamma: xxxxxx

.. py:function::
  xuance.mindspore.learners.multi_agent_rl.mfac_learner.MFAC_Learner.update(sample)

  xxxxxx.

  :param sample: xxxxxx.
  :type sample: xxxxxx
  :return: xxxxxx.
  :rtype: xxxxxx

.. raw:: html

    <br><hr>

Source Code
-----------------

.. tabs::

  .. group-tab:: PyTorch

    .. code-block:: python

        """
        MFAC: Mean Field Actor-Critic
        Paper link:
        http://proceedings.mlr.press/v80/yang18d/yang18d.pdf
        Implementation: Pytorch
        """
        import torch

        from xuance.torch.learners import *


        class MFAC_Learner(LearnerMAS):
            def __init__(self,
                         config: Namespace,
                         policy: nn.Module,
                         optimizer: Sequence[torch.optim.Optimizer],
                         scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
                         device: Optional[Union[int, str, torch.device]] = None,
                         model_dir: str = "./",
                         gamma: float = 0.99,
                         ):
                self.gamma = gamma
                self.tau = config.tau
                self.mse_loss = nn.MSELoss()
                super(MFAC_Learner, self).__init__(config, policy, optimizer, scheduler, device, model_dir)
                self.optimizer = {
                    'actor': optimizer[0],
                    'critic': optimizer[1]
                }
                self.scheduler = {
                    'actor': scheduler[0],
                    'critic': scheduler[1]
                }

            def update(self, sample):
                self.iterations += 1
                obs = torch.Tensor(sample['obs']).to(self.device)
                actions = torch.Tensor(sample['actions']).to(self.device)
                obs_next = torch.Tensor(sample['obs_next']).to(self.device)
                act_mean = torch.Tensor(sample['act_mean']).to(self.device)
                # act_mean_next = torch.Tensor(sample['act_mean_next']).to(self.device)
                rewards = torch.Tensor(sample['rewards']).to(self.device)
                terminals = torch.Tensor(sample['terminals']).float().reshape(-1, self.n_agents, 1).to(self.device)
                agent_mask = torch.Tensor(sample['agent_mask']).float().reshape(-1, self.n_agents, 1).to(self.device)
                batch_size = obs.shape[0]
                IDs = torch.eye(self.n_agents).unsqueeze(0).expand(batch_size, -1, -1).to(self.device)

                act_mean_n = act_mean.unsqueeze(1).repeat([1, self.n_agents, 1])

                # train critic network
                target_pi_dist_next = self.policy.target_actor(obs_next, IDs)
                target_pi_next = target_pi_dist_next.logits.softmax(dim=-1)
                actions_next = target_pi_dist_next.stochastic_sample()
                actions_next_onehot = self.onehot_action(actions_next, self.dim_act).type(torch.float)
                act_mean_next = actions_next_onehot.mean(dim=-2, keepdim=False)
                act_mean_n_next = act_mean_next.unsqueeze(1).repeat([1, self.n_agents, 1])

                q_eval = self.policy.critic(obs, act_mean_n, IDs)
                q_eval_a = q_eval.gather(-1, actions.long().reshape([batch_size, self.n_agents, 1]))

                q_eval_next = self.policy.target_critic(obs_next, act_mean_n_next, IDs)
                shape = q_eval_next.shape
                v_mf = torch.bmm(q_eval_next.reshape(-1, 1, shape[-1]), target_pi_next.reshape(-1, shape[-1], 1))
                v_mf = v_mf.reshape(*(list(shape[0:-1]) + [1]))
                q_target = rewards + (1 - terminals) * self.args.gamma * v_mf
                td_error = (q_eval_a - q_target.detach()) * agent_mask
                loss_c = (td_error ** 2).sum() / agent_mask.sum()
                self.optimizer["critic"].zero_grad()
                loss_c.backward()
                self.optimizer["critic"].step()
                if self.scheduler['critic'] is not None:
                    self.scheduler['critic'].step()

                # train actor network
                _, pi_dist = self.policy(obs, IDs)
                actions_ = pi_dist.stochastic_sample()
                advantages = self.policy.target_critic(obs, act_mean_n, IDs)
                advantages = advantages.gather(-1, actions_.long().reshape([batch_size, self.n_agents, 1]))
                log_pi_prob = pi_dist.log_prob(actions_).unsqueeze(-1)
                advantages = log_pi_prob * advantages.detach()
                loss_a = -(advantages.sum() / agent_mask.sum())
                self.optimizer["actor"].zero_grad()
                loss_a.backward()
                grad_norm_actor = torch.nn.utils.clip_grad_norm_(self.policy.parameters_actor, self.args.clip_grad)
                self.optimizer["actor"].step()
                if self.scheduler['actor'] is not None:
                    self.scheduler['actor'].step()

                self.policy.soft_update(self.tau)
                # Logger
                lr_a = self.optimizer['actor'].state_dict()['param_groups'][0]['lr']
                lr_c = self.optimizer['critic'].state_dict()['param_groups'][0]['lr']

                info = {
                    "learning_rate_actor": lr_a,
                    "learning_rate_critic": lr_c,
                    "actor_loss": loss_a.item(),
                    "critic_loss": loss_c.item(),
                    "actor_gradient_norm": grad_norm_actor.item()
                }

                return info


  .. group-tab:: TensorFlow

    .. code-block:: python


  .. group-tab:: MindSpore

    .. code-block:: python

        """
        MFAC: Mean Field Actor-Critic
        Paper link:
        http://proceedings.mlr.press/v80/yang18d/yang18d.pdf
        Implementation: MindSpore
        """
        from xuance.mindspore.learners import *


        class MFAC_Learner(LearnerMAS):
            class NetWithLossCell(nn.Cell):
                def __init__(self, backbone, vf_coef, ent_coef):
                    super(MFAC_Learner.NetWithLossCell, self).__init__()
                    self._backbone = backbone
                    self.vf_coef = vf_coef
                    self.ent_coef = ent_coef

                def construct(self, obs, actions, returns, advantages, act_mean_n, agt_mask, ids):
                    # actor loss
                    _, act_probs = self._backbone(obs, ids)
                    log_pi = self._backbone.actor.log_prob(value=actions, probs=act_probs).unsqueeze(-1)
                    entropy = self._backbone.actor.entropy(act_probs).unsqueeze(-1)

                    targets = returns
                    value_pred = self._backbone.get_values(obs, act_mean_n, ids)
                    td_error = value_pred - targets

                    pg_loss = -((advantages * log_pi) * agt_mask).sum() / agt_mask.sum()
                    vf_loss = ((td_error ** 2) * agt_mask).sum() / agt_mask.sum()
                    entropy_loss = (entropy * agt_mask).sum() / agt_mask.sum()
                    loss = pg_loss + self.vf_coef * vf_loss - self.ent_coef * entropy_loss

                    return loss

            def __init__(self,
                         config: Namespace,
                         policy: nn.Cell,
                         optimizer: Sequence[nn.Optimizer],
                         scheduler: Optional[nn.exponential_decay_lr] = None,
                         model_dir: str = "./",
                         gamma: float = 0.99,
                         ):
                self.gamma = gamma
                self.clip_range = config.clip_range
                self.use_linear_lr_decay = config.use_linear_lr_decay
                self.use_grad_norm, self.max_grad_norm = config.use_grad_norm, config.max_grad_norm
                self.use_value_norm = config.use_value_norm
                self.vf_coef, self.ent_coef = config.vf_coef, config.ent_coef
                self.tau = config.tau
                self.mse_loss = nn.MSELoss()
                super(MFAC_Learner, self).__init__(config, policy, optimizer, scheduler, model_dir)
                self.optimizer = optimizer
                self.scheduler = scheduler
                self.bmm = ops.BatchMatMul()
                self.loss_net = self.NetWithLossCell(policy, self.vf_coef, self.ent_coef)
                self.policy_train = TrainOneStepCellWithGradClip(self.loss_net, self.optimizer,
                                                                 clip_type=config.clip_type, clip_value=config.max_grad_norm)
                self.policy_train.set_train()

            def update(self, sample):
                self.iterations += 1
                obs = Tensor(sample['obs'])
                actions = Tensor(sample['actions'])
                act_mean = Tensor(sample['act_mean'])
                returns = Tensor(sample['returns'])
                agent_mask = Tensor(sample['agent_mask']).astype(ms.float32).view(-1, self.n_agents, 1)
                batch_size = obs.shape[0]
                IDs = ops.broadcast_to(self.expand_dims(self.eye(self.n_agents, self.n_agents, ms.float32), 0),
                                       (batch_size, -1, -1))

                act_mean_n = ops.broadcast_to(self.expand_dims(act_mean, 1), (-1, self.n_agents, -1))

                targets = returns
                value_pred = self.policy.get_values(obs, act_mean_n, IDs)
                advantages = targets - value_pred
                loss = self.policy_train(obs, actions, returns, advantages, act_mean_n, agent_mask, IDs)

                lr = self.scheduler(self.iterations)

                info = {
                    "learning_rate": lr.asnumpy(),
                    "loss": loss.asnumpy()
                }

                return info

