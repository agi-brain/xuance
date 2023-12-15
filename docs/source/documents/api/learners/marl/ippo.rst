IPPO_Learner
=====================================

xxxxxx.

.. raw:: html

    <br><hr>

**PyTorch:**

.. py:class::
  xuance.torch.learners.multi_agent_rl.ippo_learner.IPPO_Learner(config, policy, optimizer, scheduler, device, model_dir, gamma)

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
  xuance.torch.learners.multi_agent_rl.ippo_learner.IPPO_Learner.lr_decay(i_step)

  xxxxxx.

  :param i_step: xxxxxx.
  :type i_step: xxxxxx
  :return: xxxxxx.
  :rtype: xxxxxx

.. py:function::
  xuance.torch.learners.multi_agent_rl.ippo_learner.IPPO_Learner.update(sample)

  xxxxxx.

  :param sample: xxxxxx.
  :type sample: xxxxxx
  :return: xxxxxx.
  :rtype: xxxxxx

.. py:function::
  xuance.torch.learners.multi_agent_rl.ippo_learner.IPPO_Learner.update_recurrent(sample)

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
  xuance.mindspore.learners.multi_agent_rl.ippo_learner.IPPO_Learner(config, policy, optimizer, scheduler, model_dir, gamma)

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
  xuance.mindspore.learners.multi_agent_rl.ippo_learner.IPPO_Learner.lr_decay(i_step)

  xxxxxx.

  :param i_step: xxxxxx.
  :type i_step: xxxxxx
  :return: xxxxxx.
  :rtype: xxxxxx

.. py:function::
  xuance.mindspore.learners.multi_agent_rl.ippo_learner.IPPO_Learner.update(sample)

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
        Independent Proximal Policy Optimization (IPPO)
        Paper link:
        https://arxiv.org/pdf/2103.01955.pdf
        Implementation: Pytorch
        """
        from xuance.torch.learners import *
        from xuance.torch.utils.value_norm import ValueNorm
        from xuance.torch.utils.operations import update_linear_decay


        class IPPO_Learner(LearnerMAS):
            def __init__(self,
                         config: Namespace,
                         policy: nn.Module,
                         optimizer: torch.optim.Optimizer,
                         scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
                         device: Optional[Union[int, str, torch.device]] = None,
                         model_dir: str = "./",
                         gamma: float = 0.99):
                self.gamma = gamma
                self.clip_range = config.clip_range
                self.use_linear_lr_decay = config.use_linear_lr_decay
                self.use_grad_norm, self.max_grad_norm = config.use_grad_norm, config.max_grad_norm
                self.use_value_clip, self.value_clip_range = config.use_value_clip, config.value_clip_range
                self.use_huber_loss, self.huber_delta = config.use_huber_loss, config.huber_delta
                self.use_value_norm = config.use_value_norm
                self.use_global_state = config.use_global_state
                self.vf_coef, self.ent_coef = config.vf_coef, config.ent_coef
                self.mse_loss = nn.MSELoss()
                self.huber_loss = nn.HuberLoss(reduction="none", delta=self.huber_delta)
                super(IPPO_Learner, self).__init__(config, policy, optimizer, scheduler, device, model_dir)
                if self.use_value_norm:
                    self.value_normalizer = ValueNorm(1).to(device)
                else:
                    self.value_normalizer = None
                self.lr = config.learning_rate
                self.end_factor_lr_decay = config.end_factor_lr_decay

            def lr_decay(self, i_step):
                if self.use_linear_lr_decay:
                    update_linear_decay(self.optimizer, i_step, self.running_steps, self.lr, self.end_factor_lr_decay)

            def update(self, sample):
                info = {}
                self.iterations += 1
                state = torch.Tensor(sample['state']).to(self.device)
                obs = torch.Tensor(sample['obs']).to(self.device)
                actions = torch.Tensor(sample['actions']).to(self.device)
                values = torch.Tensor(sample['values']).to(self.device)
                returns = torch.Tensor(sample['returns']).to(self.device)
                advantages = torch.Tensor(sample['advantages']).to(self.device)
                log_pi_old = torch.Tensor(sample['log_pi_old']).to(self.device)
                agent_mask = torch.Tensor(sample['agent_mask']).float().reshape(-1, self.n_agents, 1).to(self.device)
                batch_size = obs.shape[0]
                IDs = torch.eye(self.n_agents).unsqueeze(0).expand(batch_size, -1, -1).to(self.device)

                # actor loss
                _, pi_dist = self.policy(obs, IDs)
                log_pi = pi_dist.log_prob(actions)
                ratio = torch.exp(log_pi - log_pi_old).reshape(batch_size, self.n_agents, 1)
                advantages_mask = advantages.detach() * agent_mask
                surrogate1 = ratio * advantages_mask
                surrogate2 = torch.clip(ratio, 1 - self.clip_range, 1 + self.clip_range) * advantages_mask
                loss_a = -torch.sum(torch.min(surrogate1, surrogate2), dim=-2, keepdim=True).mean()

                # entropy loss
                entropy = pi_dist.entropy().reshape(agent_mask.shape) * agent_mask
                loss_e = entropy.mean()

                # critic loss
                _, value_pred = self.policy.get_values(obs, IDs)
                value_pred = value_pred
                value_target = returns
                if self.use_value_clip:
                    value_clipped = values + (value_pred - values).clamp(-self.value_clip_range, self.value_clip_range)
                    if self.use_huber_loss:
                        loss_v = self.huber_loss(value_pred, value_target)
                        loss_v_clipped = self.huber_loss(value_clipped, value_target)
                    else:
                        loss_v = (value_pred - value_target) ** 2
                        loss_v_clipped = (value_clipped - value_target) ** 2
                    loss_c = torch.max(loss_v, loss_v_clipped) * agent_mask
                    loss_c = loss_c.sum() / agent_mask.sum()
                else:
                    if self.use_huber_loss:
                        loss_v = self.huber_loss(value_pred, value_target) * agent_mask
                    else:
                        loss_v = ((value_pred - value_target) ** 2) * agent_mask
                    loss_c = loss_v.sum() / agent_mask.sum()

                loss = loss_a + self.vf_coef * loss_c - self.ent_coef * loss_e
                self.optimizer.zero_grad()
                loss.backward()
                if self.use_grad_norm:
                    grad_norm = torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                    info["gradient_norm"] = grad_norm.item()
                self.optimizer.step()
                if self.scheduler is not None:
                    self.scheduler.step()

                # Logger
                lr = self.optimizer.state_dict()['param_groups'][0]['lr']

                info.update({
                    "learning_rate": lr,
                    "actor_loss": loss_a.item(),
                    "critic_loss": loss_c.item(),
                    "entropy": loss_e.item(),
                    "loss": loss.item(),
                    "predict_value": value_pred.mean().item()
                })

                return info

            def update_recurrent(self, sample):
                info = {}
                self.iterations += 1
                state = torch.Tensor(sample['state']).to(self.device)
                if self.use_global_state:
                    state = state.unsqueeze(1).expand(-1, self.n_agents, -1, -1)
                obs = torch.Tensor(sample['obs']).to(self.device)
                actions = torch.Tensor(sample['actions']).to(self.device)
                values = torch.Tensor(sample['values']).to(self.device)
                returns = torch.Tensor(sample['returns']).to(self.device)
                advantages = torch.Tensor(sample['advantages']).to(self.device)
                log_pi_old = torch.Tensor(sample['log_pi_old']).to(self.device)
                avail_actions = torch.Tensor(sample['avail_actions']).float().to(self.device)
                filled = torch.Tensor(sample['filled']).float().to(self.device)
                batch_size = obs.shape[0]
                episode_length = actions.shape[2]
                IDs = torch.eye(self.n_agents).unsqueeze(1).unsqueeze(0).expand(batch_size, -1, episode_length + 1, -1).to(
                    self.device)

                # actor loss
                rnn_hidden_actor = self.policy.representation.init_hidden(batch_size * self.n_agents)
                _, pi_dist = self.policy(obs[:, :, :-1].reshape(-1, episode_length, self.dim_obs),
                                         IDs[:, :, :-1].reshape(-1, episode_length, self.n_agents),
                                         *rnn_hidden_actor,
                                         avail_actions=avail_actions[:, :, :-1].reshape(-1, episode_length, self.dim_act))
                log_pi = pi_dist.log_prob(actions.reshape(-1, episode_length)).reshape(batch_size, self.n_agents, episode_length)
                ratio = torch.exp(log_pi - log_pi_old).unsqueeze(-1)
                filled_n = filled.unsqueeze(1).expand(batch_size, self.n_agents, episode_length, 1)
                surrogate1 = ratio * advantages
                surrogate2 = torch.clip(ratio, 1 - self.clip_range, 1 + self.clip_range) * advantages
                loss_a = -(torch.min(surrogate1, surrogate2) * filled_n).sum() / filled_n.sum()

                # entropy loss
                entropy = pi_dist.entropy().reshape(batch_size, self.n_agents, episode_length, 1)
                entropy = entropy * filled_n
                loss_e = entropy.sum() / filled_n.sum()

                # critic loss
                rnn_hidden_critic = self.policy.representation_critic.init_hidden(batch_size * self.n_agents)
                if self.use_global_state:
                    _, value_pred = self.policy.get_values(state[:, :, :-1], IDs[:, :, :-1], *rnn_hidden_critic)
                else:
                    _, value_pred = self.policy.get_values(obs[:, :, :-1], IDs[:, :, :-1], *rnn_hidden_critic)
                value_target = returns.reshape(-1, 1)
                values = values.reshape(-1, 1)
                value_pred = value_pred.reshape(-1, 1)
                filled_all = filled_n.reshape(-1, 1)
                if self.use_value_clip:
                    value_clipped = values + (value_pred - values).clamp(-self.value_clip_range, self.value_clip_range)
                    if self.use_value_norm:
                        self.value_normalizer.update(value_target)
                        value_target = self.value_normalizer.normalize(value_target)
                    if self.use_huber_loss:
                        loss_v = self.huber_loss(value_pred, value_target)
                        loss_v_clipped = self.huber_loss(value_clipped, value_target)
                    else:
                        loss_v = (value_pred - value_target) ** 2
                        loss_v_clipped = (value_clipped - value_target) ** 2
                    loss_c = torch.max(loss_v, loss_v_clipped) * filled_all
                    loss_c = loss_c.sum() / filled_all.sum()
                else:
                    if self.use_value_norm:
                        self.value_normalizer.update(value_target)
                        value_pred = self.value_normalizer.normalize(value_pred)
                    if self.use_huber_loss:
                        loss_v = self.huber_loss(value_pred, value_target)
                    else:
                        loss_v = (value_pred - value_target) ** 2
                    loss_c = (loss_v * filled_all).sum() / filled_all.sum()

                loss = loss_a + self.vf_coef * loss_c - self.ent_coef * loss_e
                self.optimizer.zero_grad()
                loss.backward()
                if self.use_grad_norm:
                    grad_norm = torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                    info["gradient_norm"] = grad_norm.item()
                self.optimizer.step()
                if self.scheduler is not None:
                    self.scheduler.step()

                # Logger
                lr = self.optimizer.state_dict()['param_groups'][0]['lr']

                info.update({
                    "learning_rate": lr,
                    "actor_loss": loss_a.item(),
                    "critic_loss": loss_c.item(),
                    "entropy": loss_e.item(),
                    "loss": loss.item(),
                    "predict_value": value_pred.mean().item()
                })

                return info


  .. group-tab:: TensorFlow

    .. code-block:: python


  .. group-tab:: MindSpore

    .. code-block:: python

        """
        Multi-Agent Proximal Policy Optimization (MAPPO)
        Paper link:
        https://arxiv.org/pdf/2103.01955.pdf
        Implementation: MindSpore
        """
        from xuance.mindspore.learners import *
        from xuance.mindspore.utils.operations import update_linear_decay


        class IPPO_Learner(LearnerMAS):
            class PolicyNetWithLossCell(nn.Cell):
                def __init__(self, backbone, n_agents, vf_coef, ent_coef, clip_range, use_value_clip, value_clip_range,
                             use_huber_loss):
                    super(IPPO_Learner.PolicyNetWithLossCell, self).__init__()
                    self._backbone = backbone
                    self.n_agents = n_agents
                    self.vf_coef = vf_coef
                    self.ent_coef = ent_coef
                    self.clip_range = clip_range * 0.5
                    self.use_value_clip = use_value_clip
                    self.value_clip_range = Tensor(value_clip_range)
                    self.use_huber_loss = use_huber_loss
                    self.mse_loss = nn.MSELoss()
                    self.huber_loss = nn.HuberLoss()
                    self.exp = ops.Exp()
                    self.miminum = ops.Minimum()
                    self.maximum = ops.Maximum()
                    self.expand_dims = ops.ExpandDims()
                    self.broadcast_to = ops.BroadcastTo((-1, self.n_agents, -1))

                def construct(self, bs, s, o, a, log_pi_old, values, returns, advantages, agt_mask, ids):
                    _, act_prob = self._backbone(o, ids)
                    log_pi = self._backbone.actor.log_prob(value=a, probs=act_prob)
                    ratio = self.exp(log_pi - log_pi_old).view(bs, self.n_agents, 1)
                    advantages_mask = advantages * agt_mask
                    surrogate1 = ratio * advantages_mask
                    surrogate2 = ops.clip_by_value(ratio, Tensor(1 - self.clip_range), Tensor(1 + self.clip_range)) * advantages_mask
                    loss_a = -self.miminum(surrogate1, surrogate2).sum(axis=-2, keepdims=True).mean()

                    entropy = self._backbone.actor.entropy(probs=act_prob).reshape(agt_mask.shape) * agt_mask
                    loss_e = entropy.mean()

                    _, value_pred = self._backbone.get_values(o, ids)
                    value_pred = value_pred * agt_mask
                    value_target = returns
                    if self.use_value_clip:
                        value_clipped = values + ops.clip_by_value(value_pred - values, -self.value_clip_range, self.value_clip_range)
                        if self.use_huber_loss:
                            loss_v = self.huber_loss(logits=value_pred, labels=value_target)
                            loss_v_clipped = self.huber_loss(logits=value_clipped, labels=value_target)
                        else:
                            loss_v = (value_pred - value_target) ** 2
                            loss_v_clipped = (value_clipped - value_target) ** 2
                        loss_c = self.maximum(loss_v, loss_v_clipped) * agt_mask
                        loss_c = loss_c.sum() / agt_mask.sum()
                    else:
                        if self.use_huber_loss:
                            loss_v = self.huber_loss(logits=value_pred, labels=value_target) * agt_mask
                        else:
                            loss_v = ((value_pred - value_target) ** 2) * agt_mask
                        loss_c = loss_v.sum() / agt_mask.sum()

                    loss = loss_a + self.vf_coef * loss_c - self.ent_coef * loss_e
                    return loss

            def __init__(self,
                         config: Namespace,
                         policy: nn.Cell,
                         optimizer: nn.Optimizer,
                         scheduler: Optional[nn.exponential_decay_lr] = None,
                         model_dir: str = "./",
                         gamma: float = 0.99,
                         ):
                self.gamma = gamma
                self.clip_range = config.clip_range
                self.use_linear_lr_decay = config.use_linear_lr_decay
                self.use_grad_norm, self.max_grad_norm = config.use_grad_norm, config.max_grad_norm
                self.use_value_clip, self.value_clip_range = config.use_value_clip, config.value_clip_range
                self.use_huber_loss, self.huber_delta = config.use_huber_loss, config.huber_delta
                self.use_value_norm = config.use_value_norm
                self.use_global_state = config.use_global_state
                self.vf_coef, self.ent_coef = config.vf_coef, config.ent_coef
                self.mse_loss = nn.MSELoss()
                super(IPPO_Learner, self).__init__(config, policy, optimizer, scheduler, model_dir)
                # define mindspore trainers
                self.loss_net = self.PolicyNetWithLossCell(policy, self.n_agents, config.vf_coef, config.ent_coef,
                                                           config.clip_range, config.use_value_clip, config.value_clip_range,
                                                           config.use_huber_loss)
                if self.args.use_grad_norm:
                    self.policy_train = TrainOneStepCellWithGradClip(self.loss_net, self.optimizer,
                                                                     clip_type=config.clip_type, clip_value=config.max_grad_norm)
                else:
                    self.policy_train = nn.TrainOneStepCell(self.loss_net, self.optimizer)
                self.lr = config.learning_rate
                self.end_factor_lr_decay = config.end_factor_lr_decay

            def lr_decay(self, i_step):
                if self.use_linear_lr_decay:
                    update_linear_decay(self.optimizer, i_step, self.running_steps, self.lr, self.end_factor_lr_decay)

            def update(self, sample):
                self.iterations += 1
                state = Tensor(sample['state'])
                obs = Tensor(sample['obs'])
                actions = Tensor(sample['actions'])
                values = Tensor(sample['values'])
                returns = Tensor(sample['returns'])
                advantages = Tensor(sample['advantages'])
                log_pi_old = Tensor(sample['log_pi_old'])
                agent_mask = Tensor(sample['agent_mask']).view(-1, self.n_agents, 1)
                batch_size = obs.shape[0]
                IDs = ops.broadcast_to(self.expand_dims(self.eye(self.n_agents, self.n_agents, ms.float32), 0),
                                       (batch_size, -1, -1))

                loss = self.policy_train(batch_size, state, obs, actions, log_pi_old, values, returns, advantages, agent_mask, IDs)

                # Logger
                lr = self.scheduler(self.iterations).asnumpy()

                info = {
                    "learning_rate": lr,
                    "loss": loss.asnumpy()
                }

                return info

