MAPPO_Learner
=====================================

A learner class for the Multi-Agent Proximal Policy Optimization (MAPPO) algorithm implemented.

.. raw:: html

    <br><hr>

PyTorch
------------------------------------------

.. py:class::
  xuance.torch.learners.multi_agent_rl.mappo_learner.MAPPO_Clip_Learner(config, policy, optimizer, scheduler, device, model_dir, gamma)

  :param config: Provides hyper parameters.
  :type config: Namespace
  :param policy: The policy that provides actions and values.
  :type policy: nn.Module
  :param optimizer: The optimizer that update the parameters of the model.
  :type optimizer: Optimizer
  :param scheduler: The tool for learning rate decay.
  :type scheduler: lr_scheduler
  :param device: The calculating device.
  :type device: str
  :param model_dir: The directory for saving or loading the model parameters.
  :type model_dir: str
  :param gamma: The discount factor.
  :type gamma: float

.. py:function::
  xuance.torch.learners.multi_agent_rl.mappo_learner.MAPPO_Clip_Learner.lr_decay(i_step)

  Linearly decay the learning rate based on the current training step.

  :param i_step: The current step.
  :type i_step: int
  :return: Current learning rate.
  :rtype: float

.. py:function::
  xuance.torch.learners.multi_agent_rl.mappo_learner.MAPPO_Clip_Learner.update(sample)

  Update the parameters of the model.

  :param sample: A dictionary containing necessary experience data that is sampled from experience replay buffer.
  :type sample: dict
  :return: The information of the training.
  :rtype: dict

.. py:function::
  xuance.torch.learners.multi_agent_rl.mappo_learner.MAPPO_Clip_Learner.update_recurrent(sample)

  Update the parameters of the model with recurrent neural networks.

  :param sample: A dictionary containing necessary experience data that is sampled from experience replay buffer.
  :type sample: dict
  :return: The information of the training.
  :rtype: dict

.. raw:: html

    <br><hr>

TensorFlow
------------------------------------------

.. py:class::
  xuance.tensorflow.learners.multi_agent_rl.mappo_learner.MAPPO_Clip_Learner(config, policy, optimizer, device, model_dir, gamma)

  :param config: Provides hyper parameters.
  :type config: Namespace
  :param policy: The policy that provides actions and values.
  :type policy: nn.Module
  :param optimizer: The optimizer that update the parameters of the model.
  :type optimizer: Optimizer
  :param device: The calculating device.
  :type device: str
  :param model_dir: The directory for saving or loading the model parameters.
  :type model_dir: str
  :param gamma: The discount factor.
  :type gamma: float

.. py:function::
  xuance.tensorflow.learners.multi_agent_rl.mappo_learner.MAPPO_Clip_Learner.lr_decay(i_step)

  Linearly decay the learning rate based on the current training step.

  :param i_step: The current step.
  :type i_step: int
  :return: Current learning rate.
  :rtype: float

.. py:function::
  xuance.tensorflow.learners.multi_agent_rl.mappo_learner.MAPPO_Clip_Learner.update(sample)

  Update the parameters of the model.

  :param sample: A dictionary containing necessary experience data that is sampled from experience replay buffer.
  :type sample: dict
  :return: The information of the training.
  :rtype: dict

.. raw:: html

    <br><hr>

MindSpore
------------------------------------------

.. py:class::
  xuance.mindspore.learners.multi_agent_rl.mappo_learner.MAPPO_Clip_Learner(config, policy, optimizer, scheduler, model_dir, gamma)

  :param config: Provides hyper parameters.
  :type config: Namespace
  :param policy: The policy that provides actions and values.
  :type policy: nn.Module
  :param optimizer: The optimizer that update the parameters of the model.
  :type optimizer: Optimizer
  :param scheduler: The tool for learning rate decay.
  :type scheduler: lr_scheduler
  :param model_dir: The directory for saving or loading the model parameters.
  :type model_dir: str
  :param gamma: The discount factor.
  :type gamma: float

.. py:function::
  xuance.mindspore.learners.multi_agent_rl.mappo_learner.MAPPO_Clip_Learner.lr_decay(i_step)

  Update the parameters of the model via backpropagation.

  :param i_step: The current training step.
  :type i_step: int
  :return: Current learning rate.
  :rtype: float

.. py:function::
  xuance.mindspore.learners.multi_agent_rl.mappo_learner.MAPPO_Clip_Learner.update(sample)

  Update the parameters of the model via backpropagation.

  :param sample: The sampled data.
  :type sample: dict
  :return: The information of the training.
  :rtype: dict

.. raw:: html

    <br><hr>

Source Code
-----------------

.. tabs::

  .. group-tab:: PyTorch

    .. code-block:: python

        """
        Multi-Agent Proximal Policy Optimization (MAPPO)
        Paper link:
        https://arxiv.org/pdf/2103.01955.pdf
        Implementation: Pytorch
        """
        from xuance.torch.learners import *
        from xuance.torch.utils.value_norm import ValueNorm
        from xuance.torch.utils.operations import update_linear_decay


        class MAPPO_Clip_Learner(LearnerMAS):
            def __init__(self,
                         config: Namespace,
                         policy: nn.Module,
                         optimizer: torch.optim.Optimizer,
                         scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
                         device: Optional[Union[int, str, torch.device]] = None,
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
                self.huber_loss = nn.HuberLoss(reduction="none", delta=self.huber_delta)
                super(MAPPO_Clip_Learner, self).__init__(config, policy, optimizer, scheduler, device, model_dir)
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
                critic_in = torch.Tensor(obs).reshape([batch_size, 1, -1]).to(self.device)
                critic_in = critic_in.expand(-1, self.n_agents, -1)
                _, value_pred = self.policy.get_values(critic_in, IDs)
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
                    critic_in_obs = obs[:, :, :-1].transpose(1, 2).reshape(batch_size, episode_length, -1)
                    critic_in_obs = critic_in_obs.unsqueeze(1).expand(-1, self.n_agents, -1, -1)
                    critic_in_state = state[:, :, :-1]
                    critic_in = torch.concat([critic_in_obs, critic_in_state], dim=-1)
                    _, value_pred = self.policy.get_values(critic_in, IDs[:, :, :-1], *rnn_hidden_critic)
                else:
                    critic_in = obs[:, :, :-1].transpose(1, 2).reshape(batch_size, episode_length, -1)
                    critic_in = critic_in.unsqueeze(1).expand(-1, self.n_agents, -1, -1)
                    _, value_pred = self.policy.get_values(critic_in, IDs[:, :, :-1], *rnn_hidden_critic)
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

        """
        Multi-Agent Proximal Policy Optimization (MAPPO)
        Paper link:
        https://arxiv.org/pdf/2103.01955.pdf
        Implementation: TensorFlow 2.X
        """
        from xuance.tensorflow.learners import *
        from xuance.tensorflow.utils.operations import update_linear_decay


        class MAPPO_Learner(LearnerMAS):
            def __init__(self,
                         config: Namespace,
                         policy: tk.Model,
                         optimizer: tk.optimizers.Optimizer,
                         device: str = "cpu:0",
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
                self.huber_loss = tk.losses.Huber(reduction="none", delta=self.huber_delta)
                super(MAPPO_Learner, self).__init__(config, policy, optimizer, device, model_dir)
                self.lr = config.learning_rate
                self.end_factor_lr_decay = config.end_factor_lr_decay

            def lr_decay(self, i_step):
                if self.use_linear_lr_decay:
                    update_linear_decay(self.optimizer, i_step, self.running_steps, self.lr, self.end_factor_lr_decay)

            def update(self, sample):
                self.iterations += 1
                with tf.device(self.device):
                    state = tf.convert_to_tensor(sample['state'])
                    obs = tf.convert_to_tensor(sample['obs'])
                    actions = tf.convert_to_tensor(sample['actions'])
                    values = tf.convert_to_tensor(sample['values'])
                    returns = tf.convert_to_tensor(sample['values'])
                    advantages = tf.convert_to_tensor(sample['advantages'])
                    log_pi_old = tf.convert_to_tensor(sample['log_pi_old'])
                    agent_mask = tf.reshape(tf.convert_to_tensor(sample['agent_mask'], tf.float32), (-1, self.n_agents, 1))
                    batch_size = obs.shape[0]
                    IDs = tf.tile(tf.expand_dims(tf.eye(self.n_agents), axis=0), multiples=(batch_size, 1, 1))

                    with tf.GradientTape() as tape:
                        # actor loss
                        inputs = {'obs': obs, 'ids': IDs}
                        _, pi_dist = self.policy(inputs)
                        log_pi = pi_dist.log_prob(actions)
                        ratio = tf.reshape(tf.math.exp(log_pi - log_pi_old), [batch_size, self.n_agents, 1])
                        advantages_mask = tf.stop_gradient(advantages * agent_mask)
                        surrogate1 = ratio * advantages_mask
                        surrogate2 = tf.clip_by_value(ratio, 1.0 - self.clip_range, 1.0 + self.clip_range) * advantages_mask
                        loss_a = -tf.reduce_mean(tf.reduce_sum(tf.minimum(surrogate1, surrogate2), axis=-1))

                        # entropy loss
                        entropy = tf.reshape(pi_dist.entropy(), agent_mask.shape) * agent_mask
                        loss_e = tf.reduce_mean(entropy)

                        # critic loss
                        critic_in = tf.reshape(obs, [batch_size, 1, -1])
                        critic_in = tf.repeat(critic_in, repeats=self.n_agents, axis=1)
                        _, value_pred = self.policy.get_values(critic_in, IDs)
                        value_pred = value_pred
                        value_target = returns
                        if self.use_value_clip:
                            value_clipped = values + tf.clip_by_value(value_pred - values, -self.value_clip_range,
                                                                      self.value_clip_range)
                            if self.use_huber_loss:
                                loss_v = self.huber_loss(value_target, value_pred)
                                loss_v_clipped = self.huber_loss(value_target, value_clipped)
                            else:
                                loss_v = (value_pred - value_target) ** 2
                                loss_v_clipped = (value_clipped - value_target) ** 2
                            loss_c = tf.maximum(loss_v, loss_v_clipped) * tf.squeeze(agent_mask, -1)
                            loss_c = tf.reduce_sum(loss_c) / tf.reduce_sum(agent_mask)
                        else:
                            if self.use_huber_loss:
                                loss_v = self.huber_loss(value_pred, value_target) * agent_mask
                            else:
                                loss_v = ((value_pred - value_target) ** 2) * agent_mask
                            loss_c = tf.reduce_sum(loss_v) / tf.reduce_sum(agent_mask)

                        loss = loss_a + self.vf_coef * loss_c - self.ent_coef * loss_e
                        gradients = tape.gradient(loss, self.policy.trainable_param())
                        self.optimizer.apply_gradients([
                            (tf.clip_by_norm(grad, self.max_grad_norm), var)
                            for (grad, var) in zip(gradients, self.policy.trainable_param())
                            if grad is not None
                        ])

                    # Logger
                    lr = self.optimizer._decayed_lr(tf.float32)

                    info = {
                        "learning_rate": lr.numpy(),
                        "actor_loss": loss_a.numpy(),
                        "critic_loss": loss_c.numpy(),
                        "entropy": loss_e.numpy(),
                        "loss": loss.numpy(),
                        "predict_value": tf.math.reduce_mean(value_pred).numpy()
                    }

                    return info

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


        class MAPPO_Learner(LearnerMAS):
            class PolicyNetWithLossCell(nn.Cell):
                def __init__(self, backbone, n_agents, vf_coef, ent_coef, clip_range, use_value_clip, value_clip_range,
                             use_huber_loss):
                    super(MAPPO_Learner.PolicyNetWithLossCell, self).__init__()
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
                    # actor loss
                    _, act_probs = self._backbone(o, ids)
                    log_pi = self._backbone.actor.log_prob(value=a, probs=act_probs)
                    ratio = self.exp(log_pi - log_pi_old).view(bs, self.n_agents, 1)
                    advantages_mask = advantages * agt_mask
                    surrogate1 = ratio * advantages_mask
                    surrogate2 = ops.clip_by_value(ratio, Tensor(1 - self.clip_range), Tensor(1 + self.clip_range)) * advantages_mask
                    loss_a = -self.miminum(surrogate1, surrogate2).sum(axis=-2, keepdims=True).mean()

                    # entropy loss
                    entropy = self._backbone.actor.entropy(probs=act_probs).reshape(agt_mask.shape) * agt_mask
                    loss_e = entropy.mean()

                    # critic loss
                    critic_in = self.broadcast_to(o.reshape([bs, 1, -1]))
                    _, value_pred = self._backbone.get_values(critic_in, ids)
                    value_pred = value_pred * agt_mask
                    value_target = returns
                    if self.use_value_clip:
                        value_clipped = values + ops.clip_by_value(value_pred - values, -self.value_clip_range, self.value_clip_range)
                        if self.use_huber_loss:
                            loss_v = self.huber_loss(value_pred, value_target)
                            loss_v_clipped = self.huber_loss(value_clipped, value_target)
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
                super(MAPPO_Learner, self).__init__(config, policy, optimizer, scheduler, model_dir)
                # define mindspore trainers
                self.loss_net = self.PolicyNetWithLossCell(policy, self.n_agents, config.vf_coef, config.ent_coef,
                                                           config.clip_range, config.use_value_clip, config.value_clip_range,
                                                           config.use_huber_loss)
                if self.args.use_grad_norm:
                    self.policy_train = TrainOneStepCellWithGradClip(self.loss_net, self.optimizer, clip_type=config.clip_type,
                                                                     clip_value=config.max_grad_norm)
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

