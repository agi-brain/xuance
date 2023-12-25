VDAC_Learner
======================

An implementation of the Value-Decomposition Actor-Critic (VDAC) algorithm.

.. raw:: html

    <br><hr>

PyTorch
------------------------------------------

.. py:class::
  xuance.torch.learners.multi_agent_rl.vdac_learner.VDAC_Learner(config, policy, optimizer, scheduler, device, model_dir, gamma)

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
  xuance.torch.learners.multi_agent_rl.vdac_learner.VDAC_Learner.lr_decay(i_step)

  Update the learning rate of the VDAC learner based on the current training step.

  :param i_step: The current step.
  :type i_step: int
  :return: The information of the training.
  :rtype: dict

.. py:function::
  xuance.torch.learners.multi_agent_rl.vdac_learner.VDAC_Learner.update(sample)

  Update the VDAC learner based on the sampled experience from the experience replay buffer.

  :param sample: A dictionary containing necessary experience data that is sampled from experience replay buffer.
  :type sample: dict
  :return: The information of the training.
  :rtype: dict

.. py:function::
  xuance.torch.learners.multi_agent_rl.vdac_learner.VDAC_Learner.update_recurrent(sample)

  Update the VDAC learner for recurrent architectures based on the sampled experience from the experience replay buffer.

  :param sample: A dictionary containing necessary experience data that is sampled from experience replay buffer.
  :type sample: dict
  :return: The information of the training.
  :rtype: dict

.. raw:: html

    <br><hr>

TensorFlow
------------------------------------------

.. py:class::
  xuance.tensorflow.learners.multi_agent_rl.vdac_learner.VDAC_Learner(config, policy, optimizer, device, model_dir, gamma)

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
  xuance.tensorflow.learners.multi_agent_rl.vdac_learner.VDAC_Learner.lr_decay(i_step)

  Update the learning rate of the VDAC learner based on the current training step.

  :param i_step: The current step.
  :type i_step: int
  :return: The information of the training.
  :rtype: dict

.. py:function::
  xuance.tensorflow.learners.multi_agent_rl.vdac_learner.VDAC_Learner.update(sample)

  Update the VDAC learner based on the sampled experience from the experience replay buffer.

  :param sample: A dictionary containing necessary experience data that is sampled from experience replay buffer.
  :type sample: dict
  :return: The information of the training.
  :rtype: dict

.. raw:: html

    <br><hr>

MindSpore
------------------------------------------

.. py:class::
  xuance.mindspore.learners.multi_agent_rl.vdac_learner.VDAC_Learner(config, policy, optimizer, scheduler, model_dir, gamma)

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
  xuance.mindspore.learners.multi_agent_rl.vdac_learner.VDAC_Learner.lr_decay(i_step)

  Update the learning rate of the VDAC learner based on the current training step.

  :param i_step: The current step.
  :type i_step: int
  :return: The information of the training.
  :rtype: dict

.. py:function::
  xuance.mindspore.learners.multi_agent_rl.vdac_learner.VDAC_Learner.update(sample)

  Update the VDAC learner for recurrent architectures based on the sampled experience from the experience replay buffer.

  :param sample: A dictionary containing necessary experience data that is sampled from experience replay buffer.
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
        Value-Dcomposition Actor-Critic (VDAC)
        Paper link:
        https://ojs.aaai.org/index.php/AAAI/article/view/17353
        Implementation: Pytorch
        """
        from xuance.torch.learners import *
        from xuance.torch.utils.value_norm import ValueNorm
        from xuance.torch.utils.operations import update_linear_decay


        class VDAC_Learner(LearnerMAS):
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
                self.use_value_norm = config.use_value_norm
                self.vf_coef, self.ent_coef = config.vf_coef, config.ent_coef
                super(VDAC_Learner, self).__init__(config, policy, optimizer, scheduler, device, model_dir)
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
                value_pred = self.policy.value_tot(value_pred, global_state=state)
                value_target = returns.mean(1)
                values = values.mean(1)
                if self.use_value_clip:
                    value_clipped = values + (value_pred - values).clamp(-self.value_clip_range, self.value_clip_range)
                    if self.use_huber_loss:
                        loss_v = self.huber_loss(value_pred, value_target)
                        loss_v_clipped = self.huber_loss(value_clipped, value_target)
                    else:
                        loss_v = (value_pred - value_target) ** 2
                        loss_v_clipped = (value_clipped - value_target) ** 2
                    loss_c = torch.max(loss_v, loss_v_clipped)
                    loss_c = loss_c.sum()
                else:
                    if self.use_huber_loss:
                        loss_v = self.huber_loss(value_pred, value_target)
                    else:
                        loss_v = (value_pred - value_target) ** 2
                    loss_c = loss_v.sum()

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
                obs = torch.Tensor(sample['obs']).to(self.device)
                actions = torch.Tensor(sample['actions']).to(self.device)
                returns = torch.Tensor(sample['returns']).to(self.device)
                avail_actions = torch.Tensor(sample['avail_actions']).float().to(self.device)
                filled = torch.Tensor(sample['filled']).float().to(self.device)
                batch_size = obs.shape[0]
                episode_length = actions.shape[2]
                IDs = torch.eye(self.n_agents).unsqueeze(1).unsqueeze(0).expand(batch_size, -1, episode_length + 1, -1).to(
                    self.device)

                filled_n = filled.unsqueeze(1).expand(batch_size, self.n_agents, episode_length, 1)

                # actor loss
                rnn_hidden = self.policy.representation.init_hidden(batch_size * self.n_agents)
                _, pi_dist, value_pred = self.policy(obs[:, :, :-1].reshape(-1, episode_length, self.dim_obs),
                                                     IDs[:, :, :-1],
                                                     *rnn_hidden,
                                                     avail_actions=avail_actions[:, :, :-1],
                                                     state=state[:, :-1])
                log_pi = pi_dist.log_prob(actions).unsqueeze(-1)
                entropy = pi_dist.entropy().unsqueeze(-1)

                targets = returns
                advantages = targets - value_pred
                td_error = value_pred - targets.detach()

                pg_loss = -((advantages.detach() * log_pi) * filled_n).sum() / filled_n.sum()
                vf_loss = ((td_error ** 2) * filled_n).sum() / filled_n.sum()
                entropy_loss = (entropy * filled_n).sum() / filled_n.sum()
                loss = pg_loss + self.vf_coef * vf_loss - self.ent_coef * entropy_loss

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
                    "pg_loss": pg_loss.item(),
                    "vf_loss": vf_loss.item(),
                    "entropy_loss": entropy_loss.item(),
                    "loss": loss.item(),
                    "predict_value": value_pred.mean().item()
                })

                return info

  .. group-tab:: TensorFlow

    .. code-block:: python

        """
        Value Decomposition Actor-Critic (VDAC)
        Paper link: https://ojs.aaai.org/index.php/AAAI/article/view/17353
        Implementation: Pytorch
        """
        from xuance.tensorflow.learners import *
        from xuance.torch.utils.operations import update_linear_decay


        class VDAC_Learner(LearnerMAS):
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
                self.use_value_norm = config.use_value_norm
                self.vf_coef, self.ent_coef = config.vf_coef, config.ent_coef
                super(VDAC_Learner, self).__init__(config, policy, optimizer, device, model_dir)
                self.lr = config.learning_rate
                self.end_factor_lr_decay = config.end_factor_lr_decay

            def lr_decay(self, i_step):
                if self.use_linear_lr_decay:
                    update_linear_decay(self.optimizer, i_step, self.running_steps, self.lr, self.end_factor_lr_decay)

            def update(self, sample):
                info = {}
                self.iterations += 1
                with tf.device(self.device):
                    state = tf.convert_to_tensor(sample['state'])
                    obs = tf.convert_to_tensor(sample['obs'])
                    actions = tf.convert_to_tensor(sample['actions'])
                    returns = tf.reduce_mean(tf.convert_to_tensor(sample['values']), axis=1)
                    agent_mask = tf.reshape(tf.convert_to_tensor(sample['agent_mask'], tf.float32), (-1, self.n_agents, 1))
                    batch_size = obs.shape[0]
                    IDs = tf.tile(tf.expand_dims(tf.eye(self.n_agents), axis=0), multiples=(batch_size, 1, 1))

                    with tf.GradientTape() as tape:
                        inputs = {'obs': obs, 'ids': IDs}
                        _, pi_dist, value_pred = self.policy(inputs, state=state)
                        log_pi = tf.expand_dims(pi_dist.log_prob(actions), -1)
                        entropy = tf.reshape(pi_dist.entropy(), agent_mask.shape) * agent_mask

                        targets = returns
                        advantages = tf.expand_dims(tf.stop_gradient(targets - value_pred), -1)
                        td_error = tf.expand_dims(value_pred - tf.stop_gradient(targets), -1)

                        pg_loss = -tf.reduce_sum((advantages * log_pi) * agent_mask) / tf.reduce_sum(agent_mask)
                        vf_loss = tf.reduce_sum((td_error ** 2) * agent_mask) / tf.reduce_sum(agent_mask)
                        entropy_loss = tf.reduce_sum(entropy * agent_mask) / tf.reduce_sum(agent_mask)
                        loss = pg_loss + self.vf_coef * vf_loss - self.ent_coef * entropy_loss

                        gradients = tape.gradient(loss, self.policy.trainable_param())
                        self.optimizer.apply_gradients([
                            (tf.clip_by_norm(grad, self.max_grad_norm), var)
                            for (grad, var) in zip(gradients, self.policy.trainable_param())
                            if grad is not None
                        ])

                    # Logger
                    lr = self.optimizer._decayed_lr(tf.float32)

                    info.update({
                        "learning_rate": lr.numpy(),
                        "pg_loss": pg_loss.numpy(),
                        "vf_loss": vf_loss.numpy(),
                        "entropy_loss": entropy_loss.numpy(),
                        "loss": loss.numpy(),
                        "predict_value": tf.reduce_mean(value_pred).numpy()
                    })

                    return info


  .. group-tab:: MindSpore

    .. code-block:: python

        """
        Value Decomposition Actor-Critic (VDAC)
        Paper link: https://ojs.aaai.org/index.php/AAAI/article/view/17353
        Implementation: MindSpore
        """
        from xuance.mindspore.learners import *
        from xuance.torch.utils.operations import update_linear_decay


        class VDAC_Learner(LearnerMAS):
            class PolicyNetWithLossCell(nn.Cell):
                def __init__(self, backbone, vf_coef, ent_coef):
                    super(VDAC_Learner.PolicyNetWithLossCell, self).__init__()
                    self._backbone = backbone
                    self._vf_coef = vf_coef
                    self._ent_coef = ent_coef
                    self.loss_c = nn.MSELoss()

                def construct(self, o, s, a, adv, ret, ids, agt_mask):
                    _, act_probs, v_pred = self._backbone(o, ids)
                    v_pred_tot = self._backbone.value_tot(v_pred * agt_mask, s)
                    log_prob = self._backbone.actor.log_prob(value=a, probs=act_probs).reshape(adv.shape)
                    entropy = self._backbone.actor.entropy(probs=act_probs).reshape(agt_mask.shape) * agt_mask

                    loss_a = -(adv * log_prob * agt_mask).mean()
                    loss_c = self.loss_c(logits=v_pred_tot, labels=ret)
                    loss_e = entropy.mean()

                    loss = loss_a + self._vf_coef * loss_c - self._ent_coef * loss_e
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
                self.use_value_norm = config.use_value_norm
                self.vf_coef, self.ent_coef = config.vf_coef, config.ent_coef
                self.mse_loss = nn.MSELoss()
                super(VDAC_Learner, self).__init__(config, policy, optimizer, scheduler, model_dir)
                self.loss_net = self.PolicyNetWithLossCell(policy, config.vf_coef, config.ent_coef)
                self.policy_train = TrainOneStepCellWithGradClip(self.loss_net, optimizer,
                                                                 clip_type=config.clip_type, clip_value=config.max_grad_norm)
                self.policy_train.set_train()
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
                returns = Tensor(sample['values']).mean(axis=1)
                advantages = Tensor(sample['advantages'])
                agent_mask = Tensor(sample['agent_mask']).view(-1, self.n_agents, 1)
                batch_size = obs.shape[0]
                IDs = ops.broadcast_to(self.expand_dims(self.eye(self.n_agents, self.n_agents, ms.float32), 0),
                                       (batch_size, -1, -1))

                loss = self.policy_train(obs, state, actions, advantages, returns, IDs, agent_mask)

                # Logger
                lr = self.scheduler(self.iterations).asnumpy()

                info = {
                    "learning_rate": lr,
                    "loss": loss.asnumpy()
                }

                return info

