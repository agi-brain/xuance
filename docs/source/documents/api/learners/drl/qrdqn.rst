QRDQN_Learner
=====================================

.. raw:: html

    <br><hr>

PyTorch
------------------------------------------

.. py:class::
  xuance.torch.learners.qlearning_family.qrdqn_learner.QRDQN_Learner(policy, optimizer, scheduler, device, model_dir, gamma, sync_frequency)

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
  :param sync_frequency: The frequency to synchronize the target networks.
  :type sync_frequency: int

.. py:function::
  xuance.torch.learners.qlearning_family.qrdqn_learner.QRDQN_Learner.update(obs_batch, act_batch, rew_batch, terminal_batch)

  :param obs_batch: A batch of observations sampled from experience replay buffer.
  :type obs_batch: np.ndarray
  :param act_batch: A batch of actions sampled from experience replay buffer.
  :type act_batch: np.ndarray
  :param rew_batch: A batch of rewards sampled from experience replay buffer.
  :type rew_batch: np.ndarray
  :param terminal_batch: A batch of terminal data sampled from experience replay buffer.
  :type terminal_batch: np.ndarray
  :return: The information of the training.
  :rtype: dict

.. raw:: html

    <br><hr>

TensorFlow
------------------------------------------

.. py:class::
  xuance.tensorflow.learners.qlearning_family.qrdqn_learner.QRDQN_Learner(policy, optimizer, device, model_dir, gamma, sync_frequency)

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
  :param sync_frequency: The frequency to synchronize the target networks.
  :type sync_frequency: int

.. py:function::
  xuance.tensorflow.learners.qlearning_family.qrdqn_learner.QRDQN_Learner.update(obs_batch, act_batch, rew_batch, next_batch, terminal_batch)

  :param obs_batch: A batch of observations sampled from experience replay buffer.
  :type obs_batch: np.ndarray
  :param act_batch: A batch of actions sampled from experience replay buffer.
  :type act_batch: np.ndarray
  :param rew_batch: A batch of rewards sampled from experience replay buffer.
  :type rew_batch: np.ndarray
  :param next_batch: A batch of next observations sampled from experience replay buffer.
  :type next_batch: np.ndarray
  :param terminal_batch: A batch of terminal data sampled from experience replay buffer.
  :type terminal_batch: np.ndarray
  :return: The information of the training.
  :rtype: dict

.. raw:: html

    <br><hr>

MindSpore
------------------------------------------

.. py:class::
  xuance.mindspore.learners.qlearning_family.qrdqn_learner.QRDQN_Learner(policy, optimizer, scheduler, model_dir, gamma, sync_frequency)

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
  :param sync_frequency: The frequency to synchronize the target networks.
  :type sync_frequency: int

.. py:function::
  xuance.mindspore.learners.qlearning_family.qrdqn_learner.QRDQN_Learner.update(obs_batch, act_batch, rew_batch, next_batch, terminal_batch)

  :param obs_batch: A batch of observations sampled from experience replay buffer.
  :type obs_batch: np.ndarray
  :param act_batch: A batch of actions sampled from experience replay buffer.
  :type act_batch: np.ndarray
  :param rew_batch: A batch of rewards sampled from experience replay buffer.
  :type rew_batch: np.ndarray
  :param next_batch: A batch of next observations sampled from experience replay buffer.
  :type next_batch: np.ndarray
  :param terminal_batch: A batch of terminal data sampled from experience replay buffer.
  :type terminal_batch: np.ndarray
  :return: The information of the training.
  :rtype: dict

.. raw:: html

    <br><hr>

Source Code
-----------------

.. tabs::

  .. group-tab:: PyTorch

    .. code-block:: python

       from xuance.torch.learners import *


        class DRQN_Learner(Learner):
            def __init__(self,
                         policy: nn.Module,
                         optimizer: torch.optim.Optimizer,
                         scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
                         device: Optional[Union[int, str, torch.device]] = None,
                         model_dir: str = "./",
                         gamma: float = 0.99,
                         sync_frequency: int = 100):
                self.gamma = gamma
                self.sync_frequency = sync_frequency
                super(DRQN_Learner, self).__init__(policy, optimizer, scheduler, device, model_dir)

            def update(self, obs_batch, act_batch, rew_batch, terminal_batch):
                self.iterations += 1
                act_batch = torch.as_tensor(act_batch, device=self.device)
                rew_batch = torch.as_tensor(rew_batch, device=self.device)
                ter_batch = torch.as_tensor(terminal_batch, device=self.device, dtype=torch.float)
                batch_size = obs_batch.shape[0]

                rnn_hidden = self.policy.init_hidden(batch_size)
                _, _, evalQ, _ = self.policy(obs_batch[:, 0:-1], *rnn_hidden)
                target_rnn_hidden = self.policy.init_hidden(batch_size)
                _, targetA, targetQ, _ = self.policy.target(obs_batch[:, 1:], *target_rnn_hidden)
                # targetQ = targetQ.max(dim=-1).values

                targetA = F.one_hot(targetA, targetQ.shape[-1])
                targetQ = (targetQ * targetA).sum(dim=-1)

                targetQ = rew_batch + self.gamma * (1 - ter_batch) * targetQ
                predictQ = (evalQ * F.one_hot(act_batch.long(), evalQ.shape[-1])).sum(dim=-1)

                loss = F.mse_loss(predictQ, targetQ)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                if self.scheduler is not None:
                    self.scheduler.step()

                # hard update for target network
                if self.iterations % self.sync_frequency == 0:
                    self.policy.copy_target()
                lr = self.optimizer.state_dict()['param_groups'][0]['lr']

                info = {
                    "Qloss": loss.item(),
                    "learning_rate": lr,
                    "predictQ": predictQ.mean().item()
                }

                return info



  .. group-tab:: TensorFlow

    .. code-block:: python

        from xuance.tensorflow.learners import *


        class QRDQN_Learner(Learner):
            def __init__(self,
                         policy: tk.Model,
                         optimizer: tk.optimizers.Optimizer,
                         device: str = "cpu:0",
                         model_dir: str = "./",
                         gamma: float = 0.99,
                         sync_frequency: int = 100):
                self.gamma = gamma
                self.sync_frequency = sync_frequency
                super(QRDQN_Learner, self).__init__(policy, optimizer, device, model_dir)

            def update(self, obs_batch, act_batch, rew_batch, next_batch, terminal_batch):
                self.iterations += 1
                with tf.device(self.device):
                    act_batch = tf.convert_to_tensor(act_batch, dtype=tf.int64)
                    rew_batch = tf.convert_to_tensor(rew_batch)
                    ter_batch = tf.convert_to_tensor(terminal_batch)

                    with tf.GradientTape() as tape:
                        _, _, evalZ = self.policy(obs_batch)
                        _, targetA, targetZ = self.policy.target(next_batch)
                        current_quantile = tf.math.reduce_sum(evalZ * tf.expand_dims(tf.one_hot(act_batch, evalZ.shape[1]), axis=-1), axis=1)
                        target_quantile = tf.math.reduce_sum(targetZ * tf.expand_dims(tf.one_hot(targetA, evalZ.shape[1]), axis=-1), axis=1)
                        target_quantile = tf.expand_dims(rew_batch, 1) + self.gamma * target_quantile * (1 - tf.expand_dims(ter_batch, 1))
                        target_quantile = tf.stop_gradient(target_quantile)
                        loss = tk.losses.mean_squared_error(tf.reshape(target_quantile, [-1, ]), tf.reshape(current_quantile, [-1, ]))
                        gradients = tape.gradient(loss, self.policy.trainable_variables)
                        self.optimizer.apply_gradients([
                            (grad, var)
                            for (grad, var) in zip(gradients, self.policy.trainable_variables)
                            if grad is not None
                        ])

                    # hard update for target network
                    if self.iterations % self.sync_frequency == 0:
                        self.policy.copy_target()

                    lr = self.optimizer._decayed_lr(tf.float32)

                    info = {
                        "Qloss": loss.numpy(),
                        "predictQ": tf.math.reduce_mean(current_quantile).numpy(),
                        "lr": lr.numpy()
                    }

                    return info


  .. group-tab:: MindSpore

    .. code-block:: python

        from xuance.mindspore.learners import *
        from mindspore.ops import OneHot,ExpandDims,ReduceSum


        class QRDQN_Learner(Learner):
            class PolicyNetWithLossCell(nn.Cell):
                def __init__(self, backbone, loss_fn):
                    super(QRDQN_Learner.PolicyNetWithLossCell, self).__init__(auto_prefix=False)
                    self._backbone = backbone
                    self._loss_fn = loss_fn
                    self._onehot = OneHot()
                    self.on_value = Tensor(1.0, ms.float32)
                    self.off_value = Tensor(0.0, ms.float32)
                    self._unsqueeze = ExpandDims()
                    self._sum = ReduceSum()

                def construct(self, x, a, target_quantile):
                    _,_,evalZ = self._backbone(x)
                    current_quantile = self._sum(evalZ * self._unsqueeze(self._onehot(a, evalZ.shape[1], self.on_value, self.off_value), -1), 1)
                    loss = self._loss_fn(target_quantile, current_quantile)
                    return loss

            def __init__(self,
                         policy: nn.Cell,
                         optimizer: nn.Optimizer,
                         scheduler: Optional[nn.exponential_decay_lr] = None,
                         model_dir: str = "./",
                         gamma: float = 0.99,
                         sync_frequency: int = 100):
                self.gamma = gamma
                self.sync_frequency = sync_frequency
                super(QRDQN_Learner, self).__init__(policy, optimizer, scheduler, model_dir)
                # define loss function
                loss_fn = nn.MSELoss()
                # connect the feed forward network with loss function.
                self.loss_net = self.PolicyNetWithLossCell(policy, loss_fn)
                # define the training network
                self.policy_train = nn.TrainOneStepCell(self.loss_net, optimizer)
                # set the training network as train mode.
                self.policy_train.set_train()

                self._onehot = OneHot()
                self.on_value = Tensor(1.0, ms.float32)
                self.off_value = Tensor(0.0, ms.float32)
                self._unsqueeze = ExpandDims()
                self._sum = ReduceSum()

            def update(self, obs_batch, act_batch, rew_batch, next_batch, terminal_batch):
                self.iterations += 1
                obs_batch = Tensor(obs_batch)
                act_batch = Tensor(act_batch, ms.int32)
                rew_batch = Tensor(rew_batch)
                next_batch = Tensor(next_batch)
                ter_batch = Tensor(terminal_batch)

                _, targetA, targetZ = self.policy(next_batch)
                target_quantile = self._sum(targetZ * self._unsqueeze(self._onehot(targetA, targetZ.shape[1], self.on_value, self.off_value), -1), 1)
                target_quantile = self._unsqueeze(rew_batch, 1) + self.gamma * target_quantile * (1-self._unsqueeze(ter_batch, 1))

                loss = self.policy_train(obs_batch, act_batch, target_quantile)

                # hard update for target network
                if self.iterations % self.sync_frequency == 0:
                    self.policy.copy_target()

                lr = self.scheduler(self.iterations).asnumpy()

                info = {
                    "Qloss": loss.asnumpy(),
                    "learning_rate": lr
                }

                return info

