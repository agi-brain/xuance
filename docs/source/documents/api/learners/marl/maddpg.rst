MADDPG_Learner
=====================================

xxxxxx.

.. raw:: html

    <br><hr>

**PyTorch:**

.. py:class::
  xuance.torch.learners.multi_agent_rl.maddpg_learner.MADDPG_Learner(config, policy, optimizer, scheduler, device, model_dir, gamma, sync_frequency)

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
  :param sync_frequency: xxxxxx.
  :type sync_frequency: xxxxxx

.. py:function::
  xuance.torch.learners.multi_agent_rl.isac_learner.ISAC_Learner.update(sample)

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
  xuance.mindspore.learners.multi_agent_rl.maddpg_learner.MADDPG_Learner(config, policy, optimizer, scheduler, model_dir, gamma, sync_frequency)

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
  :param sync_frequency: xxxxxx.
  :type sync_frequency: xxxxxx

.. py:function::
  xuance.mindspore.learners.multi_agent_rl.isac_learner.ISAC_Learner.update(sample)

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
        Multi-Agent Deep Deterministic Policy Gradient
        Paper link:
        https://proceedings.neurips.cc/paper/2017/file/68a9750337a418a86fe06c1991a1d64c-Paper.pdf
        Implementation: Pytorch
        Trick: Parameter sharing for all agents, with agents' one-hot IDs as actor-critic's inputs.
        """
        from xuance.torch.learners import *


        class MADDPG_Learner(LearnerMAS):
            def __init__(self,
                         config: Namespace,
                         policy: nn.Module,
                         optimizer: Sequence[torch.optim.Optimizer],
                         scheduler: Sequence[torch.optim.lr_scheduler._LRScheduler] = None,
                         device: Optional[Union[int, str, torch.device]] = None,
                         model_dir: str = "./",
                         gamma: float = 0.99,
                         sync_frequency: int = 100
                         ):
                self.gamma = gamma
                self.tau = config.tau
                self.sync_frequency = sync_frequency
                self.mse_loss = nn.MSELoss()
                super(MADDPG_Learner, self).__init__(config, policy, optimizer, scheduler, device, model_dir)
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
                rewards = torch.Tensor(sample['rewards']).to(self.device)
                terminals = torch.Tensor(sample['terminals']).float().reshape(-1, self.n_agents, 1).to(self.device)
                agent_mask = torch.Tensor(sample['agent_mask']).float().reshape(-1, self.n_agents, 1).to(self.device)
                IDs = torch.eye(self.n_agents).unsqueeze(0).expand(self.args.batch_size, -1, -1).to(self.device)

                # train actor
                _, actions_eval = self.policy(obs, IDs)
                loss_a = -(self.policy.critic(obs, actions_eval, IDs) * agent_mask).sum() / agent_mask.sum()
                self.optimizer['actor'].zero_grad()
                loss_a.backward()
                if self.args.use_grad_clip:
                    torch.nn.utils.clip_grad_norm_(self.policy.parameters_actor, self.args.grad_clip_norm)
                self.optimizer['actor'].step()
                if self.scheduler['actor'] is not None:
                    self.scheduler['actor'].step()

                # train critic
                actions_next = self.policy.target_actor(obs_next, IDs)
                q_eval = self.policy.critic(obs, actions, IDs)
                q_next = self.policy.target_critic(obs_next, actions_next, IDs)
                q_target = rewards + (1 - terminals) * self.args.gamma * q_next
                td_error = (q_eval - q_target.detach()) * agent_mask
                loss_c = (td_error ** 2).sum() / agent_mask.sum()
                self.optimizer['critic'].zero_grad()
                loss_c.backward()
                if self.args.use_grad_clip:
                    torch.nn.utils.clip_grad_norm_(self.policy.parameters_critic, self.args.grad_clip_norm)
                self.optimizer['critic'].step()
                if self.scheduler['critic'] is not None:
                    self.scheduler['critic'].step()

                self.policy.soft_update(self.tau)

                lr_a = self.optimizer['actor'].state_dict()['param_groups'][0]['lr']
                lr_c = self.optimizer['critic'].state_dict()['param_groups'][0]['lr']

                info = {
                    "learning_rate_actor": lr_a,
                    "learning_rate_critic": lr_c,
                    "loss_actor": loss_a.item(),
                    "loss_critic": loss_c.item(),
                    "predictQ": q_eval.mean().item()
                }

                return info











  .. group-tab:: TensorFlow

    .. code-block:: python


  .. group-tab:: MindSpore

    .. code-block:: python

        """
        Multi-Agent Deep Deterministic Policy Gradient
        Paper link:
        https://proceedings.neurips.cc/paper/2017/file/68a9750337a418a86fe06c1991a1d64c-Paper.pdf
        Implementation: MindSpore
        Trick: Parameter sharing for all agents, with agents' one-hot IDs as actor-critic's inputs.
        """
        from xuance.mindspore.learners import *


        class MADDPG_Learner(LearnerMAS):
            class ActorNetWithLossCell(nn.Cell):
                def __init__(self, backbone, n_agents):
                    super(MADDPG_Learner.ActorNetWithLossCell, self).__init__()
                    self._backbone = backbone
                    self._mean = ms.ops.ReduceMean(keep_dims=True)
                    self.n_agents = n_agents

                def construct(self, bs, o, ids, agt_mask):
                    _, actions_eval = self._backbone(o, ids)
                    loss_a = -(self._backbone.critic(o, actions_eval, ids) * agt_mask).sum() / agt_mask.sum()
                    return loss_a

            class CriticNetWithLossCell(nn.Cell):
                def __init__(self, backbone):
                    super(MADDPG_Learner.CriticNetWithLossCell, self).__init__()
                    self._backbone = backbone
                    self._loss = nn.MSELoss()

                def construct(self, o, a_n, ids, agt_mask, tar_q):
                    q_eval = self._backbone.critic(o, a_n, ids)
                    td_error = (q_eval - tar_q) * agt_mask
                    loss_c = (td_error ** 2).sum() / agt_mask.sum()
                    return loss_c

            def __init__(self,
                         config: Namespace,
                         policy: nn.Cell,
                         optimizer: Sequence[nn.Optimizer],
                         scheduler: Sequence[nn.exponential_decay_lr] = None,
                         model_dir: str = "./",
                         gamma: float = 0.99,
                         sync_frequency: int = 100
                         ):
                self.gamma = gamma
                self.tau = config.tau
                self.sync_frequency = sync_frequency
                self.mse_loss = nn.MSELoss()
                super(MADDPG_Learner, self).__init__(config, policy, optimizer, scheduler, model_dir)
                self.optimizer = {
                    'actor': optimizer[0],
                    'critic': optimizer[1]
                }
                self.scheduler = {
                    'actor': scheduler[0],
                    'critic': scheduler[1]
                }
                # define mindspore trainers
                self.actor_loss_net = self.ActorNetWithLossCell(policy, self.n_agents)
                self.actor_train = nn.TrainOneStepCell(self.actor_loss_net, self.optimizer['actor'])
                self.actor_train.set_train()
                self.critic_loss_net = self.CriticNetWithLossCell(policy)
                self.critic_train = nn.TrainOneStepCell(self.critic_loss_net, self.optimizer['critic'])
                self.critic_train.set_train()

            def update(self, sample):
                self.iterations += 1
                obs = Tensor(sample['obs'])
                actions = Tensor(sample['actions'])
                obs_next = Tensor(sample['obs_next'])
                rewards = Tensor(sample['rewards'])
                terminals = Tensor(sample['terminals']).view(-1, self.n_agents, 1)
                agent_mask = Tensor(sample['agent_mask']).view(-1, self.n_agents, 1)
                batch_size = obs.shape[0]
                IDs = ops.broadcast_to(self.expand_dims(self.eye(self.n_agents, self.n_agents, ms.float32), 0),
                                       (batch_size, -1, -1))
                # calculate the loss and train
                actions_next = self.policy.target_actor(obs_next, IDs)
                q_next = self.policy.target_critic(obs_next, actions_next, IDs)
                q_target = rewards + (1 - terminals) * self.args.gamma * q_next

                # calculate the loss and train
                loss_a = self.actor_train(batch_size, obs, IDs, agent_mask)
                loss_c = self.critic_train(obs, actions, IDs, agent_mask, q_target)
                self.policy.soft_update(self.tau)

                lr_a = self.scheduler['actor'](self.iterations).asnumpy()
                lr_c = self.scheduler['critic'](self.iterations).asnumpy()

                info = {
                    "learning_rate_actor": lr_a,
                    "learning_rate_critic": lr_c,
                    "loss_actor": loss_a.asnumpy(),
                    "loss_critic": loss_c.asnumpy()
                }

                return info

    