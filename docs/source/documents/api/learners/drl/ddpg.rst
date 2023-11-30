DDPG_Learner
=====================================
A2C_Learner
=====================================

.. raw:: html

    <br><hr>

**PyTorch:**

.. py:class::
  xuance.torch.learners.policy_gradient.a2c_learner.A2C_Learner(policy, optimizer, scheduler, device, model_dir, vf_coef, ent_coef, clip_grad)

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
  :param vf_coef: xxxxxx.
  :type vf_coef: xxxxxx
  :param ent_coef: xxxxxx.
  :type ent_coef: xxxxxx
  :param clip_grad: xxxxxx.
  :type clip_grad: xxxxxx

.. py:function::
  xuance.torch.learners.policy_gradient.a2c_learner.A2C_Learner.update(obs_batch, act_batch, ret_batch, adv_batch)

  :param obs_batch: xxxxxx.
  :type obs_batch: xxxxxx
  :param act_batch: xxxxxx.
  :type act_batch: xxxxxx
  :param ret_batch: xxxxxx.
  :type ret_batch: xxxxxx
  :param adv_batch: xxxxxx.
  :type adv_batch: xxxxxx
  :return: xxxxxx.
  :rtype: xxxxxx


.. py:class::
  xuance.torch.policies.deterministic_marl.BasicQnetwork(action_space, n_agents, representation, hidden_size, normalize, initialize, activation, device)

  :param action_space: xxxxxx.
  :type action_space: xxxxxx
  :param n_agents: xxxxxx.
  :type n_agents: xxxxxx
  :param representation: xxxxxx.
  :type representation: xxxxxx
  :param hidden_sizes: xxxxxx.
  :type hidden_sizes: xxxxxx
  :param normalize: xxxxxx.
  :type normalize: xxxxxx
  :param initialize: xxxxxx.
  :type initialize: xxxxxx
  :param activation: xxxxxx.
  :type activation: xxxxxx
  :param device: xxxxxx.
  :type device: xxxxxx

.. py:function::
  xuance.torch.policies.deterministic_marl.BasicQnetwork.forward(observation, agent_ids)

  :param observation: xxxxxx.
  :type observation: xxxxxx
  :param agent_ids: xxxxxx.
  :type agent_ids: xxxxxx
  :return: xxxxxx.
  :rtype: xxxxxx

.. py:function::
  xuance.torch.policies.deterministic_marl.BasicQnetwork.target_Q(observation, agent_ids, *rnn_hidden)

  :param observation: xxxxxx.
  :type observation: xxxxxx
  :param agent_ids: xxxxxx.
  :type agent_ids: xxxxxx
  :param *rnn_hidden: xxxxxx.
  :type *rnn_hidden: xxxxxx
  :return: xxxxxx.
  :rtype: xxxxxx

.. py:function::
  xuance.torch.policies.deterministic_marl.BasicQnetwork.copy_target()

  :return: None.
  :rtype: xxxxxx

.. py:class::
  xuance.torch.policies.deterministic_marl.MFQnetwork(action_space, n_agents, representation, hidden_sizes, normalize, initialize, activation, device)

  :param action_space: xxxxxx.
  :type action_space: xxxxxx
  :param n_agents: xxxxxx.
  :type n_agents: xxxxxx
  :param representation: xxxxxx.
  :type representation: xxxxxx
  :param hidden_sizes: xxxxxx.
  :type hidden_sizes: xxxxxx
  :param normalize: xxxxxx.
  :type normalize: xxxxxx
  :param initialize: xxxxxx.
  :type initialize: xxxxxx
  :param activation: xxxxxx.
  :type activation: xxxxxx
  :param device: xxxxxx.
  :type device: xxxxxx

.. py:function::
  xuance.torch.policies.deterministic_marl.MFQnetwork.forward(observation, actions_mean, agent_ids)

  :param observation: xxxxxx.
  :type observation: xxxxxx
  :param actions_mean: xxxxxx.
  :type actions_mean: xxxxxx
  :param agent_ids: xxxxxx.
  :type agent_ids: xxxxxx
  :return: xxxxxx.
  :rtype: xxxxxx

.. py:function::
  xuance.torch.policies.deterministic_marl.MFQnetwork.sample_actions(logits)

  :param logits: xxxxxx.
  :type logits: xxxxxx
  :return: xxxxxx.
  :rtype: xxxxxx

.. py:function::
  xuance.torch.policies.deterministic_marl.MFQnetwork.target_Q(observation, actions_mean, agent_ids)

  :param observation: xxxxxx.
  :type observation: xxxxxx
  :param actions_mean: xxxxxx.
  :type actions_mean: xxxxxx
  :param agent_ids: xxxxxx.
  :type agent_ids: xxxxxx
  :return: xxxxxx.
  :rtype: xxxxxx

.. py:function::
  xuance.torch.policies.deterministic_marl.MFQnetwork.copy_target()

  :return: None.
  :rtype: xxxxxx

.. py:class::
  xuance.torch.policies.deterministic_marl.MixingQnetwork(action_space, n_agents, representation, mixer, hidden_size, normalize, initialize, activation, device)

  :param action_space: xxxxxx.
  :type action_space: xxxxxx
  :param n_agents: xxxxxx.
  :type n_agents: xxxxxx
  :param representation: xxxxxx.
  :type representation: xxxxxx
  :param mixer: xxxxxx.
  :type mixer: xxxxxx
  :param hidden_size: xxxxxx.
  :type hidden_size: xxxxxx
  :param normalize: xxxxxx.
  :type normalize: xxxxxx
  :param initialize: xxxxxx.
  :type initialize: xxxxxx
  :param activation: xxxxxx.
  :type activation: xxxxxx
  :param device: xxxxxx.
  :type device: xxxxxx

.. py:function::
  xuance.torch.policies.deterministic_marl.MixingQnetwork.forward(observation, agent_ids, *rnn_hidden, avail_actions)

  :param observation: xxxxxx.
  :type observation: xxxxxx
  :param agent_ids: xxxxxx.
  :type agent_ids: xxxxxx
  :param *rnn_hidden: xxxxxx.
  :type *rnn_hidden: xxxxxx
  :param avail_actions: xxxxxx.
  :type avail_actions: xxxxxx
  :return: xxxxxx.
  :rtype: xxxxxx

.. py:function::
  xuance.torch.policies.deterministic_marl.MixingQnetwork.target_Q(observation, agent_ids, *rnn_hidden)

  :param observation: xxxxxx.
  :type observation: xxxxxx
  :param agent_ids: xxxxxx.
  :type agent_ids: xxxxxx
  :param *rnn_hidden: xxxxxx.
  :type *rnn_hidden: xxxxxx
  :return: xxxxxx.
  :rtype: xxxxxx

.. py:function::
  xuance.torch.policies.deterministic_marl.MixingQnetwork.Q_tot(q, states)

  :param q: xxxxxx.
  :type q: xxxxxx
  :param states: xxxxxx.
  :type gstates: xxxxxx
  :return: xxxxxx.
  :rtype: xxxxxx

.. py:function::
  xuance.torch.policies.deterministic_marl.MixingQnetwork.target_Q_tot(q, states)

  :param q: xxxxxx.
  :type q: xxxxxx
  :param states: xxxxxx.
  :type gstates: xxxxxx
  :return: xxxxxx.
  :rtype: xxxxxx

.. py:function::
  xuance.torch.policies.deterministic_marl.MixingQnetwork.copy_target()

  :return: None.
  :rtype: xxxxxx

.. py:class::
  xuance.torch.policies.deterministic_marl.Weighted_MixingQnetwork(action_space, n_agents, representation, mixer, ff_mixer, hidden_size, normalize, initialize, activation, device)

  :param action_space: xxxxxx.
  :type action_space: xxxxxx
  :param n_agents: xxxxxx.
  :type n_agents: xxxxxx
  :param representation: xxxxxx.
  :type representation: xxxxxx
  :param mixer: xxxxxx.
  :type mixer: xxxxxx
  :param ff_mixer: xxxxxx.
  :type ff_mixer: xxxxxx
  :param hidden_size: xxxxxx.
  :type hidden_size: xxxxxx
  :param normalize: xxxxxx.
  :type normalize: xxxxxx
  :param initialize: xxxxxx.
  :type initialize: xxxxxx
  :param activation: xxxxxx.
  :type activation: xxxxxx
  :param device: xxxxxx.
  :type device: xxxxxx

.. py:function::
  xuance.torch.policies.deterministic_marl.Weighted_MixingQnetwork.q_centralized(observation, agent_ids, *rnn_hidden)

  :param observation: xxxxxx.
  :type observation: xxxxxx
  :param agent_ids: xxxxxx.
  :type agent_ids: xxxxxx
  :param *rnn_hidden: xxxxxx.
  :type *rnn_hidden: xxxxxx
  :return: xxxxxx.
  :rtype: xxxxxx

.. py:function::
  xuance.torch.policies.deterministic_marl.Weighted_MixingQnetwork.target_q_centralized(observation, agent_ids, *rnn_hidden)

  :param observation: xxxxxx.
  :type observation: xxxxxx
  :param agent_ids: xxxxxx.
  :type agent_ids: xxxxxx
  :param *rnn_hidden: xxxxxx.
  :type *rnn_hidden: xxxxxx
  :return: xxxxxx.
  :rtype: xxxxxx

.. py:function::
  xuance.torch.policies.deterministic_marl.Weighted_MixingQnetwork.copy_target()

  :return: None.
  :rtype: xxxxxx

.. py:class::
  xuance.torch.policies.deterministic_marl.Qtran_MixingQnetwork(action_space, n_agents, representation, mixer, qtran_mixer, hidden_size, normalize, initialize, activation, device)

  :param action_space: xxxxxx.
  :type action_space: xxxxxx
  :param n_agents: xxxxxx.
  :type n_agents: xxxxxx
  :param representation: xxxxxx.
  :type representation: xxxxxx
  :param mixer: xxxxxx.
  :type mixer: xxxxxx
  :param qtran_mixer: xxxxxx.
  :type qtran_mixer: xxxxxx
  :param critic_hidden_size: xxxxxx.
  :type critic_hidden_size: xxxxxx
  :param normalize: xxxxxx.
  :type normalize: xxxxxx
  :param initialize: xxxxxx.
  :type initialize: xxxxxx
  :param activation: xxxxxx.
  :type activation: xxxxxx
  :param device: xxxxxx.
  :type device: xxxxxx

.. py:function::
  xuance.torch.policies.deterministic_marl.Qtran_MixingQnetwork.forward(observation, agent_ids)

  :param observation: xxxxxx.
  :type observation: xxxxxx
  :param agent_ids: xxxxxx.
  :type agent_ids: xxxxxx
  :return: xxxxxx.
  :rtype: xxxxxx

.. py:function::
  xuance.torch.policies.deterministic_marl.Qtran_MixingQnetwork.target_Q(observation, agent_ids)

  :param observation: xxxxxx.
  :type observation: xxxxxx
  :param agent_ids: xxxxxx.
  :type agent_ids: xxxxxx
  :return: xxxxxx.
  :rtype: xxxxxx

.. py:function::
  xuance.torch.policies.deterministic_marl.Qtran_MixingQnetwork.copy_target()

  :return: None.
  :rtype: xxxxxx

.. py:class::
 xuance.torch.policies.deterministic_marl.DCG_policy(action_space, global_state_dim, representation, utility, payoffs, dcgraph, hidden_size_bias, normalize, initialize, activation, device)

  :param action_space: xxxxxx.
  :type action_space: xxxxxx
  :param global_state_dim: xxxxxx.
  :type global_state_dim: xxxxxx
  :param representation: xxxxxx.
  :type representation: xxxxxx
  :param utility: xxxxxx.
  :type utility: xxxxxx
  :param payoffs: xxxxxx.
  :type payoffs: xxxxxx
  :param hidden_size_bias: xxxxxx.
  :type hidden_size_bias: xxxxxx
  :param normalize: xxxxxx.
  :type normalize: xxxxxx
  :param initialize: xxxxxx.
  :type initialize: xxxxxx
  :param activation: xxxxxx.
  :type activation: xxxxxx
  :param device: xxxxxx.
  :type device: xxxxxx

.. py:function::
  xuance.torch.policies.deterministic_marl.DCG_policy.forward(observation, agent_ids, *rnn_hidden, avail_actions)

  :param observation: xxxxxx.
  :type observation: xxxxxx
  :param agent_ids: xxxxxx.
  :type agent_ids: xxxxxx
  :param *rnn_hidden: xxxxxx.
  :type *rnn_hidden: xxxxxx
  :param avail_actions: xxxxxx.
  :type avail_actions: xxxxxx
  :return: xxxxxx.
  :rtype: xxxxxx

.. py:function::
  xuance.torch.policies.deterministic_marl.DCG_policy.copy_target()

  :return: None.
  :rtype: xxxxxx

.. py:class::
 xuance.torch.policies.deterministic_marl.ActorNet(state_dim, n_agents, action_space, hidden_sizes, normalize, initialize, activation, device)

  :param state_dim: xxxxxx.
  :type state_dim: xxxxxx
  :param n_agents: xxxxxx.
  :type n_agents: xxxxxx
  :param action_space: xxxxxx.
  :type action_space: xxxxxx
  :param hidden_sizes: xxxxxx.
  :type hidden_sizes: xxxxxx
  :param normalize: xxxxxx.
  :type normalize: xxxxxx
  :param initialize: xxxxxx.
  :type initialize: xxxxxx
  :param activation: xxxxxx.
  :type activation: xxxxxx
  :param device: xxxxxx.
  :type device: xxxxxx

.. py:function::
  xuance.torch.policies.deterministic_marl.ActorNet.forward()

  :return: None.
  :rtype: xxxxxx

.. py:class::
 xuance.torch.policies.deterministic_marl.CriticNet(independent, state_dim, n_agents, action_dim, hidden_sizes, normalize, initialize, activation, device)

  :param independent: xxxxxx.
  :type independent: xxxxxx
  :param state_dim: xxxxxx.
  :type state_dim: xxxxxx
  :param n_agents: xxxxxx.
  :type n_agents: xxxxxx
  :param action_dim: xxxxxx.
  :type action_dim: xxxxxx
  :param hidden_sizes: xxxxxx.
  :type hidden_sizes: xxxxxx
  :param normalize: xxxxxx.
  :type normalize: xxxxxx
  :param initialize: xxxxxx.
  :type initialize: xxxxxx
  :param activation: xxxxxx.
  :type activation: xxxxxx
  :param device: xxxxxx.
  :type device: xxxxxx

.. py:function::
  xuance.torch.policies.deterministic_marl.ACriticNet.forward()

  :return: None.
  :rtype: xxxxxx


.. py:class::
 xuance.torch.policies.deterministic_marl.Basic_DDPG_policy(action_space, n_agents, representation, actor_hidden_size, critic_hidden_size, normalize, initialize, activation, device)

  :param action_space: xxxxxx.
  :type action_space: xxxxxx
  :param n_agents: xxxxxx.
  :type n_agents: xxxxxx
  :param representation: xxxxxx.
  :type representation: xxxxxx
  :param actor_hidden_size: xxxxxx.
  :type actor_hidden_size: xxxxxx
  :param critic_hidden_size: xxxxxx.
  :type critic_hidden_size: xxxxxx
  :param normalize: xxxxxx.
  :type normalize: xxxxxx
  :param initialize: xxxxxx.
  :type initialize: xxxxxx
  :param activation: xxxxxx.
  :type activation: xxxxxx
  :param device: xxxxxx.
  :type device: xxxxxx

.. py:function::
  xuance.torch.policies.deterministic_marl.Basic_DDPG_policy.forward(observation, agent_ids)

  :param observation: xxxxxx.
  :type observation: xxxxxx
  :param agent_ids: xxxxxx.
  :type agent_ids: xxxxxx
  :return: None.
  :rtype: xxxxxx

.. py:function::
  xuance.torch.policies.deterministic_marl.Basic_DDPG_policy.critic(observation, actions, agent_ids)

  :param observation: xxxxxx.
  :type observation: xxxxxx
  :param actions: xxxxxx.
  :type actions: xxxxxx
  :param agent_ids: xxxxxx.
  :type agent_ids: xxxxxx
  :return: xxxxxx.
  :rtype: xxxxxx

.. py:function::
  xuance.torch.policies.deterministic_marl.Basic_DDPG_policy.target_critic(observation, actions, agent_ids)

  :param observation: xxxxxx.
  :type observation: xxxxxx
  :param actions: xxxxxx.
  :type actions: xxxxxx
  :param agent_ids: xxxxxx.
  :type agent_ids: xxxxxx
  :return: xxxxxx.
  :rtype: xxxxxx

.. py:function::
  xuance.torch.policies.deterministic_marl.Basic_DDPG_policy.soft_update(tau)

  :param tau: xxxxxx.
  :type tau: xxxxxx
  :return: xxxxxx.
  :rtype: xxxxxx

.. py:class::
 xuance.torch.policies.deterministic_marl.MADDPG_policy(action_space, n_agents, representation, actor_hidden_size, critic_hidden_size, normalize, initialize, activation, device)

  :param action_space: xxxxxx.
  :type action_space: xxxxxx
  :param n_agents: xxxxxx.
  :type n_agents: xxxxxx
  :param representation: xxxxxx.
  :type representation: xxxxxx
  :param actor_hidden_size: xxxxxx.
  :type actor_hidden_size: xxxxxx
  :param critic_hidden_size: xxxxxx.
  :type critic_hidden_size: xxxxxx
  :param normalize: xxxxxx.
  :type normalize: xxxxxx
  :param initialize: xxxxxx.
  :type initialize: xxxxxx
  :param activation: xxxxxx.
  :type activation: xxxxxx
  :param device: xxxxxx.
  :type device: xxxxxx

.. py:function::
  xuance.torch.policies.deterministic_marl.MADDPG_policy.critic(observation, actions, agent_ids)

  :param observation: xxxxxx.
  :type observation: xxxxxx
  :param actions: xxxxxx.
  :type actions: xxxxxx
  :param agent_ids: xxxxxx.
  :type agent_ids: xxxxxx
  :return: xxxxxx.
  :rtype: xxxxxx

.. py:function::
  xuance.torch.policies.deterministic_marl.MADDPG_policy.target_critic(observation, actions, agent_ids)

  :param observation: xxxxxx.
  :type observation: xxxxxx
  :param actions: xxxxxx.
  :type actions: xxxxxx
  :param agent_ids: xxxxxx.
  :type agent_ids: xxxxxx
  :return: xxxxxx.
  :rtype: xxxxxx

.. py:class::
 xuance.torch.policies.deterministic_marl.MATD3_policy(action_space, n_agents, representation, actor_hidden_size, critic_hidden_size, normalize, initialize, activation, device)

  :param action_space: xxxxxx.
  :type action_space: xxxxxx
  :param n_agents: xxxxxx.
  :type n_agents: xxxxxx
  :param representation: xxxxxx.
  :type representation: xxxxxx
  :param actor_hidden_size: xxxxxx.
  :type actor_hidden_size: xxxxxx
  :param critic_hidden_size: xxxxxx.
  :type critic_hidden_size: xxxxxx
  :param normalize: xxxxxx.
  :type normalize: xxxxxx
  :param initialize: xxxxxx.
  :type initialize: xxxxxx
  :param activation: xxxxxx.
  :type activation: xxxxxx
  :param device: xxxxxx.
  :type device: xxxxxx

.. py:function::
  xuance.torch.policies.deterministic_marl.MATD3_policy.Qpolicy(observation, actions, agent_ids)

  :param observation: xxxxxx.
  :type observation: xxxxxx
  :param actions: xxxxxx.
  :type actions: xxxxxx
  :param agent_ids: xxxxxx.
  :type agent_ids: xxxxxx
  :return: xxxxxx.
  :rtype: xxxxxx

.. py:function::
  xuance.torch.policies.deterministic_marl.MATD3_policy.Qtarget(observation, actions, agent_ids)

  :param observation: xxxxxx.
  :type observation: xxxxxx
  :param actions: xxxxxx.
  :type actions: xxxxxx
  :param agent_ids: xxxxxx.
  :type agent_ids: xxxxxx
  :return: xxxxxx.
  :rtype: xxxxxx

.. py:function::
  xuance.torch.policies.deterministic_marl.MATD3_policy.Qaction(observation, actions, agent_ids)

  :param observation: xxxxxx.
  :type observation: xxxxxx
  :param actions: xxxxxx.
  :type actions: xxxxxx
  :param agent_ids: xxxxxx.
  :type agent_ids: xxxxxx
  :return: xxxxxx.
  :rtype: xxxxxx

.. py:function::
  xuance.torch.policies.deterministic_marl.MATD3_policy.soft_update()

  :return: None.
  :rtype: xxxxxx

.. raw:: html

    <br><hr>

**TensorFlow:**

.. raw:: html

    <br><hr>

**MindSpore:**

.. raw:: html

    <br><hr>

Source Code
-----------------

.. tabs::

  .. group-tab:: PyTorch

    .. code-block:: python

        from xuance.torch.learners import *


        class A2C_Learner(Learner):
            def __init__(self,
                         policy: nn.Module,
                         optimizer: torch.optim.Optimizer,
                         scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
                         device: Optional[Union[int, str, torch.device]] = None,
                         model_dir: str = "./",
                         vf_coef: float = 0.25,
                         ent_coef: float = 0.005,
                         clip_grad: Optional[float] = None):
                super(A2C_Learner, self).__init__(policy, optimizer, scheduler, device, model_dir)
                self.vf_coef = vf_coef
                self.ent_coef = ent_coef
                self.clip_grad = clip_grad

            def update(self, obs_batch, act_batch, ret_batch, adv_batch):
                self.iterations += 1
                act_batch = torch.as_tensor(act_batch, device=self.device)
                ret_batch = torch.as_tensor(ret_batch, device=self.device)
                adv_batch = torch.as_tensor(adv_batch, device=self.device)
                outputs, a_dist, v_pred = self.policy(obs_batch)
                log_prob = a_dist.log_prob(act_batch)

                a_loss = -(adv_batch * log_prob).mean()
                c_loss = F.mse_loss(v_pred, ret_batch)
                e_loss = a_dist.entropy().mean()

                loss = a_loss - self.ent_coef * e_loss + self.vf_coef * c_loss
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.clip_grad)
                self.optimizer.step()
                if self.scheduler is not None:
                    self.scheduler.step()

                # Logger
                lr = self.optimizer.state_dict()['param_groups'][0]['lr']

                info = {
                    "actor-loss": a_loss.item(),
                    "critic-loss": c_loss.item(),
                    "entropy": e_loss.item(),
                    "learning_rate": lr,
                    "predict_value": v_pred.mean().item()
                }

                return info




  .. group-tab:: TensorFlow

    .. code-block:: python


  .. group-tab:: MindSpore

    .. code-block:: python