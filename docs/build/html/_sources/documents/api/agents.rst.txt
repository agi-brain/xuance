Agents
======================

.. toctree::
  :hidden:

  Agent <agents/drl/basic_drl_class>
  MARLAgents <agents/marl/basic_marl_class>
  DQN_Agent <agents/drl/dqn>
  C51_Agent <agents/drl/c51>
  DDQN_Agent <agents/drl/ddqn>
  DuelDQN_Agent <agents/drl/dueldqn>
  NoisyDQN_Agent <agents/drl/noisydqn>
  PerDQN_Agent <agents/drl/perdqn>
  QRDQN_Agent <agents/drl/qrdqn>
  PG_Agent <agents/drl/pg>
  PPG_Agent <agents/drl/ppg>
  PPOCLIP_Agent <agents/drl/ppo_clip>
  PPOCKL_Agent <agents/drl/ppo_kl>
  PDQN_Agent <agents/drl/pdqn>
  SPDQN_Agent <agents/drl/spdqn>
  MPDQN_Agent <agents/drl/mpdqn>
  A2C_Agent <agents/drl/a2c>
  SAC_Agent <agents/drl/sac>
  SACDIS_Agent <agents/drl/sac_dis>
  DDPG_Agent <agents/drl/ddpg>
  TD3_Agent <agents/drl/td3>

  IQL_Agents <agents/marl/iql>
  VDN_Agents <agents/marl/vdn>
  QMIX_Agents <agents/marl/qmix>
  WQMIX_Agents <agents/marl/wqmix>
  QTRAN_Agents <agents/marl/qtran>
  DCG_Agents <agents/marl/dcg>
  IDDPG_Agents <agents/marl/iddpg>
  MADDPG_Agents <agents/marl/maddpg>
  ISAC_Agents <agents/marl/isac>
  MASAC_Agents <agents/marl/masac>
  IPPO_Agents <agents/marl/ippo>
  MAPPO_Agents <agents/marl/mappo>
  MATD3_Agents <agents/marl/matd3>
  VDAC_Agents <agents/marl/vdac>
  COMA_Agents <agents/marl/coma>
  MFQ_Agents <agents/marl/mfq>
  MFAC_Agents <agents/marl/mfac>

.. raw:: html

   <br><hr>

强化学习Agents（智能体）是能够与环境进行交互的、具有自主决策能力和自主学习能力的独立单元。
在与环境交互过程中，Agents获取观测信息，根据观测信息计算出动作信息并执行该动作，使得环境进入下一步状态。
通过不断和环境进行交互，Agents收集经验数据，再根据经验数据训练模型，从而获得更优的策略。
以下列出了“玄策”平台中包含的单&多智能体强化学习Agents。


.. list-table:: 
   :header-rows: 1

   * - Agent
     - PyTorch 
     - TensorFlow
     - MindSpore
   * - :doc:`DQN <agents/drl/dqn>`: Deep Q-Networks
     - .. centered:: :math:`\checkmark`
     - .. centered:: :math:`\checkmark`
     - .. centered:: :math:`\checkmark`
   * - :doc:`C51DQN <agents/drl/c51>`: Distributional Reinforcement Learning
     - .. centered:: :math:`\checkmark`
     - .. centered:: :math:`\checkmark`
     - .. centered:: :math:`\checkmark`
   * - :doc:`Double DQN <agents/drl/ddqn>`: DQN with Double Q-learning
     - .. centered:: :math:`\checkmark`
     - .. centered:: :math:`\checkmark`
     - .. centered:: :math:`\checkmark`
   * - :doc:`Dueling DQN <agents/drl/dueldqn>`: DQN with Dueling network
     - .. centered:: :math:`\checkmark`
     - .. centered:: :math:`\checkmark`
     - .. centered:: :math:`\checkmark`
   * - :doc:`Noisy DQN <agents/drl/noisydqn>`: DQN with Parameter Space Noise
     - .. centered:: :math:`\checkmark`
     - .. centered:: :math:`\checkmark`
     - .. centered:: :math:`\checkmark`
   * - :doc:`PERDQN <agents/drl/perdqn>`: DQN with Prioritized Experience Replay
     - .. centered:: :math:`\checkmark`
     - .. centered:: :math:`\checkmark`
     - .. centered:: :math:`\checkmark`
   * - :doc:`QRDQN <agents/drl/qrdqn>`: DQN with Quantile Regression
     - .. centered:: :math:`\checkmark`
     - .. centered:: :math:`\checkmark`
     - .. centered:: :math:`\checkmark`
   * - :doc:`VPG <agents/drl/pg>`: Vanilla Policy Gradient
     - .. centered:: :math:`\checkmark`
     - .. centered:: :math:`\checkmark`
     - .. centered:: :math:`\checkmark`
   * - :doc:`PPG <agents/drl/ppg>`: Phasic Policy Gradient
     - .. centered:: :math:`\checkmark`
     - .. centered:: :math:`\checkmark`
     - .. centered:: :math:`\checkmark`
   * - :doc:`PPO <agents/drl/ppo_clip>`: Proximal Policy Optimization
     - .. centered:: :math:`\checkmark`
     - .. centered:: :math:`\checkmark`
     - .. centered:: :math:`\checkmark`
   * - :doc:`PDQN <agents/drl/pdqn>`: Parameterised DQN
     - .. centered:: :math:`\checkmark`
     - .. centered:: :math:`\checkmark`
     - .. centered:: :math:`\checkmark`
   * - :doc:`SPDQN <agents/drl/spdqn>`: Split PDQN
     - .. centered:: :math:`\checkmark`
     - .. centered:: :math:`\checkmark`
     - .. centered:: :math:`\checkmark`
   * - :doc:`MPDQN <agents/drl/mpdqn>`: Multi-pass PDQN
     - .. centered:: :math:`\checkmark`
     - .. centered:: :math:`\checkmark`
     - .. centered:: :math:`\checkmark`
   * - :doc:`A2C <agents/drl/a2c>`: Advantage Actor Critic
     - .. centered:: :math:`\checkmark`
     - .. centered:: :math:`\checkmark`
     - .. centered:: :math:`\checkmark`
   * - :doc:`SAC <agents/drl/sac>`: Soft Actor-Critic
     - .. centered:: :math:`\checkmark`
     - .. centered:: :math:`\checkmark`
     - .. centered:: :math:`\checkmark`
   * - :doc:`SAC-Dis <agents/drl/sac_dis>`: SAC for Discrete Actions
     - .. centered:: :math:`\checkmark`
     - .. centered:: :math:`\checkmark`
     - .. centered:: :math:`\checkmark`
   * - :doc:`DDPG <agents/drl/ddpg>`: Deep Deterministic Policy Gradient
     - .. centered:: :math:`\checkmark`
     - .. centered:: :math:`\checkmark`
     - .. centered:: :math:`\checkmark`
   * - :doc:`TD3 <agents/drl/td3>`: Twin Delayed DDPG
     - .. centered:: :math:`\checkmark`
     - .. centered:: :math:`\checkmark`
     - .. centered:: :math:`\checkmark`


.. list-table:: 
   :header-rows: 1

   * - Multi-Agent
     - PyTorch 
     - TensorFlow
     - MindSpore
   * - :doc:`IQL <agents/drl/td3>`: Independent Q-Learning
     - .. centered:: :math:`\checkmark`
     - .. centered:: :math:`\checkmark`
     - .. centered:: :math:`\checkmark`
   * - :doc:`VDN <agents/drl/td3>`: Value-Decomposition Networks
     - .. centered:: :math:`\checkmark`
     - .. centered:: :math:`\checkmark`
     - .. centered:: :math:`\checkmark`
   * - :doc:`QMIX <agents/drl/td3>`: VDN with Q-Mixer
     - .. centered:: :math:`\checkmark`
     - .. centered:: :math:`\checkmark`
     - .. centered:: :math:`\checkmark`
   * - :doc:`WQMIX <agents/drl/td3>`: Weighted QMIX
     - .. centered:: :math:`\checkmark`
     - .. centered:: :math:`\checkmark`
     - .. centered:: :math:`\checkmark`
   * - :doc:`QTRAN <agents/drl/td3>`: Q-Transformation
     - .. centered:: :math:`\checkmark`
     - .. centered:: :math:`\checkmark`
     - .. centered:: :math:`\checkmark`
   * - :doc:`DCG <agents/drl/td3>`: Deep Coordination Graph
     - .. centered:: :math:`\checkmark`
     - .. centered:: :math:`\checkmark`
     - .. centered:: :math:`\checkmark`
   * - :doc:`IDDPG <agents/drl/td3>`: Independent DDPG
     - .. centered:: :math:`\checkmark`
     - .. centered:: :math:`\checkmark`
     - .. centered:: :math:`\checkmark`
   * - :doc:`MADDPG <agents/drl/td3>`: Multi-Agent DDPG
     - .. centered:: :math:`\checkmark`
     - .. centered:: :math:`\checkmark`
     - .. centered:: :math:`\checkmark`
   * - :doc:`ISAC <agents/drl/td3>`: Independent SAC
     - .. centered:: :math:`\checkmark`
     - .. centered:: :math:`\checkmark`
     - .. centered:: :math:`\checkmark`
   * - :doc:`MASAC <agents/drl/td3>`: Multi-Agent SAC
     - .. centered:: :math:`\checkmark`
     - .. centered:: :math:`\checkmark`
     - .. centered:: :math:`\checkmark`
   * - :doc:`IPPO <agents/drl/td3>`: Independent PPO
     - .. centered:: :math:`\checkmark`
     - .. centered:: :math:`\checkmark`
     - .. centered:: :math:`\checkmark`
   * - :doc:`MAPPO <agents/drl/td3>`: Multi-Agent PPO
     - .. centered:: :math:`\checkmark`
     - .. centered:: :math:`\checkmark`
     - .. centered:: :math:`\checkmark`
   * - :doc:`MATD3 <agents/drl/td3>`: Multi-Agent TD3
     - .. centered:: :math:`\checkmark`
     - .. centered:: :math:`\checkmark`
     - .. centered:: :math:`\checkmark`
   * - :doc:`VDAC <agents/drl/td3>`: Value-Decomposition Actor-Critic
     - .. centered:: :math:`\checkmark`
     - .. centered:: :math:`\checkmark`
     - .. centered:: :math:`\checkmark`
   * - :doc:`COMA <agents/drl/td3>`: Counterfacutal Multi-Agent PG
     - .. centered:: :math:`\checkmark`
     - .. centered:: :math:`\checkmark`
     - .. centered:: :math:`\checkmark`
   * - :doc:`MFQ <agents/drl/td3>`: Mean-Field Q-Learning
     - .. centered:: :math:`\checkmark`
     - .. centered:: :math:`\checkmark`
     - .. centered:: :math:`\checkmark`
   * - :doc:`MFAC <agents/drl/td3>`: Mean-Field Actor-Critic
     - .. centered:: :math:`\checkmark`
     - .. centered:: :math:`\checkmark`
     - .. centered:: :math:`\checkmark`

.. raw:: html

   <br><hr>
