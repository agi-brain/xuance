Learners
======================

.. toctree::
  :hidden:

  Learner <learners/learner>
  DQN_Learner <learners/drl/dqn>
  C51_Learner <learners/drl/c51>
  DDQN_Learner <learners/drl/ddqn>
  DuelDQN_Learner <learners/drl/dueldqn>
  NoisyDQN_Learner <learners/drl/noisydqn>
  PerDQN_Learner <learners/drl/perdqn>
  QRDQN_Learner <learners/drl/qrdqn>
  PG_Learner <learners/drl/pg>
  PPG_Learner <learners/drl/ppg>
  PPOCLIP_Learner <learners/drl/ppo_clip>
  PPOCKL_Learner <learners/drl/ppo_kl>
  PDQN_Learner <learners/drl/pdqn>
  SPDQN_Learner <learners/drl/spdqn>
  MPDQN_Learner <learners/drl/mpdqn>
  A2C_Learner <learners/drl/a2c>
  SAC_Learner <learners/drl/sac>
  SACDIS_Learner <learners/drl/sac_dis>
  DDPG_Learner <learners/drl/ddpg>
  TD3_Learner <learners/drl/td3>

  IQL_Learner <learners/marl/iql>
  VDN_Learner <learners/marl/vdn>
  QMIX_Learner <learners/marl/qmix>
  WQMIX_Learner <learners/marl/wqmix>
  QTRAN_Learner <learners/marl/qtran>
  DCG_Learner <learners/marl/dcg>
  IDDPG_Learner <learners/marl/iddpg>
  MADDPG_Learner <learners/marl/maddpg>
  ISAC_Learner <learners/marl/isac>
  MASAC_Learner <learners/marl/masac>
  IPPO_Learner <learners/marl/ippo>
  MAPPO_Learner <learners/marl/mappo>
  MATD3_Learner <learners/marl/matd3>
  VDAC_Learner <learners/marl/vdac>
  COMA_Learner <learners/marl/coma>
  MFQ_Learner <learners/marl/mfq>
  MFAC_Learner <learners/marl/mfac>


.. list-table:: 
   :header-rows: 1

   * - Learner
     - PyTorch 
     - TensorFlow
     - MindSpore
   * - :doc:`DQN <learners/drl/dqn>`: Deep Q-Networks
     - .. centered:: :math:`\checkmark`
     - .. centered:: :math:`\checkmark`
     - .. centered:: :math:`\checkmark`
   * - :doc:`C51DQN <learners/drl/c51>`: Distributional Reinforcement Learning
     - .. centered:: :math:`\checkmark`
     - .. centered:: :math:`\checkmark`
     - .. centered:: :math:`\checkmark`
   * - :doc:`Double DQN <learners/drl/ddqn>`: DQN with Double Q-learning
     - .. centered:: :math:`\checkmark`
     - .. centered:: :math:`\checkmark`
     - .. centered:: :math:`\checkmark`
   * - :doc:`Dueling DQN <learners/drl/dueldqn>`: DQN with Dueling network
     - .. centered:: :math:`\checkmark`
     - .. centered:: :math:`\checkmark`
     - .. centered:: :math:`\checkmark`
   * - :doc:`Noisy DQN <learners/drl/noisydqn>`: DQN with Parameter Space Noise
     - .. centered:: :math:`\checkmark`
     - .. centered:: :math:`\checkmark`
     - .. centered:: :math:`\checkmark`
   * - :doc:`PERDQN <learners/drl/perdqn>`: DQN with Prioritized Experience Replay
     - .. centered:: :math:`\checkmark`
     - .. centered:: :math:`\checkmark`
     - .. centered:: :math:`\checkmark`
   * - :doc:`QRDQN <learners/drl/qrdqn>`: DQN with Quantile Regression
     - .. centered:: :math:`\checkmark`
     - .. centered:: :math:`\checkmark`
     - .. centered:: :math:`\checkmark`
   * - :doc:`VPG <learners/drl/pg>`: Vanilla Policy Gradient
     - .. centered:: :math:`\checkmark`
     - .. centered:: :math:`\checkmark`
     - .. centered:: :math:`\checkmark`
   * - :doc:`PPG <learners/drl/ppg>`: Phasic Policy Gradient
     - .. centered:: :math:`\checkmark`
     - .. centered:: :math:`\checkmark`
     - .. centered:: :math:`\checkmark`
   * - :doc:`PPO <learners/drl/ppo_clip>`: Proximal Policy Optimization
     - .. centered:: :math:`\checkmark`
     - .. centered:: :math:`\checkmark`
     - .. centered:: :math:`\checkmark`
   * - :doc:`PDQN <learners/drl/pdqn>`: Parameterised DQN
     - .. centered:: :math:`\checkmark`
     - .. centered:: :math:`\checkmark`
     - .. centered:: :math:`\checkmark`
   * - :doc:`SPDQN <learners/drl/spdqn>`: Split PDQN
     - .. centered:: :math:`\checkmark`
     - .. centered:: :math:`\checkmark`
     - .. centered:: :math:`\checkmark`
   * - :doc:`MPDQN <learners/drl/mpdqn>`: Multi-pass PDQN
     - .. centered:: :math:`\checkmark`
     - .. centered:: :math:`\checkmark`
     - .. centered:: :math:`\checkmark`
   * - :doc:`A2C <learners/drl/a2c>`: Advantage Actor Critic
     - .. centered:: :math:`\checkmark`
     - .. centered:: :math:`\checkmark`
     - .. centered:: :math:`\checkmark`
   * - :doc:`SAC <learners/drl/sac>`: Soft Actor-Critic
     - .. centered:: :math:`\checkmark`
     - .. centered:: :math:`\checkmark`
     - .. centered:: :math:`\checkmark`
   * - :doc:`SAC-Dis <learners/drl/sac_dis>`: SAC for Discrete Actions
     - .. centered:: :math:`\checkmark`
     - .. centered:: :math:`\checkmark`
     - .. centered:: :math:`\checkmark`
   * - :doc:`DDPG <learners/drl/ddpg>`: Deep Deterministic Policy Gradient
     - .. centered:: :math:`\checkmark`
     - .. centered:: :math:`\checkmark`
     - .. centered:: :math:`\checkmark`
   * - :doc:`TD3 <learners/drl/td3>`: Twin Delayed DDPG
     - .. centered:: :math:`\checkmark`
     - .. centered:: :math:`\checkmark`
     - .. centered:: :math:`\checkmark`


.. list-table:: 
   :header-rows: 1

   * - Multi-Agent Learner
     - PyTorch 
     - TensorFlow
     - MindSpore
   * - :doc:`IQL <learners/drl/td3>`: Independent Q-Learning
     - .. centered:: :math:`\checkmark`
     - .. centered:: :math:`\checkmark`
     - .. centered:: :math:`\checkmark`
   * - :doc:`VDN <learners/drl/td3>`: Value-Decomposition Networks
     - .. centered:: :math:`\checkmark`
     - .. centered:: :math:`\checkmark`
     - .. centered:: :math:`\checkmark`
   * - :doc:`QMIX <learners/drl/td3>`: VDN with Q-Mixer
     - .. centered:: :math:`\checkmark`
     - .. centered:: :math:`\checkmark`
     - .. centered:: :math:`\checkmark`
   * - :doc:`WQMIX <learners/drl/td3>`: Weighted QMIX
     - .. centered:: :math:`\checkmark`
     - .. centered:: :math:`\checkmark`
     - .. centered:: :math:`\checkmark`
   * - :doc:`QTRAN <learners/drl/td3>`: Q-Transformation
     - .. centered:: :math:`\checkmark`
     - .. centered:: :math:`\checkmark`
     - .. centered:: :math:`\checkmark`
   * - :doc:`DCG <learners/drl/td3>`: Deep Coordination Graph
     - .. centered:: :math:`\checkmark`
     - .. centered:: :math:`\checkmark`
     - .. centered:: :math:`\checkmark`
   * - :doc:`IDDPG <learners/drl/td3>`: Independent DDPG
     - .. centered:: :math:`\checkmark`
     - .. centered:: :math:`\checkmark`
     - .. centered:: :math:`\checkmark`
   * - :doc:`MADDPG <learners/drl/td3>`: Multi-Agent DDPG
     - .. centered:: :math:`\checkmark`
     - .. centered:: :math:`\checkmark`
     - .. centered:: :math:`\checkmark`
   * - :doc:`ISAC <learners/drl/td3>`: Independent SAC
     - .. centered:: :math:`\checkmark`
     - .. centered:: :math:`\checkmark`
     - .. centered:: :math:`\checkmark`
   * - :doc:`MASAC <learners/drl/td3>`: Multi-Agent SAC
     - .. centered:: :math:`\checkmark`
     - .. centered:: :math:`\checkmark`
     - .. centered:: :math:`\checkmark`
   * - :doc:`IPPO <learners/drl/td3>`: Independent PPO
     - .. centered:: :math:`\checkmark`
     - .. centered:: :math:`\checkmark`
     - .. centered:: :math:`\checkmark`
   * - :doc:`MAPPO <learners/drl/td3>`: Multi-Agent PPO
     - .. centered:: :math:`\checkmark`
     - .. centered:: :math:`\checkmark`
     - .. centered:: :math:`\checkmark`
   * - :doc:`MATD3 <learners/drl/td3>`: Multi-Agent TD3
     - .. centered:: :math:`\checkmark`
     - .. centered:: :math:`\checkmark`
     - .. centered:: :math:`\checkmark`
   * - :doc:`VDAC <learners/drl/td3>`: Value-Decomposition Actor-Critic
     - .. centered:: :math:`\checkmark`
     - .. centered:: :math:`\checkmark`
     - .. centered:: :math:`\checkmark`
   * - :doc:`COMA <learners/drl/td3>`: Counterfacutal Multi-Agent PG
     - .. centered:: :math:`\checkmark`
     - .. centered:: :math:`\checkmark`
     - .. centered:: :math:`\checkmark`
   * - :doc:`MFQ <learners/drl/td3>`: Mean-Field Q-Learning
     - .. centered:: :math:`\checkmark`
     - .. centered:: :math:`\checkmark`
     - .. centered:: :math:`\checkmark`
   * - :doc:`MFAC <learners/drl/td3>`: Mean-Field Actor-Critic
     - .. centered:: :math:`\checkmark`
     - .. centered:: :math:`\checkmark`
     - .. centered:: :math:`\checkmark`