.. XuanCe documentation master file, created by
   sphinx-quickstart on Wed May 31 20:18:19 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. .. image:: /_static/figures/logo_1.png
   :scale: 30%
   :align: center
   :target: https://github.com/agi-brain/xuance.git

XuanCe: A Comprehensive and Unified Deep Reinforcement Learning Library
========================================================================

.. raw:: html

   <a href="https://pypi.org/project/xuance/">
        <img alt="pypi" src="https://img.shields.io/pypi/v/xuance">
   </a>
   <a href="https://xuance.readthedocs.io">
        <img alt="pypi" src="https://readthedocs.org/projects/xuance/badge/?version=latest">
   </a>
   <a href="https://github.com/agi-brain/xuance/blob/master/LICENSE.txt">
        <img alt="pypi" src="https://img.shields.io/github/license/agi-brain/xuance">
   </a>
   <a href="https://pepy.tech/project/xuance">
        <img alt="pypi" src="https://static.pepy.tech/badge/xuance">
   </a>
   <a href="https://github.com/agi-brain/xuance/stargazers">
        <img alt="pypi" src="https://img.shields.io/github/stars/agi-brain/xuance?style=social">
   </a>
   <a href="https://github.com/agi-brain/xuance/forks">
        <img alt="pypi" src="https://img.shields.io/github/forks/agi-brain/xuance?style=social">
   </a>
   <a href="https://github.com/agi-brain/xuance/watchers">
        <img alt="pypi" src="https://img.shields.io/github/watchers/agi-brain/xuance?style=social">
   </a>

   <a href="https://pytorch.org/get-started/locally/">
        <img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-%3E%3D1.13.0-red">
   </a>
   <a href="https://www.tensorflow.org/install">
        <img alt="TensorFlow" src="https://img.shields.io/badge/TensorFlow-%3E%3D2.6.0-orange">
   </a>
   <a href="https://www.mindspore.cn/install/en">
        <img alt="MindSpore" src="https://img.shields.io/badge/MindSpore-%3E%3D1.10.1-blue">
   </a>

   <a href="https://www.gymlibrary.dev/">
        <img alt="gymnasium" src="https://img.shields.io/badge/gymnasium-%3E%3D0.28.1-blue">
   </a>
   <a href="https://pettingzoo.farama.org/">
        <img alt="pettingzoo" src="https://img.shields.io/badge/PettingZoo-%3E%3D1.23.0-blue">
   </a>
   <a href="https://img.shields.io/pypi/pyversions/xuance">
        <img alt="Python" src="https://img.shields.io/pypi/pyversions/xuance">
   </a>

.. raw:: html

   <br>

**XuanCe** is an open-source ensemble of Deep Reinforcement Learning (DRL) algorithm implementations.
We call it as **Xuan-Ce (玄策)** in Chinese.

- **Xuan (玄)** means incredible, mysterious, and even a bit like "black box" in Chinese.
- **Ce (策)** means policy or strategy, which is at the core of DRL.

DRL algorithms are sensitive to hyper-parameters tuning, varying in performance with different tricks,
and suffering from unstable training processes, therefore, sometimes DRL algorithms seems elusive and "Xuan".
That is why this project exists: to provide clean, easy-to-understand implementations of DRL algorithms.
We hope it can help uncover some of the "magic" behind DRL and make it a bit less mysterious.

We are also working to make XuanCe compatible with popular deep learning frameworks like
PyTorch_ (|_1| |torch| |_1|), TensorFlow_ (|_1| |tensorflow| |_1|), and MindSpore_ (|_1| |mindspore| |_1|).
Our goal is to turn it into a full-fledged DRL "zoo" where you can explore and experiment with a wide variety of algorithms.

.. _PyTorch: https://pytorch.org/
.. _TensorFlow: https://www.tensorflow.org/
.. _MindSpore: https://www.mindspore.cn/en

| **GitHub**: `https://github.com/agi-brain/xuance.git <https://github.com/agi-brain/xuance.git/>`_

Why XuanCe?
-----------------------------------------

XuanCe is designed to streamline the implementation and development of deep reinforcement learning algorithms.
It empowers researchers to quickly grasp fundamental principles,
making it easier to dive into algorithm design and development.
Here are its key features:

- **Highly Modular**: Designed with a modular structure to enhance flexibility and scalability.
- **User-Friendly**: Easy to learn, install, and use, making it accessible for users of all levels.
- **Flexible Model Integration**: Supports seamless combination and customization of models.
- **Diverse Algorithms**: Offers a rich collection of algorithms catering to various tasks.
- **Versatile Task Support**: Handles both deep reinforcement learning (DRL) and multi-agent reinforcement learning (MARL) scenarios.
- **Broad Compatibility**: Supports PyTorch, TensorFlow, MindSpore, and runs efficiently on CPU, GPU, and across Linux, Windows, and macOS.
- **High Performance**: Delivers fast execution speeds, leveraging vectorized environments for efficiency.
- **Distributed Training**: Enables multi-GPU training for scaling up experiments.
- **Hyperparameters Tuning**: Supports automatically hyperparameters tuning.
- **Enhanced Visualization**: Provides intuitive and comprehensive visualization with tools like TensorBoard and Weights & Biases (wandb).

List of Algorithms
-------------------

**Value-based:**

* :class:`DQN_Agent` : :doc:`Deep Q-Network (DQN) <documents/api/agents/drl/dqn_agent>`.
* :class:`DDQN_Agent` : :doc:`Double Deep Q-Network (Double DQN) <documents/api/agents/drl/ddqn_agent>`.
* :class:`DuelDQN_Agent` : :doc:`Dueling Deep Q-Network (Dueling DQN) <documents/api/agents/drl/dueldqn_agent>`.
* :class:`PerDQN_Agent` : :doc:`DQN with Prioritized Experience Replay (PER DQN) <documents/api/agents/drl/perdqn_agent>`.
* :class:`NoisyDQN_Agent` : :doc:`DQN with Noisy Layers (Noisy DQN) <documents/api/agents/drl/noisydqn_agent>`.
* :class:`DRQN_Agent` : :doc:`Deep Recurrent Q-Network (DRQN) <documents/api/agents/drl/drqn_agent>`.
* :class:`QRDQN_Agent` : :doc:`DQN with Quantile Regression (QR-DQN) <documents/api/agents/drl/qrdqn_agent>`.
* :class:`C51_Agent` : :doc:`Categorical 51 DQN (C51) <documents/api/agents/drl/c51_agent>`.

**Policy-based:**

* :class:`PG_Agent` : :doc:`Policy Gradient (PG) <documents/api/agents/drl/pg_agent>`.
* :class:`NPG_Agent` : :doc:`Natural Policy Gradient (NPG) <documents/api/agents/drl/npg_agent>`.
* :class:`PPG_Agent` : :doc:`Phasic Policy Gradient (PPG) <documents/api/agents/drl/ppg_agent>`.
* :class:`A2C_Agent` : :doc:`Advantage Actor Critic (A2C) <documents/api/agents/drl/a2c_agent>`.
* :class:`SAC_Agent` : :doc:`Soft Actor-Critic (SAC) <documents/api/agents/drl/sac_agent>`.
* :class:`PPOCLIP_Agent` : :doc:`Proximal Policy Optimization with Clipped Objective (PPO-Clip) <documents/api/agents/drl/ppoclip_agent>`.
* :class:`PPOKL_Agent` : :doc:`Proximal Policy Optimization with KL Divergence (PPO-KL) <documents/api/agents/drl/ppokl_agent>`.
* :class:`DDPG_Agent` : :doc:`Deep Deterministic Policy Gradient (DDPG) <documents/api/agents/drl/ddpg_agent>`.
* :class:`TD3_Agent` : :doc:`Twin Delayed Deep Deterministic Policy Gradient (TD3) <documents/api/agents/drl/td3_agent>`.
* :class:`PDQN_Agent` : :doc:`Parameterised Deep Q-Network (P-DQN) <documents/api/agents/drl/pdqn_agent>`.
* :class:`MPDQN_Agent` : :doc:`Multi-pass Parameterised Deep Q-Network (MP-DQN) <documents/api/agents/drl/mpdqn_agent>`.
* :class:`SPDQN_Agent` : :doc:`Split parameterised Deep Q-Network (SP-DQN) <documents/api/agents/drl/spdqn_agent>`.

**MARL-based:**

* :class:`IQL_Agents` : :doc:`Independent Q-Learning (IQL) <documents/api/agents/marl/iql_agents>`.
* :class:`VDN_Agents` : :doc:`Value Decomposition Networks (VDN) <documents/api/agents/marl/vdn_agents>`.
* :class:`QMIX_Agents` : :doc:`Q-Mixing Networks (QMIX) <documents/api/agents/marl/qmix_agents>`.
* :class:`WQMIX_Agents` : :doc:`Weighted Q-Mixing Networks (WQMIX) <documents/api/agents/marl/wqmix_agents>`.
* :class:`QTRAN_Agents` : :doc:`Q-Transformation (QTRAN) <documents/api/agents/marl/qtran_agents>`.
* :class:`DCG_Agents` : :doc:`Deep Coordination Graphs (DCG) <documents/api/agents/marl/dcg_agents>`.
* :class:`IDDPG_Agents` : :doc:`Independent Deep Deterministic Policy Gradient (IDDPG) <documents/api/agents/marl/iddpg_agents>`.
* :class:`MADDPG_Agents` : :doc:`Multi-agent Deep Deterministic Policy Gradient (MADDPG) <documents/api/agents/marl/maddpg_agents>`.
* :class:`IAC_Agents` : :doc:`Independent Actor-Critic (IAC) <documents/api/agents/marl/iac_agents>`.
* :class:`COMA_Agents` : :doc:`Counterfactual Multi-agent Policy Gradient (COMA) <documents/api/agents/marl/coma_agents>`.
* :class:`VDAC_Agents` : :doc:`Value-Decomposition Actor-Critic (VDAC) <documents/api/agents/marl/vdac_agents>`.
* :class:`IPPO_Agents` : :doc:`Independent Proximal Policy Optimization (IPPO) <documents/api/agents/marl/ippo_agents>`.
* :class:`MAPPO_Agents` : :doc:`Multi-agent Proximal Policy Optimization (MAPPO) <documents/api/agents/marl/mappo_agents>`.
* :class:`MFQ_Agents` : :doc:`Mean-Field Q-Learning (MFQ) <documents/api/agents/marl/mfq_agents>`.
* :class:`MFAC_Agents` : :doc:`Mean-Field Actor-Critic (MFAC) <documents/api/agents/marl/mfac_agents>`.
* :class:`ISAC_Agents` : :doc:`Independent Soft Actor-Critic (ISAC) <documents/api/agents/marl/isac_agents>`.
* :class:`MASAC_Agents` : :doc:`Multi-agent Soft Actor-Critic (MASAC) <documents/api/agents/marl/masac_agents>`.
* :class:`MATD3_Agents` : :doc:`Multi-agent Twin Delayed Deep Deterministic Policy Gradient (MATD3) <documents/api/agents/marl/matd3_agents>`.
* :class:`IC3Net_Agents` : :doc:`Individual Controlled Continuous Communication Model (IC3Net) <documents/api/agents/marl/ic3net_agents>`.

The Framework of XuanCe
------------------------------------------

The overall framework of XuanCe is shown as below. 

.. image:: _static/figures/xuance_framework.png


XuanCe contains four main parts:

- Part I: Configs. The configurations of hyper-parameters, environments, models, etc.
- Part II: Common tools. Reusable tools that are independent of the choice of DL toolbox.
- Part III: Environments. The supported simulated environments.
- Part IV: Algorithms. The key part to build DRL algorithms.

.. toctree::
   :hidden:
   :maxdepth: 1
   :caption: Tutorial:

   documents/usage/installation
   documents/usage/basic_usage
   documents/usage/further_usage
   documents/usage/custom_drl_envs
   documents/usage/custom_marl_envs
   documents/usage/custom_algorithm
   documents/usage/custom_callback

APIs of XuanCe
----------------

.. toctree::
   :maxdepth: 1
   :caption: APIs:

   documents/api/agents
   documents/api/environments
   documents/api/configs
   documents/api/runners
   documents/api/representations
   documents/api/policies
   documents/api/learners
   documents/api/common
   documents/api/utils

.. toctree::
   :hidden:
   :maxdepth: 1
   :caption: Benchmarks

   documents/benchmark/mujoco
   documents/benchmark/atari
   documents/benchmark/smac

.. toctree::
   :hidden:
   :caption: Development

   Github <https://github.com/agi-brain/xuance.git>
   documents/release_log
   documents/CONTRIBUTING
   Contribute to the Docs (EN) <https://github.com/agi-brain/xuance/tree/master/docs>
   Contribute to the Docs (CN) <https://github.com/agi-brain/xuance-docs-zh_CN/tree/master/docs>
