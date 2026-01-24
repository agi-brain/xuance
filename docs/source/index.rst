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
        <img alt="docs" src="https://readthedocs.org/projects/xuance/badge/?version=latest">
   </a>
   <a href="https://github.com/agi-brain/xuance/blob/master/LICENSE.txt">
        <img alt="license" src="https://img.shields.io/github/license/agi-brain/xuance">
   </a>
   <a href="https://pepy.tech/project/xuance">
        <img alt="downloads" src="https://static.pepy.tech/badge/xuance">
   </a>
   <a href="https://github.com/agi-brain/xuance/stargazers">
        <img alt="stars" src="https://img.shields.io/github/stars/agi-brain/xuance?style=social">
   </a>
   <a href="https://github.com/agi-brain/xuance/forks">
        <img alt="forks" src="https://img.shields.io/github/forks/agi-brain/xuance?style=social">
   </a>
   <a href="https://github.com/agi-brain/xuance/watchers">
        <img alt="watchers" src="https://img.shields.io/github/watchers/agi-brain/xuance?style=social">
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

   <a href="https://cn.xuance.org">
        <img alt="docs:chinese" src="https://img.shields.io/badge/Docs-%E4%B8%AD%E6%96%87-blue?logo=readthedocs">
   </a>

.. raw:: html

   <br>

**XuanCe** is an open-source ensemble of Deep Reinforcement Learning (DRL) algorithm implementations.
The name "XuanCe" (玄策) comes from two Chinese characters:

- **"Xuan" (玄)** means incredible, mysterious, or profound.
- **"Ce" (策)** means policy or strategy.

Together, XuanCe represents "incredible policies", reflecting the goal of discovering optimal policies through DRL.

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

* :class:`DQN` : :doc:`Deep Q-Network (DQN) <documents/algorithms/drl/dqn>`.
* :class:`DDQN` : :doc:`Double Deep Q-Network (Double DQN) <documents/algorithms/drl/ddqn>`.
* :class:`DuelDQN` : :doc:`Dueling Deep Q-Network (Dueling DQN) <documents/algorithms/drl/dueldqn>`.
* :class:`PerDQN` : :doc:`DQN with Prioritized Experience Replay (PER DQN) <documents/algorithms/drl/perdqn>`.
* :class:`NoisyDQN` : :doc:`DQN with Noisy Layers (Noisy DQN) <documents/algorithms/drl/noisydqn>`.
* :class:`DRQN` : :doc:`Deep Recurrent Q-Network (DRQN) <documents/algorithms/drl/drqn>`.
* :class:`QRDQN` : :doc:`DQN with Quantile Regression (QR-DQN) <documents/algorithms/drl/qrdqn>`.
* :class:`C51` : :doc:`Categorical 51 DQN (C51) <documents/algorithms/drl/c51>`.

**Policy-based:**

* :class:`PG` : :doc:`Policy Gradient (PG) <documents/algorithms/drl/pg>`.
* :class:`NPG` : :doc:`Natural Policy Gradient (NPG) <documents/algorithms/drl/npg>`.
* :class:`PPG` : :doc:`Phasic Policy Gradient (PPG) <documents/algorithms/drl/ppg>`.
* :class:`A2C` : :doc:`Advantage Actor Critic (A2C) <documents/algorithms/drl/a2c>`.
* :class:`SAC` : :doc:`Soft Actor-Critic (SAC) <documents/algorithms/drl/sac>`.
* :class:`PPOCLIP` : :doc:`Proximal Policy Optimization with Clipped Objective (PPO-Clip) <documents/algorithms/drl/ppoclip>`.
* :class:`PPOKL` : :doc:`Proximal Policy Optimization with KL Divergence (PPO-KL) <documents/algorithms/drl/ppokl>`.
* :class:`DDPG` : :doc:`Deep Deterministic Policy Gradient (DDPG) <documents/algorithms/drl/ddpg>`.
* :class:`TD3` : :doc:`Twin Delayed Deep Deterministic Policy Gradient (TD3) <documents/algorithms/drl/td3>`.
* :class:`PDQN` : :doc:`Parameterised Deep Q-Network (P-DQN) <documents/algorithms/drl/pdqn>`.
* :class:`MPDQN` : :doc:`Multi-pass Parameterised Deep Q-Network (MP-DQN) <documents/algorithms/drl/mpdqn>`.
* :class:`SPDQN` : :doc:`Split parameterised Deep Q-Network (SP-DQN) <documents/algorithms/drl/spdqn>`.

**MARL-based:**

* :class:`IQL` : :doc:`Independent Q-Learning (IQL) <documents/algorithms/marl/iql>`.
* :class:`VDN` : :doc:`Value Decomposition Networks (VDN) <documents/algorithms/marl/vdn>`.
* :class:`QMIX` : :doc:`Q-Mixing Networks (QMIX) <documents/algorithms/marl/qmix>`.
* :class:`WQMIX` : :doc:`Weighted Q-Mixing Networks (WQMIX) <documents/algorithms/marl/wqmix>`.
* :class:`QTRAN` : :doc:`Q-Transformation (QTRAN) <documents/algorithms/marl/qtran>`.
* :class:`DCG` : :doc:`Deep Coordination Graphs (DCG) <documents/algorithms/marl/dcg>`.
* :class:`IDDPG` : :doc:`Independent Deep Deterministic Policy Gradient (IDDPG) <documents/algorithms/marl/iddpg>`.
* :class:`MADDPG` : :doc:`Multi-agent Deep Deterministic Policy Gradient (MADDPG) <documents/algorithms/marl/maddpg>`.
* :class:`IAC` : :doc:`Independent Actor-Critic (IAC) <documents/algorithms/marl/iac>`.
* :class:`COMA` : :doc:`Counterfactual Multi-agent Policy Gradient (COMA) <documents/algorithms/marl/coma>`.
* :class:`VDAC` : :doc:`Value-Decomposition Actor-Critic (VDAC) <documents/algorithms/marl/vdac>`.
* :class:`IPPO` : :doc:`Independent Proximal Policy Optimization (IPPO) <documents/algorithms/marl/ippo>`.
* :class:`MAPPO` : :doc:`Multi-agent Proximal Policy Optimization (MAPPO) <documents/algorithms/marl/mappo>`.
* :class:`MFQ` : :doc:`Mean-Field Q-Learning (MFQ) <documents/algorithms/marl/mfq>`.
* :class:`MFAC` : :doc:`Mean-Field Actor-Critic (MFAC) <documents/algorithms/marl/mfac>`.
* :class:`ISAC` : :doc:`Independent Soft Actor-Critic (ISAC) <documents/algorithms/marl/isac>`.
* :class:`MASAC` : :doc:`Multi-agent Soft Actor-Critic (MASAC) <documents/algorithms/marl/masac>`.
* :class:`MATD3` : :doc:`Multi-agent Twin Delayed Deep Deterministic Policy Gradient (MATD3) <documents/algorithms/marl/matd3>`.
* :class:`IC3Net` : :doc:`Individual Controlled Continuous Communication Model (IC3Net) <documents/algorithms/marl/ic3net>`.

**Model-based:**

* :class:`DreamerV2` : :doc:`Dreamer V2 <documents/algorithms/mbrl/dreamer_v2>`.
* :class:`DreamerV3` : :doc:`Dreamer V3 <documents/algorithms/mbrl/dreamer_v3>`.
* :class:`HarmonyDreamer` : :doc:`HarmonyDreamer <documents/algorithms/mbrl/harmony_dream>`.

**Contrastive RL:**

* :class:`CURL` : :doc:`Contrastive Unsupervised Representations for Reinforcement Learning (CURL) <documents/algorithms/crl/curl>`.
* :class:`DrQ` : :doc:`Data-Regularized Q-Learning (DrQ) <documents/algorithms/crl/drq>`.
* :class:`SPR` : :doc:`Self-Predictive Representations for Reinforcement Learning (SPR) <documents/algorithms/crl/spr>`.

**Offline RL:**

* :class:`TD3BC` : :doc:`Twin Delayed Deep Deterministic Policy Gradient with Behavior Cloning (TD3BC) <documents/algorithms/offline/td3bc>`.


The Framework of XuanCe
------------------------------------------

The overall framework of XuanCe is shown as below. 

.. image:: _static/figures/xuance_framework.png


XuanCe contains four main parts:

- Part I: Configs. The configurations of hyper-parameters, environments, models, etc.
- Part II: Common tools. Reusable tools that are independent of the choice of DL toolbox.
- Part III: Environments. The supported simulated environments.
- Part IV: Algorithms. The key part to build DRL algorithms.

Who Is XuanCe For?
-----------------------------------------

XuanCe is designed for a wide range of users, including:

- **Researchers** exploring new reinforcement learning methods
- **Developers** building DRL-based applications
- **Students and beginners** learning about intelligent decision-making
- **AI practitioners** interested in single-agent and multi-agent systems

.. raw:: html

   <br><hr>

Contents
------------------------------------------

.. toctree::
   :maxdepth: 2
   :caption: Tutorial:

   documents/usage/installation
   documents/usage/basic_usage
   documents/usage/further_usage
   documents/usage/custom_env
   documents/usage/custom_algorithm
   documents/usage/custom_callback

.. toctree::
   :maxdepth: 2
   :caption: Algorithms:

   Single-Agent RL <documents/algorithms/drl>
   Multi-Agent RL <documents/algorithms/marl>
   Model-based RL <documents/algorithms/model_based_rl>
   Constructive RL <documents/algorithms/crl>
   Offline RL <documents/algorithms/offline_rl>

.. toctree::
   :maxdepth: 2
   :caption: Benchmarks:

   Start Benchmark <documents/benchmark/start_benchmark>
   Benchmark Results <documents/benchmark/benchmark_results>
   Add New Benchmark <documents/benchmark/add_new_benchmark>

.. toctree::
   :maxdepth: 5
   :caption: APIs:

   documents/api/common
   documents/api/configs
   documents/api/environments
   documents/api/torch
   documents/api/tensorflow
   documents/api/mindspore

.. toctree::
   :caption: Development:

   Github <https://github.com/agi-brain/xuance.git>
   Release Log <documents/release_log>
   Contribute to XuanCe <documents/CONTRIBUTING>
   Contribute to Docs (EN) <https://github.com/agi-brain/xuance/tree/master/docs>
   Contribute to Docs (CN) <https://github.com/agi-brain/xuance-docs-zh_CN/tree/master/docs>
