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

   <a href="https://www.anaconda.com/download">
        <img alt="Python" src="https://img.shields.io/badge/Python-3.7%7C3.8%7C3.9%7C3.10-yellow">
   </a>
   <a href="https://www.gymlibrary.dev/">
        <img alt="gym" src="https://img.shields.io/badge/gym-%3E%3D0.21.0-blue">
   </a>
   <a href="https://www.gymlibrary.dev/">
        <img alt="gymnasium" src="https://img.shields.io/badge/gymnasium-%3E%3D0.28.1-blue">
   </a>
   <a href="https://pettingzoo.farama.org/">
        <img alt="pettingzoo" src="https://img.shields.io/badge/PettingZoo-%3E%3D1.23.0-blue">
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

XuanCe is designed to simplify the implementation and design process of deep reinforcement learning algorithms.
It could help researchers interested in deep reinforcement learning to quickly understand and grasp the fundamental principles. 
This, in turn, facilitates developers in algorithm development and design. It has the following key features. 

- Highly modularized.
- Easy to learn, and easy to install.
- Felxible for model combination.
- Abundant algorithms with various tasks.
- supports bith DRL and MARL tasks.
- High compatible for different users. (PyTorch, TensorFlow, MindSpore, CPU, GPU, Linux, Windows, MacOS, etc.)
- Fast running speed with vector envrionments.
- Good visualization effect with tensorboard or wandb toolbox.

The Framework of XuanCe
------------------------------------------

The overall framework of XuanCe is shown as below. 

.. image:: _static/figures/xuance_framework.png


XuanCe contains four main parts:

- Part I: Configs. The configurations of hyper-parameters, environments, models, etc.
- Part II: Common tools. Reusable tools that are independent of the choice of DL toolbox.
- Part III: Envrionments. The supported simulated environments.
- Part IV: Algorithms. The key part to build DRL algorithms.
   
.. toctree::
   :hidden:
   :maxdepth: 1
   :caption: Tutorial:

   documents/usage/installation
   documents/usage/basic_usage
   documents/usage/further_usage
   documents/usage/new_envs
   documents/usage/new_algorithm

.. toctree::
   :hidden:
   :maxdepth: 1
   :caption: APIs:

   documents/api/agents
   documents/api/environments
   documents/api/runners
   documents/api/representations
   documents/api/policies
   documents/api/learners
   documents/api/configs
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
   Contribute to XuanCe <https://github.com/agi-brain/xuance/pulls>
   Contribute to the Docs (English) <https://github.com/agi-brain/xuance/tree/master/docs>
   Contribute to the Docs (Chinese) <https://github.com/agi-brain/xuance-docs-zh_CN/tree/master/docs>
