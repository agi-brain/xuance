.. XuanPolicy documentation master file, created by
   sphinx-quickstart on Wed May 31 20:18:19 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to XuanPolicy's documentation!
======================================

**XuanPolicy** is an open-source ensemble of Deep Reinforcement Learning (DRL) algorithm implementations.

We call it as **Xuan-Ce (玄策)** in Chinese.
"**Xuan (玄)**" means incredible and magic box, "**Ce (策)**" means policy.

DRL algorithms are sensitive to hyper-parameters tuning, varying in performance with different tricks,
and suffering from unstable training processes, therefore, sometimes DRL algorithms seems elusive and "Xuan".
This project gives a thorough, high-quality and easy-to-understand implementation of DRL algorithms,
and hope this implementation can give a hint on the magics of reinforcement learning.

We expect it to be compatible with multiple deep learning toolboxes(
PyTorch_,
TensorFlow_, and
MindSpore_,
and hope it can really become a zoo full of DRL algorithms.

.. _PyTorch: https://pytorch.org/
.. _TensorFlow: https://www.tensorflow.org/
.. _MindSpore: https://www.mindspore.cn/en

Currently, THe XuanPolicy has been open-sourced on GitHub and OpenI:

| **GitHub**： `https://github.com/agi-brain/xuanpolicy.git <https://github.com/agi-brain/xuanpolicy.git/>`_
| **OpenI**： `https://github.com/agi-brain/xuanpolicy.git <https://github.com/agi-brain/xuanpolicy.git/>`_

.. raw:: html

   <br><hr>

.. toctree::
   :maxdepth: 1
   :caption: How to use:

   documents/usage/installation
   documents/usage/basic_usage
   documents/usage/professional_usage

.. toctree::
   :maxdepth: 1
   :caption: API:

   documents/api/agents.rst
   documents/api/representations
   documents/api/policies
   documents/api/learners
   documents/api/runners
   documents/api/utils
   documents/api/configs
   documents/api/common
   documents/api/environments

.. toctree::
   :maxdepth: 1
   :caption: Benchmarks

   documents/benchmark/toy
   documents/benchmark/mujoco
   documents/benchmark/atari
   documents/benchmark/mpe
   documents/benchmark/magent

.. toctree::
   :maxdepth: 1
   :caption: Algorithms:

   documents/algorithms/index_drl
   documents/algorithms/index_marl

.. raw:: html

   <br><hr>

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
