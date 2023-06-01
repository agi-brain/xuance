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

.. toctree::
   :maxdepth: 1
   :caption: How to use:

   documents/usage/installation
   documents/usage/basic_usage
   documents/usage/professional_usage

.. toctree::
   :maxdepth: 1
   :caption: Included algorithms:

   documents/agents/index_drl
   documents/agents/index_marl

.. toctree::
   :maxdepth: 1
   :caption: Benchmark

   documents/benchmark/environments

.. toctree::
   :maxdepth: 1
   :caption: API tutorials:

   documents/components/configs


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
