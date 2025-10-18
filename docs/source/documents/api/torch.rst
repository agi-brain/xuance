torch
======================

XuanCe supports multiple deep learning frameworks for reinforcement learning research and development.
Among them, PyTorch serves as the primary backend for most algorithm implementations.

All APIs introduced in this section are implemented based on PyTorch, ensuring high flexibility, modularity, and compatibility with mainstream PyTorch utilities (such as torch.nn, torch.optim, and torch.utils.data).
Users can easily extend or modify these APIs to build custom agents, policies, or learners using the familiar PyTorch ecosystem.

.. toctree::
    :hidden:

    agents <torch/agents>
    communications <torch/communications>
    learners <torch/learners>
    policies <torch/policies>
    representations <torch/representations>
    runners <torch/runners>
    utils <torch/utils>

* :doc:`agents <torch/agents>`.
* :doc:`communications <torch/communications>`.
* :doc:`learners <torch/learners>`.
* :doc:`policies <torch/policies>`.
* :doc:`representations <torch/representations>`.
* :doc:`runners <torch/runners>`.
* :doc:`utils <torch/utils>`.

