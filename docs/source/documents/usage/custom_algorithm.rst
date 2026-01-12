Custom Algorithms
=============================

XuanCe provides a flexible framework that allows users to **design and implement their own reinforcement learning algorithms**.
In addition to the extensive collection of built-in algorithms, users can develop new methods by leveraging XuanCe's modular design.

All core algorithmic components in XuanCe—such as neural network architectures, replay buffers, optimizers, and policies—are implemented in an extensible and reusable way.
This means you can easily create a new algorithm by customizing or combining existing modules, rather than rebuilding everything from scratch.

With XuanCe, you can:

- Develop **Single-Agent Algorithms**, based on standard reinforcement learning frameworks such as DQN, PPO, SAC, and TD3, etc.
- Build **Multi-Agent Algorithms**, extending from architectures such as QMIX, MAPPO, and MADDPG, etc.
- Directly compare your custom algorithms with XuanCe's well-tested baselines using the same training and evaluation pipeline.

This design provides significant advantages for research and experimentation:

- You can focus on the **innovation** of your method rather than reimplementing basic RL infrastructure.
- You can **reuse** the rich collection of benchmark algorithms in XuanCe as **baselines**, saving substantial time and effort.
- You can **test and visualize** results seamlessly using the unified interfaces for logging, replay, and evaluation.

To get started, please refer to:

- :doc:`Custom Algorithms: DRL <custom_algorithm/custom_drl_algorithm>`
- :doc:`Custom Algorithms: MARL <custom_algorithm/custom_marl_algorithm>`

.. toctree::
   :hidden:
   :maxdepth: 1

   DRL <custom_algorithm/custom_drl_algorithm>
   MARL <custom_algorithm/custom_marl_algorithm>
