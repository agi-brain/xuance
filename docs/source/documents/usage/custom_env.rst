Custom Environments
=============================

XuanCe allows users to build and run their own **custom environments** in addition to the built-in ones.
This feature enables seamless integration between XuanCe's algorithms and your own simulation or application scenarios.

With XuanCe, you can:

- Create a **Single-Agent Environment**, based on a standard Markov Decision Process (MDP).
- Build a **Multi-Agent Environment**, following a Partial Observable Markov Decision Process (POMDP) structure.
- Run your environments directly using XuanCe’s provided algorithms, such as DQN, PPO, IPPO, and more.

This flexibility allows you to test reinforcement learning algorithms on your own tasks,
whether they are robotics control problems, trading simulations, or custom multi-agent interactions.

To get started, please refer to:

- :doc:`Custom Environments: Single-Agent <custom_env/custom_drl_env>`
- :doc:`Custom Environments: Multi-Agent <custom_env/custom_marl_env>`

.. toctree::
   :hidden:
   :maxdepth: 1

   Custom Environments: Single-Agent <custom_env/custom_drl_env>
   Custom Environments: Multi-Agent <custom_env/custom_marl_env>
