environments
========================

The software includes single agent task simulation environments such as Atari, Mujoco, Classic Control, and Box2D under gym.
It also includes multi-agent task simulation environments such as MPE and SISL under the open-source environment PettingZoo, StarCraft2, MAgent2, Google Football, etc.
Each simulation environment contains a rich variety of task scenarios, as shown in the table below.

.. toctree::
    :hidden:

    single_agent_env <environments/single_agent_env>
    multi_agent_env <environments/multi_agent_env>
    vectorization <environments/vector_envs>
    utils <environments/utils>

* :doc:`single_agent_env <environments/single_agent_env>`.
* :doc:`multi_agent_env <environments/multi_agent_env>`.
* :doc:`vector_envs <environments/vector_envs>`.
* :doc:`utils <environments/utils>`.

Make Environment
----------------------

.. automodule:: xuance.environment
    :members: make_envs
    :undoc-members:
    :show-inheritance:
