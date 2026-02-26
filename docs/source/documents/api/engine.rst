engine
=============================

The :mod:`xuance.engine` package provides the **backend-agnostic execution layer**
of Xuance. It is responsible for orchestrating the lifecycle of an experiment,
including training, evaluation, benchmarking, logging, and result management,
while remaining independent of the underlying deep learning framework.

In Xuance, the Engine coordinates *how* an experiment runs, whereas backend
implementations (``xuance.torch``, ``xuance.tensorflow``, ``xuance.mindspore``)
define *how algorithms are implemented*. This separation allows the same
experiment workflow to be executed across multiple backends without modifying
the high-level logic.

Key responsibilities of the Engine include:

- Managing experiment modes such as ``train``, ``test``, and ``benchmark``.

- Creating environments and connecting them with backend-specific agents.

- Driving the interaction loop by invoking :meth:`agent.train()` or :meth:`agent.test()`.

- Handling logging, checkpointing, and result persistence.

- Providing task-specific runners for different environments (e.g., DRL, multi-agent settings, or domain-specific benchmarks).

By isolating orchestration from implementation, Xuance ensures a clean
separation between experiment control and algorithm realization, improving
maintainability, extensibility, and cross-backend reproducibility.

run_basic
-----------------------------

.. automodule:: xuance.engine.run_basic
    :members:
    :undoc-members:
    :show-inheritance:

run_competition
-----------------------------

.. automodule:: xuance.engine.run_competition
    :members:
    :undoc-members:
    :show-inheritance:

run_drl
-----------------------------

.. automodule:: xuance.engine.run_drl
    :members:
    :undoc-members:
    :show-inheritance:

run_football
-----------------------------

.. automodule:: xuance.engine.run_football
    :members:
    :undoc-members:
    :show-inheritance:

run_marl
-----------------------------

.. automodule:: xuance.engine.run_marl
    :members:
    :undoc-members:
    :show-inheritance:

run_offlinerl
-----------------------------

.. automodule:: xuance.engine.run_offlinerl
    :members:
    :undoc-members:
    :show-inheritance:

run_pettingzoo
-----------------------------

.. automodule:: xuance.engine.run_pettingzoo
    :members:
    :undoc-members:
    :show-inheritance:

run_sc2
-----------------------------

.. automodule:: xuance.engine.run_sc2
    :members:
    :undoc-members:
    :show-inheritance:
