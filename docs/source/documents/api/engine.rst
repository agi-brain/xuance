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

.. toctree::
    :hidden:

    run_basic <engine/run_basic>
    run_drl <engine/run_drl>
    run_marl <engine/run_marl>
    run_sc2 <engine/run_sc2>
    run_football <engine/run_football>
    run_competition <engine/run_competition>
    run_offlinerl <engine/run_offlinerl>

* :doc:`run_basic <engine/run_basic>`.
* :doc:`run_drl <engine/run_drl>`.
* :doc:`run_marl <engine/run_marl>`.
* :doc:`run_sc2 <engine/run_sc2>`.
* :doc:`run_football <engine/run_football>`.
* :doc:`run_competition <engine/run_competition>`.
* :doc:`run_offlinerl <engine/run_offlinerl>`.



