Multi-Agent Particle Environment
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The Multi-Agent Particle Environment (MPE) is a lightweight and customizable simulation environment designed
for multi-agent reinforcement learning (MARL).
It provides a suite of cooperative, competitive, and mixed-scenario tasks
that can be used to test and develop algorithms for learning in multi-agent systems.
The environment emphasizes simplicity and modularity,
making it accessible for researchers to extend and adapt to their specific needs.

.. image:: ../../../../_static/figures/mpe/mpe_simple_spread.gif
    :height: 150px
.. image:: ../../../../_static/figures/mpe/mpe_simple_push.gif
    :height: 150px
.. image:: ../../../../_static/figures/mpe/mpe_simple_reference.gif
    :height: 150px
.. image:: ../../../../_static/figures/mpe/mpe_simple_world_comm.gif
    :height: 150px

Installation
''''''''''''''''''

MPE is included as part of the PettingZoo library. To install it, run:

.. code-block:: bash

    pip install pettingzoo[mpe]

To verify your installation, run:

.. code-block:: bash

    from pettingzoo.mpe import simple_v3

    env = simple_v3.parallel_env()
    env.reset()
    print("MPE environment loaded successfully!")

List of available MPE environments included in PettingZoo:

- **simple_v3**: Cooperative navigation.
- **simple_spread_v3**: Cooperative navigation.
- **simple_push_v3**: Cooperative-push task.
- **simple_adversary_v3**: Adversarial navigation.
- **simple_tag_v3**: Predator-prey environment.
- **simple_reference_v3**: Communication task.
- **simple_crypto_v3**: Communication task.
- **simple_speaker_listener**: Communication task.
- **simple_world_comm**: Communication task.


To learn more about MPE, refer to the PettingZoo documentation:
`https://pettingzoo.farama.org/environments/mpe <https://pettingzoo.farama.org/environments/mpe/>`_

Citation
''''''''''''''''''

Paper published in Neural Information Processing Systems (NeurIPs) 2017 (or NIPS 2017):

.. code-block:: bash

    @article{lowe2017multi,
      title={Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments},
      author={Lowe, Ryan and Wu, Yi and Tamar, Aviv and Harb, Jean and Abbeel, Pieter and Mordatch, Igor},
      journal={Neural Information Processing Systems (NIPS)},
      year={2017}
    }

Original particle world environment:

.. code-block:: bash

    @article{mordatch2017emergence,
      title={Emergence of Grounded Compositional Language in Multi-Agent Populations},
      author={Mordatch, Igor and Abbeel, Pieter},
      journal={arXiv preprint arXiv:1703.04908},
      year={2017}
    }

APIs
''''''''''''''''''

.. automodule:: xuance.environment.multi_agent_env.mpe
    :members:
    :undoc-members:
    :show-inheritance:

