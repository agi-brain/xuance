StarCraft Multi-Agent Challenge
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. image:: ../../../../_static/figures/smac/smac.png

The StarCraft multi-agent challenge (SMAC) is `WhiRL's <https://whirl.cs.ox.ac.uk/>`_ environment for research of cooperative MARL algorithms.
SMAC uses StarCraft II, a real-time strategy game developed by Blizzard Entertainment, as its underlying environment.

Installation
''''''''''''''''

Step 1: Install the SMAC python package
.........................................

You can install the SMAC directly from the GitHub:

.. code-block:: bash

    pip install git+https://github.com/oxwhirl/smac.git

Or you can clone the GitHub repository and install it with its dependencies:

.. code-block:: bash

    git clone https://github.com/oxwhirl/smac.git
    cd smac/
    pip install -e .

Step 2: Install StarCraft II
...............................

**Linux**

Please use the `Blizzard's repository <https://github.com/Blizzard/s2client-proto?tab=readme-ov-file#downloads>`_
to download the Linux version of StarCraft II.

**Windows/MacOS**

You need to first install StarCraft II from `BATTAL.NET <https://battle.net/>`_
or `https://starcraft2.blizzard.com <http://battle.net/sc2/en/legacy-of-the-void/>`_.

.. note::

    You would need to set the SC2PATH environment variable with the correct location of the game.
    By default, the game is expected to be in ~/StarCraftII/ directory.
    This can be changed by setting the environment variable SC2PATH.

For more information about SMAC environment, you can visit its homepage
`https://github.com/oxwhirl/smac.git <https://github.com/oxwhirl/smac.git>`_.

Citation
''''''''''''''''

The BibTex format of SMAC environment is listed as follows. Please cite the SMAC paper if you use it in your research.

::

    @article{samvelyan19smac,
      title = {{The} {StarCraft} {Multi}-{Agent} {Challenge}},
      author = {Mikayel Samvelyan and Tabish Rashid and Christian Schroeder de Witt and Gregory Farquhar and Nantas Nardelli and Tim G. J. Rudner and Chia-Man Hung and Philiph H. S. Torr and Jakob Foerster and Shimon Whiteson},
      journal = {CoRR},
      volume = {abs/1902.04043},
      year = {2019},
    }

APIs
''''''''''''''''

.. automodule:: xuance.environment.multi_agent_env.starcraft2
    :members:
    :undoc-members:
    :show-inheritance:


