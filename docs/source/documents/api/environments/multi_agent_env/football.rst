Google Research Football
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. image:: ../../../../_static/figures/football/gfootball.png

Overview
'''''''''''

The Google Research Football Environment (GRF) is an MARL environment developed by the Google Brain team.
It is specifically designed for RL research, particularly for MARL scenarios.
Here are some key details about it:

- Environment: GRF simulates a highly challenging football (soccer) game environment.
- Purpose: It is used to test and develop reinforcement learning algorithms in scenarios that involve strategy, teamwork, and complex decision-making.
- Focus: It emphasizes multi-agent RL, long-term planning, and continuous control.

Key Features
................

- Realistic Physics: The environment is based on the gameplay of football, including realistic physics and dynamic player interactions.
- Multi-Agent Scenarios: Supports both single-agent and multi-agent settings where agents can collaborate or compete.
- Customizable Matches: Users can customize team configurations, game scenarios, and even create custom reward structures.
- Diverse Observations: Provides rich observational data, such as: player positions, ball position, velocities, game state (e.g., goals scored, possession).
- Action Space:
    - Discrete: Actions like shooting, passing, moving, etc.
    - Continuous: Detailed control over players’ movements and actions.
- Challenging Environment: It includes complex strategies, requiring hierarchical planning and effective coordination among agents.

Applications
.................

- Policy Learning: Test RL algorithms in competitive and cooperative multi-agent scenarios.
- Teamwork and Collaboration: Study cooperative behaviors and communication strategies among agents.
- Generalization: Investigate how well RL policies generalize across different scenarios or team compositions.

Installation and Usage
'''''''''''''''''''''''''

Step 1: Install required packages
........................................

* **Linux:**

.. code-block:: bash

    sudo apt-get install git cmake build-essential libgl1-mesa-dev libsdl2-dev \
    libsdl2-image-dev libsdl2-ttf-dev libsdl2-gfx-dev libboost-all-dev \
    libdirectfb-dev libst-dev mesa-utils xvfb x11vnc python3-pip

    python3 -m pip install --upgrade pip setuptools psutil wheel

* **MacOS:**

.. code-block:: bash

    brew install git python3 cmake sdl2 sdl2_image sdl2_ttf sdl2_gfx boost boost-python3

    python3 -m pip install --upgrade pip setuptools psutil wheel

* **Windows:**

.. code-block::

    python -m pip install --upgrade pip setuptools psutil wheel

Step 2: Install gfootball
................................

Method 1: Install from PyPi.

.. code-block:: bash

    python3 -m pip install gfootball

Method 2: Installing from GitHub repository.

.. code-block:: bash

    git clone https://github.com/google-research/football.git
    cd football
    python3 -m pip install .

.. attention::

    All python packages including ``gfootball`` environment should be installed in a same conda environment.
    See `https://xuance.readthedocs.io/en/latest/documents/usage/installation.html#install-via-pypi <https://xuance.readthedocs.io/en/latest/documents/usage/installation.html#install-via-pypi>`_.

Test installation
...................

.. code-block:: bash

    python3 -m gfootball.play_game --action_set=full

.. raw:: html

    <br><hr>

APIs
'''''''''''''

.. automodule:: xuance.environment.multi_agent_env.football
    :members:
    :undoc-members:
    :show-inheritance:
