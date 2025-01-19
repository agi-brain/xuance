Installation
===========================

The library can be run at Linux, Windows, MacOS, and EulerOS, etc. It is easy to install XuanCe.

Before installing **XuanCe**, it is recommended to set up a Python environment using Anaconda_.

Once ready, open a terminal and follow these steps to install **XuanCe**.
You can choose between two installation methods: from PyPI or GitHub repository.

.. note::

    XuanCe can be installed on MacOS and be compatible with both Intel and Apple's M CPUs.

Install XuanCe
---------------------------------------------

**Step 1**: Create and activate a new conda environment (python>=3.7 is suggested).

.. code-block:: bash

    conda create -n xuance_env python=3.8 && conda activate xuance_env

**Step 2**: Install the ``mpi4py`` dependency.

.. code-block:: bash
    
    conda install mpi4py

**Step 3**: Install ``xuance``.

.. tabs::

    .. tab:: No DL toolbox

        .. code-block:: bash

            pip install xuance

    .. tab:: |_3| |torch| |_3|

        .. code-block:: bash

            pip install xuance[torch]

    .. tab:: |_3| |tensorflow| |_3|

        .. code-block:: bash

            pip install xuance[tensorflow]

    .. tab:: |_3| |mindspore| |_3|

        .. code-block:: bash

            pip install xuance[mindspore]

    .. tab:: All DL toolbox

        .. code-block:: bash

            pip install xuance[all]

Alternatively, you can also install ``xuance`` from its GitHub repository.

.. tabs::

    .. tab:: No DL toolbox

        .. code-block:: bash

            git clone https://github.com/agi-brain/xuance.git
            cd xuance
            pip install -e .

    .. tab:: |_4| |torch| |_4|

        .. code-block:: bash

            git clone https://github.com/agi-brain/xuance.git
            cd xuance
            pip install -e .[torch]

    .. tab:: |tensorflow|

        .. code-block:: bash

            git clone https://github.com/agi-brain/xuance.git
            cd xuance
            pip install -e .[tensorflow]

    .. tab:: |mindspore|

        .. code-block:: bash

            git clone https://github.com/agi-brain/xuance.git
            cd xuance
            pip install -e .[mindspore]

    .. tab:: All DL toolbox

        .. code-block:: bash

            git clone https://github.com/agi-brain/xuance.git
            cd xuance
            pip install -e .[all]

.. attention::

    Some extra packages should be installed manually for further usage.
    See `Install external dependencies <#id1>`_

.. error::

    During the installation of XuanCe, you might encount the following error:

    .. code-block:: bash

        Error: Failed to building wheel for mpi4py
        Failed to build mpi4py
        ERROR: Could not build wheels for mpi4py, which is required to install pyproject.toml-based projects

    **Solution 1**: You can solve that issue by install mpi4py manually via

    .. code-block:: bash

        conda install mpi4py

    **Solution 2**: If that doesn't work, you can type and install ``gcc_linux-64`` via:

    .. code-block:: bash

        conda install gcc_linux-64

    And then, retype the installation command for mpi4py via pip:

    .. code-block:: bash

        pip install mpi4py

.. tip::

    If your IP address is in Chinese mainland, you can install it with a mirror image to speed up the installation,
    for example, you can choose one of the following commands to finish installation.

    .. code-block:: bash

        pip install xuance -i https://pypi.tuna.tsinghua.edu.cn/simple
        pip install xuance -i https://pypi.mirrors.ustc.edu.cn/simple
        pip install xuance -i http://mirrors.aliyun.com/pypi/simple/
        pip install xuance -i http://pypi.douban.com/simple/

.. _Anaconda: https://www.anaconda.com/download
.. _PyTorch: https://pytorch.org/get-started/locally/
.. _TensorFlow2: https://www.tensorflow.org/install
.. _MindSpore: https://www.mindspore.cn/install/en

Test the installation
--------------------------------------------------------------------

After installing XuanCe, you can enter the Python runtime environment by typing "python" in the terminal.
Then, test the installation of xuance by typing:

.. code-block:: python

    import xuance


.. error::

    If you are using Windows OS to import xuance, you might get an error likes this:

    .. code-block:: bash

        ...
        from mpi4py import MPI
        ImportError: DLL load failed: The specified module could not be found.

    You can address the issue by the following steps:

    **Step 1**: Download Microsoft MPI v10.0 from `Official Microsoft Download Center <https://www.microsoft.com/en-us/download/details.aspx?id=57467>`_.

    **Step 2**: Remember to choose both "msmpisetup.exe" and "msmpisdk.msi" options, then click "Download" button and install the ".exe" file.

    **Step 3**: Reinstall mpi4py:

    .. code-block:: bash

        pip uninstall mpi4py
        pip install mpi4py


If no errors or warnings appear, XuanCe has been successfully installed.
You can move on to the next step and begin using it. (`Move to next page <basic_usage.html>`_)

.. raw:: html

    <br><hr>

Install external dependencies
-------------------------------

Some dependencies are not included in XuanCe’s installation process.
You can install the external dependencies listed below as needed.

Box2D
^^^^^^^^

`Box2D environment <../api/environments/single_agent_env/gym.html#box2d>`_ is built using `box2d <https://box2d.org/>`_ for physics control.
It contains three different tasks: Bipedal Walker, Car Racing, Lunar Lander.
If you want to try these tasks, you can install it via commands below.

.. tabs::

    .. tab:: From PyPI

        .. code-block:: bash

            pip install swig==4.3.0
            pip install gymnasium[box2d]==0.28.1

    .. tab:: From XuanCe

        .. code-block:: bash

            pip install xuance[box2d]

MuJoCo
^^^^^^^^

`MuJoCo environment <../api/environments/single_agent_env/gym.html#mujoco>`_ is a physics engine for facilitating research and development in robotics, biomechanics, graphics and animation,
and other areas where fast and accurate simulation is needed.

**Step 1: Install MuJoCo**

* Download the MuJoCo version 2.1 binaries for `Linux <https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz>`_ or `OSX <https://mujoco.org/download/mujoco210-macos-x86_64.tar.gz>`_.
* Extract the downloaded ``mujoco210`` directory into ``~/.mujoco/mujoco210``.

**Step 2: Install mujoco-py**

.. tabs::

    .. tab:: From PyPI

        .. code-block:: bash

            pip install gymnasium[mujoco]

    .. tab:: From XuanCe

        .. code-block:: bash

            pip install xuance[mujoco]

Atari
^^^^^^^^

`Atari environment <../api/environments/single_agent_env/gym.html#atari>`_ is simulated via the
`Arcade Learning Environment (ALE) <https://www.jair.org/index.php/jair/article/view/10819>`_,
which contains 62 different tasks.

.. tabs::

    .. tab:: From PyPI

        .. code-block:: bash

            pip install gymnasium[accept-rom-license]==0.28.1
            pip install gymnasium[atari]==0.28.1
            pip install ale-py==0.8.1

    .. tab:: From XuanCe

        .. code-block:: bash

            pip install xuance[atari]

MiniGrid
^^^^^^^^^

`MiniGrid environment <../api/environments/single_agent_env/minigrid.html>`_ is a lightweight, grid-based environment designed for research in DRL.
It is highly customizable, supporting a variety of tasks and challenges for training agents
with partial observability, sparse rewards, and symbolic inputs.

.. tabs::

    .. tab:: From PyPI

        .. code-block::

            pip install minigrid

    .. tab:: From GitHub Repository

        .. code-block::

            git clone https://github.com/Farama-Foundation/Minigrid.git
            cd Minigrid
            pip install -e .

    .. tab:: From XuanCe

        .. code-block::

            pip install xuance[minigrid]


MetaDrive
^^^^^^^^^^^

`MetaDrive <../api/environments/single_agent_env/metadrive.html>`_ is an autonomous driving simulator that supports generating infinite scenes with various road maps and traffic settings for research of generalizable RL.

.. tabs::

    .. tab:: From PyPI

        .. code-block::

            pip install metadrive

    .. tab:: From GitHub Repository

        .. code-block::

            git clone https://github.com/metadriverse/metadrive.git
            cd metadrive
            pip install -e .

    .. tab:: From XuanCe

        .. code-block::

            pip install xuance[metadrive]


StarCraft2
^^^^^^^^^^^^

The `StarCraft multi-agent challenge (SMAC) <../api/environments/multi_agent_env/smac.html>`_ is `WhiRL's <https://whirl.cs.ox.ac.uk/>`_ environment for research of cooperative MARL algorithms.
SMAC uses StarCraft II, a real-time strategy game developed by Blizzard Entertainment, as its underlying environment.

**Step 1: Install the smac python package**

You can install the SMAC package directly from the GitHub:

.. tabs::

    .. tab:: Method 1

        .. code-block:: bash

            pip install git+https://github.com/oxwhirl/smac.git

    .. tab:: Method 2

        .. code-block:: bash

            git clone https://github.com/oxwhirl/smac.git
            cd smac/
            pip install -e .


**Step 2: Install StarCraft II**

.. tabs::

    .. tab:: Linux

        Please use the `Blizzard's repository <https://github.com/Blizzard/s2client-proto?tab=readme-ov-file#downloads>`_
        to download the Linux version of StarCraft II.

    .. tab:: Windows/MacOS

        You need to first install StarCraft II from `BATTAL.NET <https://battle.net/>`_
        or `https://starcraft2.blizzard.com <http://battle.net/sc2/en/legacy-of-the-void/>`_.

.. note::

    You would need to set the SC2PATH environment variable with the correct location of the game.
    By default, the game is expected to be in ~/StarCraftII/ directory.
    This can be changed by setting the environment variable SC2PATH.

**Step 3: SMAC Maps**

Once you have installed ``smac`` and StarCraft II, you need to download the
`SMAC Maps <https://github.com/oxwhirl/smac/releases/download/v0.1-beta1/SMAC_Maps.zip>`_,
and extract it to the ``$SC2PATH/Maps$`` directory.
If you installed ``smac`` via git, simply copy the ``SMAC_Maps`` directory
from ``smac/env/starcraft2/maps/`` into ``$SC2PATH/Maps`` directory.

Google Research Football
^^^^^^^^^^^^^^^^^^^^^^^^^

`Google Research Football Environment (GRF) <../api/environments/multi_agent_env/football.html>`_ is an MARL environment developed by the Google Brain team.
It is specifically designed for RL research, particularly for MARL scenarios.

**Step 1: Install required packages**

.. tabs::

    .. tab:: Linux

        .. code-block:: bash

            sudo apt-get install git cmake build-essential libgl1-mesa-dev libsdl2-dev \
            libsdl2-image-dev libsdl2-ttf-dev libsdl2-gfx-dev libboost-all-dev \
            libdirectfb-dev libst-dev mesa-utils xvfb x11vnc python3-pip

            python3 -m pip install --upgrade pip setuptools psutil wheel

    .. tab:: MacOS

        .. code-block:: bash

            brew install git python3 cmake sdl2 sdl2_image sdl2_ttf sdl2_gfx boost boost-python3

            python3 -m pip install --upgrade pip setuptools psutil wheel

    .. tab:: Windows

        .. code-block::

            python -m pip install --upgrade pip setuptools psutil wheel

**Step 2: Install gfootball**

.. tabs::

    .. tab:: From PyPI

        .. code-block:: bash

            python3 -m pip install gfootball

    .. tab:: From GitHub repository

        .. code-block:: bash

            git clone https://github.com/google-research/football.git
            cd football
            python3 -m pip install .


.. attention::

    All python packages including ``gfootball`` environment should be installed in a same conda environment.
    See `https://xuance.readthedocs.io/en/latest/documents/usage/installation.html#install-via-pypi <https://xuance.readthedocs.io/en/latest/documents/usage/installation.html#install-via-pypi>`_.


Robotic Warehouse
^^^^^^^^^^^^^^^^^^^

`Robotic Warehouse <../api/environments/multi_agent_env/robotic_warehouse.html>`_ is an MARL environment often used to simulate warehouse automation scenarios.
It serves as a testbed for studying cooperative, competitive, and mixed interaction among multiple agents, such as robots.
The environment is designed to model tasks commonly found in real-world warehouses,
such as navigation, item retrieval, obstacle avoidance, and task allocation.

.. tabs::

    .. tab:: From PyPI

        .. code-block::

            pip install rware

    .. tab:: From GitHub Repository

        .. code-block::

            git clone git@github.com:uoe-agents/robotic-warehouse.git
            cd robotic-warehouse
            pip install -e .

    .. tab:: From XuanCe

        .. code-block::

            pip install xuance[rware]


gym-pybullet-drones
^^^^^^^^^^^^^^^^^^^^

.. tip::

    Before preparing the software packages for this simulator, it is recommended to create a new conda environment with **Python 3.10**.

Open terminal and type the following commands, then a new conda environment for xuance with drones could be built:

.. code-block:: bash

    conda create -n xuance_drones python=3.10
    conda activate xuance_drones
    pip install xuance  # refer to the installation of XuanCe.

    git clone https://github.com/utiasDSL/gym-pybullet-drones.git
    cd gym-pybullet-drones/
    pip install --upgrade pip
    pip install -e .  # if needed, `sudo apt install build-essential` to install `gcc` and build `pybullet`

During the installation of gym-pybullet-drones, you might encounter the errors like:

.. error::

    | gym-pybullet-drones 2.0.0 requires numpy<2.0,>1.24, but you have numpy 1.22.4 which is incompatible.
    | gym-pybullet-drones 2.0.0 requires scipy<2.0,>1.10, but you have scipy 1.7.3 which is incompatible.

**Solution**: Upgrade the above incompatible packages.

.. code-block:: bash

    pip install numpy==1.24.0
    pip install scipy==1.12.0

DCG algorithm dependency (torch-scatter)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

    pip install torch-scatter
