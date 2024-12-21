Installation
===========================

The library can be run at Linux, Windows, MacOS, and EulerOS, etc. It is easy to install XuanCe.

Before installing **XuanCe**, it is recommended to set up a Python environment using Anaconda_.

Once ready, open a terminal and follow these steps to install **XuanCe**.
You can choose between two installation methods: from PyPI or GitHub repository.

.. note::

    Note: XuanCe can be installed on MacOS and be compatible with both Intel and Apple's M CPUs.

Install via PyPI
---------------------------------------------

**Step 1**: Create a new conda environment (python>=3.7 is suggested).

.. code-block:: bash

    conda create -n xuance_env python=3.7

**Step 2**: Activate conda environment.

.. code-block:: bash
    
    conda activate xuance_env

**Step 3**: Install the library.

.. tabs::

    .. tab:: No DL toolbox

        .. code-block:: bash

            pip install xuance

    .. tab:: PyTorch

        .. code-block:: bash

            pip install xuance[torch]

    .. tab:: TensorFlow

        .. code-block:: bash

            pip install xuance[tensorflow]

    .. tab:: MindSpore

        .. code-block:: bash

            pip install xuance[mindspore]

    .. tab:: All DL toolbox

        .. code-block:: bash

            pip install xuance[all]

    .. tab:: Atari

        .. code-block:: bash

            pip install xuance[atari]

    .. tab:: Box2D

        .. code-block:: bash

            pip install xuance[box2d]


Install from GitHub repository
---------------------------------------------

Alternatively, you can install XuanCe from its GitHub repository.

.. note::

    Note: The steps 1-2 are the same as above.

**Step 1**: Create a new conda environment (python>=3.7 is suggested).

.. code-block:: bash

    conda create -n xuance_env python=3.7

**Step 2**: Activate conda environment.

.. code-block:: bash

    conda activate xuance_env

**Step 3**: Download the source code of XuanCe from GitHub.

.. code-block:: bash

    git clone https://github.com/agi-brain/xuance.git

**Step 4**: Change directory to the xuance.

.. code-block:: bash

    cd xuance

**Step 5**: Install xuance.

.. tabs::

    .. tab:: No DL toolbox

        .. code-block:: bash

            pip install -e .

    .. tab:: PyTorch

        .. code-block:: bash

            pip install -e .[torch]

    .. tab:: TensorFlow

        .. code-block:: bash

            pip install -e .[tensorflow]

    .. tab:: MindSpore

        .. code-block:: bash

            pip install -e .[mindspore]

    .. tab:: All DL toolbox

        .. code-block:: bash

            pip install -e .[all]

    .. tab:: Atari

        .. code-block:: bash

            pip install -e .[atari]

    .. tab:: Box2D

        .. code-block:: bash

            pip install -e .[box2d]

.. attention::

    Some extra packages should be installed manually for further usage.

.. tip::

    If your IP address is in Chinese mainland, you can install it with a mirror image to speed up the installation,
    for example, you can choose one of the following commands to finish installation.

    .. code-block:: bash

        pip install xuance -i https://pypi.tuna.tsinghua.edu.cn/simple
        pip install xuance -i https://pypi.mirrors.ustc.edu.cn/simple
        pip install xuance -i http://mirrors.aliyun.com/pypi/simple/
        pip install xuance -i http://pypi.douban.com/simple/

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


.. _Anaconda: https://www.anaconda.com/download
.. _PyTorch: https://pytorch.org/get-started/locally/
.. _TensorFlow2: https://www.tensorflow.org/install
.. _MindSpore: https://www.mindspore.cn/install/en

Testing whether the installation was successful
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
You can move on to the next step and begin using it.

