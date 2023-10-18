Installation
===========================

The library can be run at Linux, Windows, MacOS, and EulerOS, etc. It is easy to be installed.

Before installing **XuanPolicy**, you should install Anaconda_ to prepare a python environment.

After that, open a terminal and install **XuanPolicy** by the following steps.
You can choose two ways to install XuanPolicy.

.. raw:: html

   <br><hr>

Install via PyPI
---------------------------------------------

**Step 1**: Create a new conda environment (python>=3.7 is suggested).

.. code-block:: console

    $ conda create -n xpolicy python=3.7

**Step 2**: Activate conda environment.

.. code-block:: console
    
    $ conda activate xpolicy

**Step 3**: Install the library.

.. code-block:: console
    
    $ pip install xuanpolicy

This command does not include the dependencies of deep learning toolboxes. 

You can also install the **XuanPolicy** with PyTorch_, TensorFlow2_, MindSpore_, or all of them.

.. code-block:: console
    
    $ pip install xuanpolicy[torch]

or

.. code-block:: console
    
    $ pip install xuanpolicy[tensorflow]

or

.. code-block:: console
    
    $ pip install xuanpolicy[mindspore]

or

.. code-block:: console

    $ pip install xuanpolicy[all]

Install from GitHub repository
---------------------------------------------

Alternatively, you can install XuanPolicy from its GitHub repository.

.. note::

    Note: The steps 1-2 are the same as above.

**Step 1**: Create a new conda environment (python>=3.7 is suggested).

.. code-block:: console

    $ conda create -n xpolicy python=3.7

**Step 2**: Activate conda environment.

.. code-block:: console

    $ conda activate xpolicy

**Step 3**: Download the source code of XuanPolicy from GitHub.

.. code-block:: console

    $ git clone https://github.com/agi-brain/xuanpolicy.git

**Step 4**: Change directory to the xuanpolicy.

.. code-block:: console

    $ cd xuanpolicy

**Step 5**: Install xuanpolicy.

.. code-block:: console

    $ pip install -e .

.. note::

    Note: Some extra packages should be installed manually for further usage.

.. _Anaconda: https://www.anaconda.com/download
.. _PyTorch: https://pytorch.org/get-started/locally/
.. _TensorFlow2: https://www.tensorflow.org/install
.. _MindSpore: https://www.mindspore.cn/install/en

.. raw:: html

   <br><hr>

Testing whether the installation was successful
--------------------------------------------------------------------

After installing XuanPolicy, you can enter the Python runtime environment by typing "python" in the terminal.
Then, test the installation of xuanpolicy by typing:

.. code-block:: python

    import xuanpolicy

If no error or warning messages are displayed, it indicates that XuanPolicy has been successfully installed.
You can proceed to the next step and start using it.

