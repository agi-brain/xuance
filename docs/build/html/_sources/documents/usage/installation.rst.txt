Installation
===========================

The library can be run at Linux, Windows, MacOS, and EulerOS, etc. It is easy to be installed.

Before installing **XuanPolicy**, you should install Anaconda_ to prepare a python environment.

After that, open a terminal and install **XuanPolicy** by the following steps.

**Step 1**: Create a new conda environment (python>=3.7 is suggested):
::

    conda create -n xpolicy python=3.7

**Step 2**: Activate conda environment:
::
    
    conda activate xpolicy

**Step 3**: Install the library:
::
    
    pip install xuanpolicy

This command does not include the dependencies of deep learning toolboxes. 

Install the **XuanPolicy** with PyTorch_:
::
    
    pip install xuanpolicy[torch]

Install the **XuanPolicy** with TensorFlow2_:
::
    
    pip install xuanpolicy[tensorflow]

Install the **XuanPolicy** with MindSpore_:
::
    
    pip install xuanpolicy[mindspore]

Install the **XuanPolicy** with all dependencies:
::
    pip install xuanpolicy[all]

Note: Some extra packages should be installed manually for further usage. 

.. _Anaconda: https://www.anaconda.com/download
.. _PyTorch: https://pytorch.org/get-started/locally/
.. _TensorFlow2: https://www.tensorflow.org/install
.. _MindSpore: https://www.mindspore.cn/install/en