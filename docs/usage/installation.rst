Installation
====

The library can be run at Linux, Windows, MacOS, and EulerOS, etc.

Before installing **XuanPolicy**, you should install [Anaconda](https://www.anaconda.com/download) to prepare a python environment.

After that, open a terminal and install **XuanPolicy** by the following steps.

**Step 1**: Create a new conda environment (python>=3.7 is suggested):

| conda create -n xpolicy python=3.7

**Step 2**: Activate conda environment:

| conda activate xpolicy

**Step 3**: Install the library:

| pip install xuanpolicy

This command does not include the dependencies of deep learning toolboxes. To install the **XuanPolicy** with 
deep learning tools, such as for PyTorch_, TensorFlow2_, MindSpore_, you can type

| pip install xuanpolicy[torch]

| pip install xuanpolicy[tensorflow]

| pip install xuanpolicy[mindspore]

For all dependencies, you can install it by typing:
| pip install xuanpolicy[all]

Note: Some extra packages should be installed manually for further usage. 

.. _PyTorch: https://pytorch.org/get-started/locally/
.. _TensorFlow2: https://www.tensorflow.org/install
.. _MindSpore: https://www.mindspore.cn/install/en