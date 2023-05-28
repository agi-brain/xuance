Installation
====

The library can be run at Linux, Windows, MacOS, and EulerOS, etc.

Before installing **XuanPolicy**, you should install [Anaconda](https://www.anaconda.com/download) to prepare a python environment.

After that, open a terminal and install **XuanPolicy** by the following steps.

**Step 1**: Create a new conda environment (python>=3.7 is suggested):

```commandline
conda create -n xpolicy python=3.7
```

**Step 2**: Activate conda environment:

```commandline
conda activate xpolicy
```

**Step 3**: Install the library:

```commandline
pip install xuanpolicy
```

This command does not include the dependencies of deep learning toolboxes. To install the **XuanPolicy** with 
deep learning tools, you can type `pip install xuanpolicy[torch]` for [PyTorch](https://pytorch.org/get-started/locally/),
`pip install xuanpolicy[tensorflow]` for [TensorFlow2](https://www.tensorflow.org/install),
`pip install xuanpolicy[mindspore]` for [MindSpore](https://www.mindspore.cn/install/en),
and `pip install xuanpolicy[all]` for all dependencies.

Note: Some extra packages should be installed manually for further usage. 