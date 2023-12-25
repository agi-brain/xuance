Basic Learner
=======================

To create new learner, you should build a class inherit from ``xuance.torch.learners.learner.Learner`` , ``xuance.tensorflow.learners.learner.Learner``, or ``xuance.mindspore.learners.learner.Learner``.

PyTorch
------------------------------------------

.. py:class:: 
    xuance.torch.learners.learner.Learner(policy, optimizer, scheduler=None, device=None, model_dir="./")

    The basic class of the learner.

    :param policy: The policy that provides actions and values.
    :type policy: nn.Module
    :param optimizer: The optimizer that update the parameters of the model.
    :type optimizer: torch.optim.Optimizer
    :param scheduler: The tool for learning rate decay.
    :type scheduler: torch.optim.lr_scheduler
    :param device: The choice of the calculating device.
    :type device: str, int, torch.device
    :param model_dir: The directory of model file, default is "./".
    :type model_dir: str

.. py:function:: xuance.torch.learners.learner.Learner.save_model(model_path)
    
    Save the model.

    :param model_path: The model's path.
    :type model_path: str

.. py:function:: xuance.torch.learners.learner.Learner.load_model(path, seed=1)

    Load a model by specifying the ``path`` and ``seed`` .

    :param path: The model's path where to load.
    :type path: str
    :param seed: Select the seed that model was trained with if it exits.
    :type seed: int

.. py:function:: xuance.torch.learners.learner.Learner.update(*args)
   
    Update the policies with self.optimizer.

.. raw:: html

   <br><hr>

TensorFlow
------------------------------------------

.. py:class:: 
    xuance.tensorflow.learners.learner.Learner(policy, optimizer, device=None, model_dir="./")

    The basic class of the learner.

    :param policy: The policy that provides actions and values.
    :type policy: tk.Model
    :param optimizer: The optimizer that update the parameters of the model.
    :type optimizer: tk.optimizers.Optimizer
    :param device: Choose CPU or GPU to train the model.
    :type device: str, int, torch.device
    :param model_dir: The directory of model file, default is "./".
    :type model_dir: str

.. py:function:: xuance.tensorflow.learners.learner.Learner.save_model(model_path)
    
    Save the model.

    :param model_path: The model's path.
    :type model_path: str

.. py:function:: xuance.tensorflow.learners.learner.Learner.load_model(path, seed=1)

    Load a model by specifying the ``path`` and ``seed`` .

    :param path: The model's path where to load.
    :type path: str
    :param seed: Select the seed that model was trained with if it exits.
    :type seed: int

.. py:function:: xuance.tensorflow.learners.learner.Learner.update(*args)
   
    Update the policies with self.optimizer.

.. raw:: html

   <br><hr>

MindSpore
------------------------------------------

.. py:class:: 
    xuance.mindspore.learners.learner.Learner(policy, optimizer, scheduler=None, model_dir="./")

    The basic class of the learner.

    :param policy: The policy that provides actions and values.
    :type policy: nn.Cell
    :param optimizer: The optimizer that update the parameters of the model.
    :type optimizer: nn.Optimizer
    :param scheduler: The tool for learning rate decay.
    :type scheduler: nn.Cell
    :param model_dir: The directory of model file, default is "./".
    :type model_dir: str

.. py:function:: xuance.mindspore.learners.learner.Learner.save_model(model_path)
    
    Save the model.

    :param model_path: The model's path.
    :type model_path: str

.. py:function:: xuance.mindspore.learners.learner.Learner.load_model(path, seed=1)

    Load a model by specifying the ``path`` and ``seed`` .

    :param path: The model's path where to load.
    :type path: str
    :param seed: Select the seed that model was trained with if it exits.
    :type seed: int

.. py:function:: xuance.mindspore.learners.learner.Learner.update(*args)
   
    Update the policies with self.optimizer.


.. raw:: html

   <br><hr>

Source Code
-----------------

.. tabs::

    .. group-tab:: PyTorch

        .. code-block:: python
            
            import torch
            import time
            import torch.nn.functional as F
            from abc import ABC, abstractmethod
            from typing import Optional, Sequence, Union
            from argparse import Namespace
            import os

            class Learner(ABC):
                def __init__(self,
                            policy: torch.nn.Module,
                            optimizer: Union[torch.optim.Optimizer, Sequence[torch.optim.Optimizer]],
                            scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
                            device: Optional[Union[int, str, torch.device]] = None,
                            model_dir: str = "./"):
                    self.policy = policy
                    self.optimizer = optimizer
                    self.scheduler = scheduler
                    self.device = device
                    self.model_dir = model_dir
                    self.iterations = 0

                def save_model(self, model_path):
                    torch.save(self.policy.state_dict(), model_path)

                def load_model(self, path, seed=1):
                    file_names = os.listdir(path)
                    for f in file_names:
                        '''Change directory to the specified seed (if exists)'''
                        if f"seed_{seed}" in f:
                            path = os.path.join(path, f)
                            break
                    model_names = os.listdir(path)
                    if os.path.exists(path + "/obs_rms.npy"):
                        model_names.remove("obs_rms.npy")
                    model_names.sort()
                    model_path = os.path.join(path, model_names[-1])
                    self.policy.load_state_dict(torch.load(model_path, map_location={
                        "cuda:0": self.device,
                        "cuda:1": self.device,
                        "cuda:2": self.device,
                        "cuda:3": self.device,
                        "cuda:4": self.device,
                        "cuda:5": self.device,
                        "cuda:6": self.device,
                        "cuda:7": self.device
                    }))

                @abstractmethod
                def update(self, *args):
                    raise NotImplementedError


            class LearnerMAS(ABC):
                def __init__(self,
                            config: Namespace,
                            policy: torch.nn.Module,
                            optimizer: Union[torch.optim.Optimizer, Sequence[torch.optim.Optimizer]],
                            scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
                            device: Optional[Union[int, str, torch.device]] = None,
                            model_dir: str = "./"):
                    self.value_normalizer = None
                    self.args = config
                    self.n_agents = config.n_agents
                    self.dim_obs = self.args.dim_obs
                    self.dim_act = self.args.dim_act
                    self.dim_id = self.n_agents
                    self.device = torch.device("cuda" if (torch.cuda.is_available() and self.args.device == "gpu") else "cpu")
                    if self.device.type == "cuda":
                        torch.cuda.set_device(config.gpu_id)
                        print("Use cuda, gpu ID: ", config.gpu_id)

                    self.policy = policy
                    self.optimizer = optimizer
                    self.scheduler = scheduler
                    self.device = device
                    self.model_dir = model_dir
                    self.running_steps = config.running_steps
                    self.iterations = 0

                def onehot_action(self, actions_int, num_actions):
                    return F.one_hot(actions_int.long(), num_classes=num_actions)

                def save_model(self, model_path):
                    torch.save(self.policy.state_dict(), model_path)

                def load_model(self, path, seed=1):
                    file_names = os.listdir(path)
                    for f in file_names:
                        '''Change directory to the specified seed (if exists)'''
                        if f"seed_{seed}" in f:
                            path = os.path.join(path, f)
                            break
                    model_names = os.listdir(path)
                    if os.path.exists(path + "/obs_rms.npy"):
                        model_names.remove("obs_rms.npy")
                    model_names.sort()
                    model_path = os.path.join(path, model_names[-1])
                    self.policy.load_state_dict(torch.load(model_path, map_location={
                        "cuda:0": self.device,
                        "cuda:1": self.device,
                        "cuda:2": self.device,
                        "cuda:3": self.device,
                        "cuda:4": self.device,
                        "cuda:5": self.device,
                        "cuda:6": self.device,
                        "cuda:7": self.device
                    }))

                @abstractmethod
                def update(self, *args):
                    raise NotImplementedError

                def update_recurrent(self, *args):
                    pass

                def act(self, *args, **kwargs):
                    pass

                def get_hidden_states(self, *args):
                    pass

                def lr_decay(self, *args):
                    pass

   
    .. group-tab:: TensorFlow

        .. code-block:: python

            from xuance.tensorflow.learners import *
            from argparse import Namespace


            class Learner(ABC):
                def __init__(self,
                            policy: tk.Model,
                            optimizer: Union[tk.optimizers.Optimizer, Sequence[tk.optimizers.Optimizer]],
                            device: str = "cpu:0",
                            model_dir: str = "./"):
                    self.policy = policy
                    self.optimizer = optimizer
                    self.device = device
                    self.model_dir = model_dir
                    self.iterations = 0

                def save_model(self, model_path):
                    self.policy.save_weights(model_path)

                def load_model(self, path, seed=1):
                    try: file_names = os.listdir(path)
                    except: raise "Failed to load model! Please train and save the model first."

                    for f in file_names:
                        '''Change directory to the specified seed (if exists)'''
                        if f"seed_{seed}" in f:
                            path = os.path.join(path, f)
                            break
                    latest = tf.train.latest_checkpoint(path)
                    try:
                        self.policy.load_weights(latest)
                    except:
                        raise "Failed to load model! Please train and save the model first."

                @abstractmethod
                def update(self, *args):
                    raise NotImplementedError


            class LearnerMAS(ABC):
                def __init__(self,
                            config: Namespace,
                            policy: tk.Model,
                            optimizer: Union[tk.optimizers.Optimizer, Sequence[tk.optimizers.Optimizer]],
                            device: str = "cpu:0",
                            model_dir: str = "./"):
                    self.args = config
                    self.handle = config.handle
                    self.n_agents = config.n_agents
                    self.agent_keys = config.agent_keys
                    self.agent_index = config.agent_ids
                    self.dim_obs = self.args.dim_obs
                    self.dim_act = self.args.dim_act
                    self.dim_id = self.n_agents
                    self.device = device

                    self.policy = policy
                    self.optimizer = optimizer
                    self.device = device
                    self.model_dir = model_dir
                    self.running_steps = config.running_steps
                    self.iterations = 0

                def onehot_action(self, actions_int, num_actions):
                    return tf.one_hot(indices=actions_int, depth=num_actions, axis=-1, dtype=tf.float32)

                def save_model(self, model_path):
                    self.policy.save_weights(model_path)

                def load_model(self, path, seed=1):
                    try: file_names = os.listdir(path)
                    except: raise "Failed to load model! Please train and save the model first."
                    model_path = ''

                    for f in file_names:
                        '''Change directory to the specified seed (if exists)'''
                        if f"seed_{seed}" in f:
                            model_path = os.path.join(path, f)
                            if os.listdir(model_path).__len__() == 0:
                                continue
                            else:
                                break
                    if model_path == '':
                        raise RuntimeError("Failed to load model! Please train and save the model first.")
                    latest = tf.train.latest_checkpoint(model_path)
                    try:
                        self.policy.load_weights(latest)
                    except:
                        raise RuntimeError("Failed to load model! Please train and save the model first.")

                @abstractmethod
                def update(self, *args):
                    raise NotImplementedError

                def update_recurrent(self, *args):
                    pass

                def act(self, *args, **kwargs):
                    pass

                def get_hidden_states(self, *args):
                    pass

                def lr_decay(self, *args):
                    pass


    .. group-tab:: MindSpore

        .. code-block:: python

            import mindspore.nn as nn
            import mindspore as ms
            from mindspore.ops import OneHot, Eye
            import time
            from abc import ABC, abstractmethod
            from typing import Optional, Sequence, Union
            from torch.utils.tensorboard import SummaryWriter
            from argparse import Namespace
            import os


            class Learner(ABC):
                def __init__(self,
                            policy: nn.Cell,
                            optimizer: nn.Optimizer,
                            scheduler: Optional[nn.exponential_decay_lr] = None,
                            model_dir: str = "./"):
                    self.policy = policy
                    self.optimizer = optimizer
                    self.scheduler = scheduler
                    self.model_dir = model_dir
                    self.iterations = 0

                def save_model(self, model_path, file_name):
                    if not os.path.exists(model_path):
                        try:
                            os.mkdir(model_path)
                        except:
                            os.makedirs(model_path)
                    ckpt_file_name = os.path.join(model_path, file_name)
                    ms.save_checkpoint(self.policy, ckpt_file_name)

                def load_model(self, path, seed=1):
                    file_names = os.listdir(path)
                    for f in file_names:
                        '''Change directory to the specified seed (if exists)'''
                        if f"seed_{seed}" in f:
                            path = os.path.join(path, f)
                            break
                    model_names = os.listdir(path)
                    if os.path.exists(path + "/obs_rms.npy"):
                        model_names.remove("obs_rms.npy")
                    model_names.sort()
                    model_path = os.path.join(path, model_names[-1])
                    ms.load_param_into_net(self.policy, ms.load_checkpoint(model_path))

                @abstractmethod
                def update(self, *args):
                    raise NotImplementedError


            class LearnerMAS(ABC):
                def __init__(self,
                            config: Namespace,
                            policy: nn.Cell,
                            optimizer: Union[nn.Optimizer, Sequence[nn.Optimizer]],
                            scheduler: Optional[nn.exponential_decay_lr] = None,
                            model_dir: str = "./"):
                    self.args = config
                    self.handle = config.handle
                    self.n_agents = config.n_agents
                    self.agent_keys = config.agent_keys
                    self.agent_index = config.agent_ids
                    self.dim_obs = self.args.dim_obs
                    self.dim_act = self.args.dim_act
                    self.dim_id = self.n_agents

                    self.policy = policy
                    self.optimizer = optimizer
                    self.scheduler = scheduler
                    self.model_dir = model_dir
                    self.running_steps = config.running_steps
                    self.iterations = 0
                    self._one_hot = OneHot()
                    self.eye = Eye()
                    self.expand_dims = ms.ops.ExpandDims()

                def onehot_action(self, actions_int, num_actions):
                    return self._one_hot(actions_int.astype(ms.int32), num_actions,
                                        ms.Tensor(1.0, ms.float32), ms.Tensor(0.0, ms.float32))

                def save_model(self, model_path, file_name):
                    if not os.path.exists(model_path):
                        try:
                            os.mkdir(model_path)
                        except:
                            os.makedirs(model_path)
                    ckpt_file_name = os.path.join(model_path, file_name)
                    ms.save_checkpoint(self.policy, ckpt_file_name)

                def load_model(self, path, seed=1):
                    file_names = os.listdir(path)
                    for f in file_names:
                        '''Change directory to the specified seed (if exists)'''
                        if f"seed_{seed}" in f:
                            path = os.path.join(path, f)
                            break
                    model_names = os.listdir(path)
                    if os.path.exists(path + "/obs_rms.npy"):
                        model_names.remove("obs_rms.npy")
                    model_names.sort()
                    model_path = os.path.join(path, model_names[-1])
                    ms.load_param_into_net(self.policy, ms.load_checkpoint(model_path))

                @abstractmethod
                def update(self, *args):
                    raise NotImplementedError



