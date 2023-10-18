Learner
=======================

To create new learner, you should build a class inherit from ``xuanpolicy.torch.learners.learner.Learner`` , ``xuanpolicy.tensorflow.learners.learner.Learner``, or ``xuanpolicy.mindspore.learners.learner.Learner``.

**PyTorch:**

.. py:class:: 
    xuanpolicy.torch.learners.learner.Learner(policy, optimizer, scheduler=None, device=None, model_dir="./")

    The basic class of the learner.

    :param policy: The policy that provides actions and values.
    :type policy: nn.Module
    :param optimizer: The optimizer that update the paramters of the model.
    :type optimizer: torch.optim.Optimizer
    :param scheduler: The tool for learning rate decay.
    :type scheduler: torch.optim.lr_scheduler
    :param device: Choose CPU or GPU to train the model.
    :type device: str, int, torch.device
    :param model_dir: The directory of model file, default is "./".
    :type model_dir: str

.. py:function:: xuanpolicy.torch.learners.learner.Learner.save_model(model_path)
    
    Save the model.

    :param model_path: The model's path.
    :type model_path: str

.. py:function:: xuanpolicy.torch.learners.learner.Learner.load_model(path, seed=1)

    Load a model by specifying the ``path`` and ``seed`` .

    :param path: The model's path where to load.
    :type path: str
    :param seed: Select the seed that model was trained with if it exits.
    :type seed: int

.. py:function:: xuanpolicy.torch.learners.learner.Learner.update(*args)
   
    Update the policies with self.optimizer.

.. raw:: html

   <br><hr>

**TensorFlow:**


.. raw:: html

   <br><hr>

**MindSpore:**


.. raw:: html

   <br><hr>

源码
-----------------

.. tabs::

    .. group-tab:: PyTorch

        .. code-block:: python3
            
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

        .. code-block:: python3

    .. group-tab:: MindSpore

        .. code-block:: python3


