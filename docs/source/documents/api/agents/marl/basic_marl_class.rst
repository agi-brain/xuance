MARLAgent
=======================

To create new MARL agents, you should build a class inherit from ``xuance.torch.agents.agents_marl.MARLAgent`` , ``xuance.tensorflow.agents.agents_marl.MARLAgent``, or ``xuance.mindspore.agents.agents_marl.MARLAgent``.

PyTorch
------------------------------------------

.. py:class:: 
   xuance.torch.agents.agents_marl.MARLAgents(config, envs, policy, memory, learner, device, log_dir, model_dir)

   :param config: Provides hyper parameters.
   :type config: Namespace
   :param envs: The vectorized environments.
   :type envs: xuance.environments.vector_envs.vector_env.VecEnv
   :param policy: The policy that provides actions and values.
   :type policy: nn.Module
   :param memory: Experice replay buffer.
   :type memory: xuance.common.memory_tools.Buffer
   :param learner: The learner that updates parameters of policy.
   :type learner: xuance.torch.learner.LearnerMAS
   :param device: Choose CPU or GPU to train the model.
   :type device: str, int, torch.device
   :param log_dir: The directory of log file, default is "./logs/".
   :type log_dir: str
   :param model_dir: The directory of model file, default is "./models/".
   :type model_dir: str

.. py:function:: xuance.torch.agents.agents_marl.MARLAgents.save_model(model_name)
   
   Save the model.

   :param model_name: The model's name to be saved.
   :type model_name: str

.. py:function:: xuance.torch.agents.agents_marl.MARLAgents.load_model(path, seed)

   Load a model by specifying the ``path`` and ``seed`` .

   :param path: The model's path where to load.
   :type path: str
   :param seed: Select the seed that model was trained with if it exits.
   :type seed: int

.. py:function:: xuance.torch.agents.agents_marl.MARLAgents.act(**kwargs)
   
   Get actions for executing according to the joint observations, global states, available actions, etc.
   
   :param kwargs: Inputs informations.
   :type observations: Dict

.. py:function:: xuance.torch.agents.agents_marl.MARLAgents.train(**kwargs)
   
   Train the multi-agent reinforcement learning models.

   :param kwargs: Informations for multi-agent training.
   :type observations: Dict
   :return: **info_train** - Informations of the training.
   :rtype: Dict


.. py:class:: 
   xuance.torch.agents.agents_marl.linear_decay_or_increase(start, end, step_length)

   :param start: Start factor.
   :type start: np.float
   :param end: End factor.
   :type end: np.float
   :param step_length: The number of steps the factor decays or increases.
   :type step_length: int

.. py:function:: xuance.torch.agents.agents_marl.linear_decay_or_increase.update()
   
   Update the factor once.


.. py:class:: 
   xuance.torch.agents.agents_marl.RandomAgents(args, envs, device=None)

   :param args: Provides hyper parameters.
   :type args: Namespace
   :param envs: The vectorized environments.
   :type envs: xuance.environments.vector_envs.vector_env.VecEnv
   :param device: Choose CPU or GPU to train the model.
   :type device: str, int, torch.device

.. py:function:: 
   xuance.torch.agents.agents_marl.RandomAgents.act()
   
   Provide random actions for RandomAgents.

   :return: **random_actions** - Output random actions.
   :rtype: np.ndarray


.. raw:: html

   <br><hr>

TensorFlow
------------------------------------------

.. py:class:: 
   xuance.tensorflow.agents.agents_marl.MARLAgents(config, envs, policy, memory, learner, device, log_dir, model_dir)

   :param config: Provides hyper parameters.
   :type config: Namespace
   :param envs: The vectorized environments.
   :type envs: xuance.environments.vector_envs.vector_env.VecEnv
   :param policy: The policy that provides actions and values.
   :type policy: tk.Model
   :param memory: Experice replay buffer.
   :type memory: xuance.common.memory_tools.Buffer
   :param learner: The learner that updates parameters of policy.
   :type learner: xuance.tensorflow.learner.LearnerMAS
   :param device: Choose CPU or GPU to train the model.
   :type device: str
   :param log_dir: The directory of log file, default is "./logs/".
   :type log_dir: str
   :param model_dir: The directory of model file, default is "./models/".
   :type model_dir: str

.. py:function:: xuance.tensorflow.agents.agents_marl.MARLAgents.save_model(model_name)
   
   Save the model.

   :param model_name: The model's name to be saved.
   :type model_name: str

.. py:function:: xuance.tensorflow.agents.agents_marl.MARLAgents.load_model(path, seed)

   Load a model by specifying the ``path`` and ``seed`` .

   :param path: The model's path where to load.
   :type path: str
   :param seed: Select the seed that model was trained with if it exits.
   :type seed: int

.. py:function:: xuance.tensorflow.agents.agents_marl.MARLAgents.act(**kwargs)
   
   Get actions for executing according to the joint observations, global states, available actions, etc.
   
   :param kwargs: Inputs informations.
   :type observations: Dict

.. py:function:: xuance.tensorflow.agents.agents_marl.MARLAgents.train(**kwargs)
   
   Train the multi-agent reinforcement learning models.

   :param kwargs: Informations for multi-agent training.
   :type observations: Dict
   :return: **info_train** - Informations of the training.
   :rtype: Dict


.. py:class:: 
   xuance.tensorflow.agents.agents_marl.linear_decay_or_increase(start, end, step_length)

   :param start: Start factor.
   :type start: np.float
   :param end: End factor.
   :type end: np.float
   :param step_length: The number of steps the factor decays or increases.
   :type step_length: int

.. py:function:: xuance.tensorflow.agents.agents_marl.linear_decay_or_increase.update()
   
   Update the factor once.


.. py:class:: 
   xuance.tensorflow.agents.agents_marl.RandomAgents(args, envs, device=None)

   :param args: Provides hyper parameters.
   :type args: Namespace
   :param envs: The vectorized environments.
   :type envs: xuance.environments.vector_envs.vector_env.VecEnv
   :param device: Choose CPU or GPU to train the model.
   :type device: str

.. py:function:: 
   xuance.tensorflow.agents.agents_marl.RandomAgents.act()
   
   Provide random actions for RandomAgents.

   :return: **random_actions** - Output random actions.
   :rtype: np.ndarray


.. raw:: html

   <br><hr>

MindSpore
------------------------------------------

.. py:class:: 
   xuance.mindspore.agents.agents_marl.MARLAgent(envs, policy, memory, learner, device, log_dir, model_dir)

   :param envs: The vectorized environments.
   :type envs: xuance.environments.vector_envs.vector_env.VecEnv
   :param policy: The policy that provides actions and values.
   :type policy: nn.Module
   :param memory: Experice replay buffer.
   :type memory: xuance.common.memory_tools.Buffer
   :param learner: The learner that updates parameters of policy.
   :type learner: xuance.mindspore.learner.Learner
   :param device: Choose CPU or GPU to train the model.
   :type device: str
   :param log_dir: The directory of log file, default is "./logs/".
   :type log_dir: str
   :param model_dir: The directory of model file, default is "./models/".
   :type model_dir: str

.. py:function:: xuance.mindspore.agents.agents_marl.MARLAgents.save_model(model_name)
   
   Save the model.

   :param model_name: The model's name to be saved.
   :type model_name: str

.. py:function:: xuance.mindspore.agents.agents_marl.MARLAgents.load_model(path, seed)

   Load a model by specifying the ``path`` and ``seed`` .

   :param path: The model's path where to load.
   :type path: str
   :param seed: Select the seed that model was trained with if it exits.
   :type seed: int

.. py:function:: xuance.mindspore.agents.agents_marl.MARLAgents.act(**kwargs)
   
   Get actions for executing according to the joint observations, global states, available actions, etc.
   
   :param kwargs: Inputs informations.
   :type observations: Dict

.. py:function:: xuance.mindspore.agents.agents_marl.MARLAgents.train(**kwargs)
   
   Train the multi-agent reinforcement learning models.

   :param kwargs: Informations for multi-agent training.
   :type observations: Dict
   :return: **info_train** - Informations of the training.
   :rtype: Dict


.. py:class:: 
   xuance.mindspore.agents.agents_marl.linear_decay_or_increase(start, end, step_length)

   :param start: Start factor.
   :type start: np.float
   :param end: End factor.
   :type end: np.float
   :param step_length: The number of steps the factor decays or increases.
   :type step_length: int

.. py:function:: xuance.mindspore.agents.agents_marl.linear_decay_or_increase.update()
   
   Update the factor once.


.. py:class:: 
   xuance.mindspore.agents.agents_marl.RandomAgents(args, envs, device=None)

   :param args: Provides hyper parameters.
   :type args: Namespace
   :param envs: The vectorized environments.
   :type envs: xuance.environments.vector_envs.vector_env.VecEnv
   :param device: Choose CPU or GPU to train the model.
   :type device: str

.. py:function:: 
   xuance.mindspore.agents.agents_marl.RandomAgents.act()
   
   Provide random actions for RandomAgents.

   :return: **random_actions** - Output random actions.
   :rtype: np.ndarray


.. raw:: html

   <br><hr>

Source Code
-----------------

.. tabs::

   .. group-tab:: PyTorch

      .. code-block:: python
         
         import os.path
         from xuance.torch.agents import *


         class MARLAgents(object):
            def __init__(self,
                        config: Namespace,
                        envs: DummyVecEnv_Pettingzoo,
                        policy: nn.Module,
                        memory: BaseBuffer,
                        learner: LearnerMAS,
                        device: Optional[Union[str, int, torch.device]] = None,
                        log_dir: str = "./logs/",
                        model_dir: str = "./models/"):
               self.args = config
               self.n_agents = config.n_agents
               self.dim_obs = self.args.dim_obs
               self.dim_act = self.args.dim_act
               self.dim_id = self.n_agents
               self.device = torch.device(
                     "cuda" if (torch.cuda.is_available() and config.device in ["gpu", "cuda:0"]) else "cpu")
               self.envs = envs
               self.start_training = config.start_training

               self.render = config.render
               self.nenvs = envs.num_envs
               self.policy = policy
               self.memory = memory
               self.learner = learner
               self.device = device
               self.log_dir = log_dir
               self.model_dir_save, self.model_dir_load = config.model_dir_save, config.model_dir_load
               create_directory(log_dir)
               create_directory(model_dir)

            def save_model(self, model_name):
               model_path = os.path.join(self.model_dir_save, model_name)
               self.learner.save_model(model_path)

            def load_model(self, path, seed=1):
               self.learner.load_model(path, seed)

            def act(self, **kwargs):
               raise NotImplementedError

            def train(self, **kwargs):
               raise NotImplementedError


         class linear_decay_or_increase(object):
            def __init__(self, start, end, step_length):
               self.start = start
               self.end = end
               self.step_length = step_length
               if self.start > self.end:
                     self.is_decay = True
                     self.delta = (self.start - self.end) / self.step_length
               else:
                     self.is_decay = False
                     self.delta = (self.end - self.start) / self.step_length
               self.epsilon = start

            def update(self):
               if self.is_decay:
                     self.epsilon = max(self.epsilon - self.delta, self.end)
               else:
                     self.epsilon = min(self.epsilon + self.delta, self.end)


         class RandomAgents(object):
            def __init__(self, args, envs, device=None):
               self.args = args
               self.n_agents = self.args.n_agents
               self.agent_keys = args.agent_keys
               self.action_space = self.args.action_space
               self.nenvs = envs.num_envs

            def act(self, obs_n, episode, test_mode, noise=False):
               rand_a = [[self.action_space[agent].sample() for agent in self.agent_keys] for e in range(self.nenvs)]
               random_actions = np.array(rand_a)
               return random_actions

            def load_model(self, model_dir):
               return

   
   .. group-tab:: TensorFlow

      .. code-block:: python

         from xuance.tensorflow.agents import *


         class MARLAgents(object):
            def __init__(self,
                        config: Namespace,
                        envs: DummyVecEnv_Pettingzoo,
                        policy: tk.Model,
                        memory: BaseBuffer,
                        learner: LearnerMAS,
                        device: str = "cpu:0",
                        log_dir: str = "./logs/",
                        model_dir: str = "./models/"):
               self.args = config
               self.n_agents = config.n_agents
               self.dim_obs = self.args.dim_obs
               self.dim_act = self.args.dim_act
               self.dim_id = self.n_agents
               self.device = device

               self.envs = envs
               self.start_training = config.start_training

               self.render = config.render
               self.nenvs = envs.num_envs
               self.policy = policy
               self.memory = memory
               self.learner = learner
               self.device = device
               self.log_dir = log_dir
               self.model_dir_save, self.model_dir_load = config.model_dir_save, config.model_dir_load
               create_directory(log_dir)
               create_directory(model_dir)

            def save_model(self, model_name):
               model_path = os.path.join(self.model_dir_save, model_name)
               self.learner.save_model(model_path)

            def load_model(self, path, seed=1):
               self.learner.load_model(path, seed)

            def act(self, **kwargs):
               raise NotImplementedError

            def train(self, **kwargs):
               raise NotImplementedError


         class linear_decay_or_increase(object):
            def __init__(self, start, end, step_length):
               self.start = start
               self.end = end
               self.step_length = step_length
               if self.start > self.end:
                     self.is_decay = True
                     self.delta = (self.start - self.end) / self.step_length
               else:
                     self.is_decay = False
                     self.delta = (self.end - self.start) / self.step_length
               self.epsilon = start

            def update(self):
               if self.is_decay:
                     self.epsilon = max(self.epsilon - self.delta, self.end)
               else:
                     self.epsilon = min(self.epsilon + self.delta, self.end)


         class RandomAgents(object):
            def __init__(self, args, envs, device=None):
               self.args = args
               self.n_agents = self.args.n_agents
               self.agent_keys = args.agent_keys
               self.action_space = self.args.action_space
               self.nenvs = envs.num_envs

            def act(self, obs_n, episode, test_mode, noise=False):
               rand_a = [[self.action_space[agent].sample() for agent in self.agent_keys] for e in range(self.nenvs)]
               random_actions = np.array(rand_a)
               return random_actions

            def load_model(self, model_dir):
               return


   .. group-tab:: MindSpore

      .. code-block:: python

         from xuance.mindspore.agents import *


         class MARLAgents(object):
            def __init__(self,
                        config: Namespace,
                        envs: DummyVecEnv_Pettingzoo,
                        policy: nn.Cell,
                        memory: BaseBuffer,
                        learner: LearnerMAS,
                        log_dir: str = "./logs/",
                        model_dir: str = "./models/"):
               self.args = config
               self.n_agents = config.n_agents
               self.dim_obs = self.args.dim_obs
               self.dim_act = self.args.dim_act
               self.dim_id = self.n_agents
               self.envs = envs
               self.start_training = config.start_training

               self.render = config.render
               self.nenvs = envs.num_envs
               self.policy = policy
               self.memory = memory
               self.learner = learner
               self.log_dir = log_dir
               self.model_dir_save, self.model_dir_load = config.model_dir_save, config.model_dir_load
               create_directory(log_dir)
               create_directory(model_dir)

               self.eye = ms.ops.Eye()
               self.expand_dims = ms.ops.ExpandDims()

            def save_model(self, model_name):
               model_path = self.model_dir_save
               self.learner.save_model(model_path, model_name)

            def load_model(self, path, seed=1):
               self.learner.load_model(path, seed)

            def act(self, **kwargs):
               raise NotImplementedError

            def train(self, **kwargs):
               raise NotImplementedError


         class linear_decay_or_increase(object):
            def __init__(self, start, end, step_length):
               self.start = start
               self.end = end
               self.step_length = step_length
               if self.start > self.end:
                     self.is_decay = True
                     self.delta = (self.start - self.end) / self.step_length
               else:
                     self.is_decay = False
                     self.delta = (self.end - self.start) / self.step_length
               self.epsilon = start

            def update(self):
               if self.is_decay:
                     self.epsilon = max(self.epsilon - self.delta, self.end)
               else:
                     self.epsilon = min(self.epsilon + self.delta, self.end)


         class RandomAgents(object):
            def __init__(self, args, envs):
               self.args = args
               self.n_agents = self.args.n_agents
               self.agent_keys = args.agent_keys
               self.action_space = self.args.action_space
               self.nenvs = envs.num_envs

            def act(self, obs_n, episode, test_mode, noise=False):
               rand_a = [[self.action_space[agent].sample() for agent in self.agent_keys] for e in range(self.nenvs)]
               random_actions = np.array(rand_a)
               return random_actions

            def load_model(self, model_dir):
               return



