Agent
=======================

To create a new Agent, you should build a class inherit from ``xuance.torch.agents.agent.Agent`` , ``xuance.tensorflow.agents.agent.Agent``, or ``xuance.mindspore.agents.agent.Agent``.

PyTorch
------------------------------------------

.. py:class:: 
   xuance.torch.agents.agent.Agent(config, envs, policy, memory, learner, device, log_dir, model_dir)

   :param config: Provides hyper parameters.
   :type config: Namespace
   :param envs: The vectorized environments.
   :type envs: xuance.environments.vector_envs.vector_env.VecEnv
   :param policy: The policy that provides actions and values.
   :type policy: nn.Module
   :param memory: Experice replay buffer.
   :type memory: xuance.common.memory_tools.Buffer
   :param learner: The learner that updates parameters of policy.
   :type learner: xuance.torch.learner.Learner
   :param device: Choose CPU or GPU to train the model.
   :type device: str, int, torch.device
   :param log_dir: The directory of log file, default is "./logs/".
   :type log_dir: str
   :param model_dir: The directory of model file, default is "./models/".
   :type model_dir: str

.. py:function:: xuance.torch.agents.agent.Agent.save_model(model_name)
   
   Save the model.

   :param model_name: The model's name to be saved.
   :type model_name: str

.. py:function:: xuance.torch.agents.agent.Agent.load_model(path, seed)

   Load a model by specifying the ``path`` and ``seed`` .

   :param path: The model's path where to load.
   :type path: str
   :param seed: Select the seed that model was trained with if it exits.
   :type seed: int

.. py:function:: xuance.torch.agents.agent.Agent.log_infos(info, x_index)

   Visualize the training information via wandb or tensorboard.

   :param info: Information to be visualized.
   :type info: dict
   :param x_index: Current step.
   :type x_index: int

.. py:function:: xuance.torch.agents.agent.Agent.log_videos(info, fps x_index)

   Visualize the interaction between agent and environment by uploading the videos with wandb or tensorboard.

   :param info: Information to be visualized.
   :type info: dict
   :param fps: Frames per second.
   :type fps: int
   :param x_index: Current step.
   :type x_index: int

.. py:function:: xuance.torch.agents.agent.Agent._process_observation(observations)

   Normalize the original observations.

   :param observations: The original observations of agent.
   :type observations: np.ndarray
   :return: The normalized observations.
   :rtype: np.ndarray

.. py:function:: xuance.torch.agents.agent.Agent._process_reward(rewards)

   Normalize the original rewards.

   :param rewards: The original rewards of agent.
   :type rewards: np.ndarray
   :return: The normalized observations rewards.
   :rtype: np.ndarray

.. py:function:: xuance.torch.agents.agent.Agent._action(observations)
   
   Get actions for executing according to the observations.
   
   :param observations: The original observations of agent.
   :type observations: np.ndarray

.. py:function:: xuance.torch.agents.agent.Agent.train(steps)
   
   Train the agents with ``steps`` steps.

   :param steps: The training steps.
   :type steps: int

.. py:function:: xuance.torch.agents.agent.Agent.test(env_fn, steps)
   
   Test the agents.

   :param env_fn: The function of making environments.
   :param steps: The training steps.
   :type steps: int

.. py:function:: xuance.torch.agents.agent.Agent.finish()
   
   Finish the wandb or tensorboard.


.. raw:: html

   <br><hr>

TensorFlow
------------------------------------------

.. py:class:: 
   xuance.tensorflowtensorflow.agent.agent.Agent(config, envs, policy, memory, learner, device, log_dir, model_dir)

   :param config: Provides hyper parameters.
   :type config: Namespace
   :param envs: The vectorized environments.
   :type envs: xuance.environments.vector_envs.vector_env.VecEnv
   :param policy: The policy that provides actions and values.
   :type policy: nn.Module
   :param memory: Experice replay buffer.
   :type memory: xuance.common.memory_tools.Buffer
   :param learner: The learner that updates parameters of policy.
   :type learner: xuance.tensorflow.learner.Learner
   :param device: Choose CPU or GPU to train the model.
   :type device: str
   :param log_dir: The directory of log file, default is "./logs/".
   :type log_dir: str
   :param model_dir: The directory of model file, default is "./models/".
   :type model_dir: str


.. raw:: html

   <br><hr>

MindSpore
------------------------------------------

.. py:class:: 
   xuance.mindsporetensorflow.agent.agent.Agent(envs, policy, memory, learner, device, log_dir, model_dir)

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


.. raw:: html

   <br><hr>

Source Code
-----------------

.. tabs::

   .. group-tab:: PyTorch

      .. code-block:: python
         
         import socket
         import time
         from pathlib import Path
         from xuance.torch.agents import *


         class Agent(ABC):
            """The class of basic agents.

            Args:
               config: the Namespace variable that provides hyper-parameters and other settings.
               envs: the vectorized environments.
               policy: the neural network modules of the agent.
               memory: the experience replay buffer.
               learner: the learner for the corresponding agent.
               device: the calculating device of the model, such as CPU or GPU.
               log_dir: the directory of the log file.
               model_dir: the directory for models saving.
            """
            def __init__(self,
                        config: Namespace,
                        envs: DummyVecEnv_Gym,
                        policy: nn.Module,
                        memory: Buffer,
                        learner: Learner,
                        device: Optional[Union[str, int, torch.device]] = None,
                        log_dir: str = "./logs/",
                        model_dir: str = "./models/"):
               self.config = config
               self.envs = envs
               self.policy = policy
               self.memory = memory
               self.learner = learner

               self.observation_space = envs.observation_space
               self.comm = MPI.COMM_WORLD
               self.obs_rms = RunningMeanStd(shape=space2shape(self.observation_space), comm=self.comm, use_mpi=False)
               self.ret_rms = RunningMeanStd(shape=(), comm=self.comm, use_mpi=False)
               self.use_obsnorm = config.use_obsnorm
               self.use_rewnorm = config.use_rewnorm
               self.obsnorm_range = config.obsnorm_range
               self.rewnorm_range = config.rewnorm_range
               self.returns = np.zeros((self.envs.num_envs,), np.float32)

               time_string = time.asctime().replace(" ", "").replace(":", "_")
               seed = f"seed_{self.config.seed}_"
               model_dir_save = os.path.join(os.getcwd(), model_dir, seed + time_string)
               if (not os.path.exists(model_dir_save)) and (not config.test_mode):
                     os.makedirs(model_dir_save)

               # logger
               if config.logger == "tensorboard":
                     log_dir = os.path.join(os.getcwd(), config.log_dir, seed + time_string)
                     if not os.path.exists(log_dir):
                        os.makedirs(log_dir)
                     self.writer = SummaryWriter(log_dir)
                     self.use_wandb = False
               elif config.logger == "wandb":
                     config_dict = vars(config)
                     wandb_dir = Path(os.path.join(os.getcwd(), config.log_dir))
                     if not wandb_dir.exists():
                        os.makedirs(str(wandb_dir))
                     wandb.init(config=config_dict,
                              project=config.project_name,
                              entity=config.wandb_user_name,
                              notes=socket.gethostname(),
                              dir=wandb_dir,
                              group=config.env_id,
                              job_type=config.agent,
                              name=time.asctime(),
                              reinit=True,
                              settings=wandb.Settings(start_method="fork")
                              )
                     # os.environ["WANDB_SILENT"] = "True"
                     self.use_wandb = True
               else:
                     raise "No logger is implemented."

               self.device = device
               self.log_dir = log_dir
               self.model_dir_save = model_dir_save
               self.model_dir_load = model_dir
               create_directory(log_dir)
               self.current_step = 0
               self.current_episode = np.zeros((self.envs.num_envs,), np.int32)

            def save_model(self, model_name):
               model_path = self.model_dir_save + "/" + model_name
               self.learner.save_model(model_path)

            def load_model(self, path, seed=1):
               self.learner.load_model(path, seed)

            def log_infos(self, info: dict, x_index: int):
               """
               info: (dict) information to be visualized
               n_steps: current step
               """
               if self.use_wandb:
                     for k, v in info.items():
                        wandb.log({k: v}, step=x_index)
               else:
                     for k, v in info.items():
                        try:
                           self.writer.add_scalar(k, v, x_index)
                        except:
                           self.writer.add_scalars(k, v, x_index)

            def log_videos(self, info: dict, fps: int, x_index: int=0):
               if self.use_wandb:
                     for k, v in info.items():
                        wandb.log({k: wandb.Video(v, fps=fps, format='gif')}, step=x_index)
               else:
                     for k, v in info.items():
                        self.writer.add_video(k, v, fps=fps, global_step=x_index)

            def _process_observation(self, observations):
               if self.use_obsnorm:
                     if isinstance(self.observation_space, Dict):
                        for key in self.observation_space.spaces.keys():
                           observations[key] = np.clip(
                                 (observations[key] - self.obs_rms.mean[key]) / (self.obs_rms.std[key] + EPS),
                                 -self.obsnorm_range, self.obsnorm_range)
                     else:
                        observations = np.clip((observations - self.obs_rms.mean) / (self.obs_rms.std + EPS),
                                                -self.obsnorm_range, self.obsnorm_range)
                     return observations
               else:
                     return observations

            def _process_reward(self, rewards):
               if self.use_rewnorm:
                     std = np.clip(self.ret_rms.std, 0.1, 100)
                     return np.clip(rewards / std, -self.rewnorm_range, self.rewnorm_range)
               else:
                     return rewards

            @abstractmethod
            def _action(self, observations):
               raise NotImplementedError

            @abstractmethod
            def train(self, steps):
               raise NotImplementedError

            @abstractmethod
            def test(self, env_fn, steps):
               raise NotImplementedError

            def finish(self):
               if self.use_wandb:
                     wandb.finish()
               else:
                     self.writer.close()


         def get_total_iters(agent_name, args):
            return args.running_steps

   
   .. group-tab:: TensorFlow

      .. code-block:: python

         import socket
         from pathlib import Path
         from xuance.tensorflow.agents import *


         class Agent(ABC):
            def __init__(self,
                        config: Namespace,
                        envs: DummyVecEnv_Gym,
                        policy: tk.Model,
                        memory: Buffer,
                        learner: Learner,
                        device: str = "cpu",
                        log_dir: str = "./logs/",
                        model_dir: str = "./models/"):
               self.config = config
               self.envs = envs
               self.policy = policy
               self.memory = memory
               self.learner = learner

               self.observation_space = envs.observation_space
               self.comm = MPI.COMM_WORLD
               self.obs_rms = RunningMeanStd(shape=space2shape(self.observation_space), comm=self.comm, use_mpi=False)
               self.ret_rms = RunningMeanStd(shape=(), comm=self.comm, use_mpi=False)
               self.use_obsnorm = config.use_obsnorm
               self.use_rewnorm = config.use_rewnorm
               self.obsnorm_range = config.obsnorm_range
               self.rewnorm_range = config.rewnorm_range
               self.returns = np.zeros((self.envs.num_envs,), np.float32)

               # logger
               time_string = time.asctime().replace(" ", "").replace(":", "_")
               seed = f"seed_{self.config.seed}_"
               model_dir_save = os.path.join(os.getcwd(), model_dir, seed + time_string)
               if (not os.path.exists(model_dir_save)) and (not config.test_mode):
                     os.makedirs(model_dir_save)

               # logger
               if config.logger == "tensorboard":
                     log_dir = os.path.join(os.getcwd(), config.log_dir, seed + time_string)
                     if not os.path.exists(log_dir):
                        os.makedirs(log_dir)
                     self.writer = SummaryWriter(log_dir)
                     self.use_wandb = False
               elif config.logger == "wandb":
                     config_dict = vars(config)
                     wandb_dir = Path(os.path.join(os.getcwd(), config.log_dir))
                     if not wandb_dir.exists():
                        os.makedirs(str(wandb_dir))
                     wandb.init(config=config_dict,
                              project=config.project_name,
                              entity=config.wandb_user_name,
                              notes=socket.gethostname(),
                              dir=wandb_dir,
                              group=config.env_id,
                              job_type=config.agent,
                              name=time.asctime(),
                              reinit=True
                              )
                     # os.environ["WANDB_SILENT"] = "True"
                     self.use_wandb = True
               else:
                     raise "No logger is implemented."

               self.device = device
               self.log_dir = log_dir
               self.model_dir_save = model_dir_save
               self.model_dir_load = model_dir
               create_directory(log_dir)
               self.current_step = 0
               self.current_episode = np.zeros((self.envs.num_envs,), np.int32)
            
            def save_model(self, model_name):
               model_path = os.path.join(self.model_dir_save, model_name)
               self.learner.save_model(model_path)

            def load_model(self, path, seed=1):
               self.learner.load_model(path, seed)

            def log_infos(self, info: dict, x_index: int):
               """
               info: (dict) information to be visualized
               n_steps: current step
               """
               if self.use_wandb:
                     for k, v in info.items():
                        wandb.log({k: v}, step=x_index)
               else:
                     for k, v in info.items():
                        try:
                           self.writer.add_scalar(k, v, x_index)
                        except:
                           self.writer.add_scalars(k, v, x_index)

            def log_videos(self, info: dict, fps: int, x_index: int=0):
               if self.use_wandb:
                     for k, v in info.items():
                        wandb.log({k: wandb.Video(v, fps=fps, format='gif')}, step=x_index)
               else:
                     for k, v in info.items():
                        self.writer.add_video(k, v, fps=fps, global_step=x_index)

            def _process_observation(self, observations):
               if self.use_obsnorm:
                     if isinstance(self.observation_space, Dict):
                        for key in self.observation_space.spaces.keys():
                           observations[key] = np.clip(
                                 (observations[key] - self.obs_rms.mean[key]) / (self.obs_rms.std[key] + EPS),
                                 -self.obsnorm_range, self.obsnorm_range)
                     else:
                        observations = np.clip((observations - self.obs_rms.mean) / (self.obs_rms.std + EPS),
                                                -self.obsnorm_range, self.obsnorm_range)
                     return observations
               else:
                     return observations

            def _process_reward(self, rewards):
               if self.use_rewnorm:
                     std = np.clip(self.ret_rms.std, 0.1, 100)
                     return np.clip(rewards / std, -self.rewnorm_range, self.rewnorm_range)
               else:
                     return rewards

            @abstractmethod
            def _action(self, observations):
               raise NotImplementedError

            @abstractmethod
            def train(self, steps):
               raise NotImplementedError

            @abstractmethod
            def test(self, env, episodes):
               raise NotImplementedError

            def finish(self):
               if self.use_wandb:
                     wandb.finish()
               else:
                     self.writer.close()


         def get_total_iters(agent_name, args):
            return args.running_steps


   .. group-tab:: MindSpore

      .. code-block:: python

         import socket
         import time
         from pathlib import Path
         from xuance.mindspore.agents import *


         class Agent(ABC):
            def __init__(self,
                        config: Namespace,
                        envs: DummyVecEnv_Gym,
                        policy: nn.Cell,
                        memory: Buffer,
                        learner: Learner,
                        log_dir: str = "./logs/",
                        model_dir: str = "./models/"):
               self.config = config
               self.envs = envs
               self.policy = policy
               self.memory = memory
               self.learner = learner

               self.observation_space = envs.observation_space
               self.comm = MPI.COMM_WORLD
               self.obs_rms = RunningMeanStd(shape=space2shape(self.observation_space), comm=self.comm, use_mpi=False)
               self.ret_rms = RunningMeanStd(shape=(), comm=self.comm, use_mpi=False)
               self.use_obsnorm = config.use_obsnorm
               self.use_rewnorm = config.use_rewnorm
               self.obsnorm_range = config.obsnorm_range
               self.rewnorm_range = config.rewnorm_range
               self.returns = np.zeros((self.envs.num_envs,), np.float32)

               time_string = time.asctime().replace(" ", "").replace(":", "_")
               seed = f"seed_{self.config.seed}_"
               model_dir_save = os.path.join(os.getcwd(), model_dir, seed + time_string)
               if (not os.path.exists(model_dir_save)) and (not config.test_mode):
                     os.makedirs(model_dir_save)

               # logger
               if config.logger == "tensorboard":
                     log_dir = os.path.join(os.getcwd(), config.log_dir, seed + time_string)
                     if not os.path.exists(log_dir):
                        os.makedirs(log_dir)
                     self.writer = SummaryWriter(log_dir)
                     self.use_wandb = False
               elif config.logger == "wandb":
                     config_dict = vars(config)
                     wandb_dir = Path(os.path.join(os.getcwd(), config.log_dir))
                     if not wandb_dir.exists():
                        os.makedirs(str(wandb_dir))
                     wandb.init(config=config_dict,
                              project=config.project_name,
                              entity=config.wandb_user_name,
                              notes=socket.gethostname(),
                              dir=wandb_dir,
                              group=config.env_id,
                              job_type=config.agent,
                              name=time.asctime(),
                              reinit=True
                              )
                     # os.environ["WANDB_SILENT"] = "True"
                     self.use_wandb = True
               else:
                     raise "No logger is implemented."

               self.log_dir = log_dir
               self.model_dir_save = model_dir_save
               self.model_dir_load = model_dir
               create_directory(log_dir)
               self.current_step = 0
               self.current_episode = np.zeros((self.envs.num_envs,), np.int32)

            def save_model(self, model_name):
               model_path = self.model_dir_save
               self.learner.save_model(model_path, model_name)

            def load_model(self, path, seed=1):
               self.learner.load_model(path, seed)

            def log_infos(self, info: dict, x_index: int):
               """
               info: (dict) information to be visualized
               n_steps: current step
               """
               if self.use_wandb:
                     for k, v in info.items():
                        wandb.log({k: v}, step=x_index)
               else:
                     for k, v in info.items():
                        try:
                           self.writer.add_scalar(k, v, x_index)
                        except:
                           self.writer.add_scalars(k, v, x_index)

            def log_videos(self, info: dict, fps: int, x_index: int = 0):
               if self.use_wandb:
                     for k, v in info.items():
                        wandb.log({k: wandb.Video(v, fps=fps, format='gif')}, step=x_index)
               else:
                     for k, v in info.items():
                        self.writer.add_video(k, v, fps=fps, global_step=x_index)

            def _process_observation(self, observations):
               if self.use_obsnorm:
                     if isinstance(self.observation_space, Dict):
                        for key in self.observation_space.spaces.keys():
                           observations[key] = np.clip(
                                 (observations[key] - self.obs_rms.mean[key]) / (self.obs_rms.std[key] + EPS),
                                 -self.obsnorm_range, self.obsnorm_range)
                     else:
                        observations = np.clip((observations - self.obs_rms.mean) / (self.obs_rms.std + EPS),
                                                -self.obsnorm_range, self.obsnorm_range)
                     return observations
               else:
                     return observations

            def _process_reward(self, rewards):
               if self.use_rewnorm:
                     std = np.clip(self.ret_rms.std, 0.1, 100)
                     return np.clip(rewards / std, -self.rewnorm_range, self.rewnorm_range)
               else:
                     return rewards

            @abstractmethod
            def _action(self, observations):
               raise NotImplementedError

            @abstractmethod
            def train(self, steps):
               raise NotImplementedError

            @abstractmethod
            def test(self, env_fn, steps):
               raise NotImplementedError

            def finish(self):
               if self.use_wandb:
                     wandb.finish()
               else:
                     self.writer.close()


         def get_total_iters(agent_name, args):
            return args.running_steps


