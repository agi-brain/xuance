Environments
======================

Included Environments
----------------------

The software includes single agent task simulation environments such as Atari, Mujoco, Classic Control, and Box2D under gym.
It also includes multi-agent task simulation environments such as MPE and SISL under the open-source environment PettingZoo, StarCraft2, MAgent2, Google Football, etc.
Each simulation environment contains a rich variety of task scenarios, as shown in the table below.

Customized Environments
-------------------------

If the simulation environment used by the user is not listed in Table 1, it can be wrapped and stored in the "./environment" directory.
The specific steps for adding are as follows:

步骤一：



::

    class My_Env(gym.Env):
        def __init__(env_id: str, seed: str)
            self.env = make_env(env_id)
            self.env.seed(seed)
            self.obeservation_space = Box(0, 1, self.env.dim_state)
        self.action_space = self.env.action_space
            self.metadata = self.env.metadata
            self.reward_range = self.env.reward_range
            self.spec = self.env.spec
            super(My_Env, self).__init__()

        def reset(self):
            return self.env.reset()

        def step(self, action):
            return self.env.step()
        def seed(self, seed):
            return self.env.seed(seed)
        def render(self, mode)
            return self.env.render(mode)
        def close(self)
            self.env.close()

步骤二：在./environment/__init__.py文件中导入自定义的环境类My_Env。
::

    from .myenv.my_env import My_Env

向量化仿真环境
----------------------
为了提高采样效率，节省算法运行时间，本软件支持向量化仿真环境设置，即运行多个仿真环境同时采样。
向量化环境基类VecEnv的定义位于./environment/vector_envs/vector_env.py文件中，
在此基类上定义继承类DummyVecEnv及DummyVecEnv_MAS，分别用于实现单智能体和多智能体向量化仿真环境，
代码位于./environment/vector_envs/dummy_vec_env.py文件中。
