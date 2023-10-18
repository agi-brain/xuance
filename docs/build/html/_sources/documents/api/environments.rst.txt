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

**Step 1**:

To create a folder named "myenv" (you can choose the name), and navigate into the "myenv" folder, follow these steps:

.. code-block:: python

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

**Step 2**:

To import the custom environment class My_Env in the ./environment/__init__.py file, you can use the following code:

.. code-block:: python

    from .myenv.my_env import My_Env


Vectorize the Environment
----------------------------------------

To improve sampling efficiency and save algorithm running time, this software supports setting up a vectorized simulation environment, which involves running multiple simulation environments simultaneously for sampling.

The definition of the base class for vectorized environments, `VecEnv`, can be found in the `./environment/vector_envs/vector_env.py` file.

On top of this base class, there are two inherited classes: `DummyVecEnv` and `DummyVecEnv_MAS`. They are respectively used to implement vectorized simulation environments for single-agent and multi-agent scenarios. The code for these classes can be found in the `./environment/vector_envs/dummy_vec_env.py` file.