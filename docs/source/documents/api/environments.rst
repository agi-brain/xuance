Environments
======================

Included Environments
----------------------

The software includes single agent task simulation environments such as Atari, Mujoco, Classic Control, and Box2D under gym.
It also includes multi-agent task simulation environments such as MPE and SISL under the open-source environment PettingZoo, StarCraft2, MAgent2, Google Football, etc.
Each simulation environment contains a rich variety of task scenarios, as shown in the table below.

Customized Environments
-------------------------

If the simulation environment used by the user is not listed in Table 1, it can be wrapped and stored in the "./xuance/environment" directory.
The specific steps for adding are as follows:

**Step 1**:

Make a directory named, e.g., ``new_env``, and change the directory to the folder. 
Then, create a new pyhton file named new_env.py, in which a class named ``New_Env`` is defined. 
The ``New_Env`` is the original environment or a wrapper of the original environment,
which contains some necessary attributes, such as env_id, _episode

.. code-block:: python

    """
    This is an example of creating a new environment in XuanCe.
    """
    from gym.spaces import Box, Discrete
    import numpy as np


    class New_Env:
        def __init__(self, env_id: str, seed: int, *args, **kwargs):
            continuous = kwargs['continuous']
            self.env_id = env_id
            self._episode_step = 0
            self._episode_score = 0.0
            self.observation_space = Box(low=0, high=1, shape=[8, ], dtype=np.float, seed=seed)
            if continuous:
                self.action_space = Box(low=0, high=1, shape=[2, ], dtype=np.float, seed=seed)
            else:
                self.action_space = Discrete(n=2, seed=seed)
            self.max_episode_steps = 100

        def close(self):
            pass

        def render(self):
            pass

        def reset(self):
            obs, info = self.observation_space.sample(), {}  # reset the environment and get observations and info here.
            self._episode_step = 0
            self._episode_score = 0.0
            info["episode_step"] = self._episode_step
            return obs, info

        def step(self, actions):
            # Execute the actions and get next observations, rewards, and other information.
            observation, reward, terminated, truncated, info = self.observation_space.sample(), 0, False, False, {}

            self._episode_step += 1
            self._episode_score += reward
            info["episode_step"] = self._episode_step  # current episode step
            info["episode_score"] = self._episode_score  # the accumulated rewards
            return observation, reward, terminated, truncated, info


**Step 2**:

To import the custom environment class My_Env in the ./xuance/environment/__init__.py file, you can use the following code:

.. code-block:: python

    from .myenv.my_env import My_Env


Vectorize the Environment
----------------------------------------

To improve sampling efficiency and save algorithm running time, this software supports setting up a vectorized simulation environment, which involves running multiple simulation environments simultaneously for sampling.

The definition of the base class for vectorized environments, `VecEnv`, can be found in the `./xuance/environment/vector_envs/vector_env.py` file.

On top of this base class, there are two inherited classes: `DummyVecEnv` and `DummyVecEnv_MAS`. They are respectively used to implement vectorized simulation environments for single-agent and multi-agent scenarios. The code for these classes can be found in the `./environment/vector_envs/dummy_vec_env.py` file.