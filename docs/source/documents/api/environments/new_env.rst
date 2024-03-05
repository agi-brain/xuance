Customized Environments
=====================================================

If the simulation environment used by the user is not included in XuanCe, it can be wrapped and stored in the "./xuance/environment" directory.
The specific steps for adding are as follows:

**Step 1: Create an Original Environment**
------------------------------------------------------------

Make a directory named, e.g., ``new_env``, and change the directory to the folder. 
Then, create a new pyhton file named new_env.py, in which a class named ``New_Env`` is defined. 
The ``New_Env`` is the original environment or a wrapper of the original environment,
which contains some necessary attributes, such as env_id, _episode_step, etc., 
and the necessary methods, such as self.render(), self.step(), etc.

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


**Step 2: Vectorize the Environment**
-------------------------------------------------------------------------

To improve sample efficiency and reduce the running time, 
it is suggested to vectorize the simulation environment, 
which enbale XuanCe to run multiple simulation environments simultaneously for sampling.

.. code-block:: python

    from xuance.environment.gym.gym_vec_env import SubprocVecEnv_Gym, DummyVecEnv_Gym, worker


    class SubprocVecEnv_New(SubprocVecEnv_Gym):
        """
        VecEnv that runs multiple environments in parallel in subproceses and communicates with them via pipes.
        Recommended to use when num_envs > 1 and step() can be a bottleneck.
        """
        def __init__(self, env_fns, context='spawn', in_series=1):
            """
            Arguments:
            env_fns: iterable of callables -  functions that create environments to run in subprocesses. Need to be cloud-pickleable
            in_series: number of environments to run in series in a single process
            (e.g. when len(env_fns) == 12 and in_series == 3, it will run 4 processes, each running 3 envs in series)
            """
            super(SubprocVecEnv_New, self).__init__(env_fns, context, in_series)


    class DummyVecEnv_New(DummyVecEnv_Gym):
        """
        VecEnv that does runs multiple environments sequentially, that is,
        the step and reset commands are send to one environment at a time.
        Useful when debugging and when num_env == 1 (in the latter case,
        avoids communication overhead)
        """
        def __init__(self, env_fns):
            super(DummyVecEnv_New, self).__init__(env_fns)


**Step 3: Import the Environment**
--------------------------------------------------------------------------

After that, import the vectorized environments in ./xuance/environments/__init__.py. 

.. code-block:: python

    from xuance.environment.new_env.new_vec_env import DummyVecEnv_New, SubprocVecEnv_New

    REGISTRY_VEC_ENV = {
        "Dummy_Gym": DummyVecEnv_Gym,
        "Dummy_Pettingzoo": DummyVecEnv_Pettingzoo,
        "Dummy_MAgent": DummyVecEnv_MAgent,
        "Dummy_StarCraft2": DummyVecEnv_StarCraft2,
        "Dummy_Football": DummyVecEnv_GFootball,
        "Dummy_Atari": DummyVecEnv_Atari,
        "Dummy_NewEnv": DummyVecEnv_New,  # Add the newly defined vectorized environment

        # multiprocess #
        "Subproc_Gym": SubprocVecEnv_Gym,
        "Subproc_Pettingzoo": SubprocVecEnv_Pettingzoo,
        "Subproc_StarCraft2": SubprocVecEnv_StarCraft2,
        "Subproc_Football": SubprocVecEnv_GFootball,
        "Subproc_Atari": SubprocVecEnv_Atari,
        "Subproc_NewEnv": SubprocVecEnv_New,  # Add the newly defined vectorized environment
    }

Then, add a condition after the "if ... elif ... else ..." statement to create the new environment.

.. code-block:: python

    def make_envs(config: Namespace):
    def _thunk():
        if config.env_name in PETTINGZOO_ENVIRONMENTS:
            from xuance.environment.pettingzoo.pettingzoo_env import PettingZoo_Env
            env = PettingZoo_Env(config.env_name, config.env_id, config.seed,
                                 continuous=config.continuous_action,
                                 render_mode=config.render_mode)
        # ...
        elif config.env_name == "NewEnv":  # Add the newly defined vectorized environment
            from xuance.environment.new_env.new_env import New_Env
            env = New_Env(config.env_id, config.seed, continuous=False)

        else:
            env = Gym_Env(config.env_id, config.seed, config.render_mode)

        return env

    if config.vectorize in REGISTRY_VEC_ENV.keys():
        return REGISTRY_VEC_ENV[config.vectorize]([_thunk for _ in range(config.parallels)])
    elif config.vectorize == "NOREQUIRED":
        return _thunk()
    else:
        raise NotImplementedError

**Step 4: Make the Environment**
----------------------------------------------------------------------

Let's take DQN for example, you need to prepare a config file named "xuance/configs/dqn/new_env.yaml".
Finally, you can run the method with new environment by the following commands:

.. code-block:: python

    import argparse
    from xuance import get_runner


    def parse_args():
        parser = argparse.ArgumentParser("Run a demo.")
        parser.add_argument("--method", type=str, default="dqn")
        parser.add_argument("--env", type=str, default="new_env")
        parser.add_argument("--env-id", type=str, default="new_id")
        parser.add_argument("--test", type=int, default=0)
        parser.add_argument("--device", type=str, default="cpu")

        return parser.parse_args()


    if __name__ == '__main__':
        parser = parse_args()
        runner = get_runner(method=parser.method,
                            env=parser.env,
                            env_id=parser.env_id,
                            parser_args=parser,
                            is_test=parser.test)
        runner.benchmark()
