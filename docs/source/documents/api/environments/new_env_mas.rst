Customized Environments for Multi-Agent Systems
=====================================================

If the multi-agent environment used by the user is not included in XuanCe, it can also be wrapped and stored in the "./xuance/environment" directory.
The specific steps for adding are as follows:

**Step 1: Create an Original Multi-Agent Environment**
------------------------------------------------------------

Make a directory named, e.g., ``new_env_mas``, and change the directory to the folder. 
Then, create a new pyhton file named new_env.py, in which a class named ``New_Env_MAS`` for example is defined. 
The ``New_Env_MAS`` is the original environment or a wrapper of the original environment,
which contains some necessary attributes and methods.

.. code-block:: python

    """
    This is an example of creating a new environment in XuanCe for multi-agent system.
    This example
    """
    from gymnasium.spaces import Box, Discrete
    import numpy as np


    class New_Env_MAS:
        def __init__(self, env_id: str, seed: int, **kwargs):
            self.n_agents = 3
            self.dim_obs = 10  # dimension of one agent's observation
            self.dim_state = 12  # dimension of global state
            self.dim_action = 2  # dimension of actions (continuous)
            self.n_actions = 5  # number of discrete actions (discrete)
            self.state_space = Box(low=0, high=1, shape=[self.dim_state, ], dtype=np.float32, seed=seed)
            self.observation_space = Box(low=0, high=1, shape=[self.dim_obs, ], dtype=np.float32, seed=seed)
            if kwargs['continuous']:
                self.action_space = Box(low=0, high=1, shape=[self.dim_action, ], dtype=np.float32, seed=seed)
            else:
                self.action_space = Discrete(n=self.n_actions, seed=seed)

            self._episode_step = 0
            self._episode_score = 0.0

            self.max_episode_steps = 100
            self.env_info = {
                "n_agents": self.n_agents,
                "observation_space": self.observation_space,
                "action_space": self.action_space,
                "state_space": self.state_space,
                "episode_limit": self.max_episode_steps,
            }

        def close(self):
            pass

        def render(self):
            return np.zeros([2, 2, 2])

        def reset(self):
            obs = np.array([self.observation_space.sample() for _ in range(self.n_agents)])
            info = {}
            self._episode_step = 0
            self._episode_score = 0.0
            info["episode_step"] = self._episode_step
            return obs, info

        def step(self, actions):
            # Execute the actions and get next observations, rewards, and other information.
            observation = np.array([self.observation_space.sample() for _ in range(self.n_agents)])
            reward, info = np.zeros([self.n_agents, 1]), {}
            terminated = [False for _ in range(self.n_agents)]
            truncated = [True for _ in range(self.n_agents)] if (self._episode_step >= self.max_episode_steps) else [False for _ in range(self.n_agents)]

            self._episode_step += 1
            self._episode_score += reward
            info["episode_step"] = self._episode_step  # current episode step
            info["episode_score"] = self._episode_score  # the accumulated rewards
            return observation, reward, terminated, truncated, info

        def get_agent_mask(self):
            return np.ones(self.n_agents, dtype=np.bool_)

        def state(self):
            return np.zeros([self.dim_state])



**Step 2: Vectorize the Environment**
-------------------------------------------------------------------------

Then, vectorize the simulation environment to enbale XuanCe to run multiple simulation environments simultaneously for sampling.

.. code-block:: python

    from xuance.environment.vector_envs.vector_env import NotSteppingError
    from xuance.environment.gym.gym_vec_env import DummyVecEnv_Gym, SubprocVecEnv_Gym
    from xuance.common import combined_shape
    from gymnasium.spaces import Box, Discrete
    import numpy as np
    import multiprocessing as mp
    from xuance.environment.vector_envs.subproc_vec_env import clear_mpi_env_vars, flatten_list, CloudpickleWrapper
    from xuance.environment.vector_envs.vector_env import VecEnv


    def worker(remote, parent_remote, env_fn_wrappers):
        def step_env(env, action):
            obs, reward_n, terminated, truncated, info = env.step(action)
            return obs, reward_n, terminated, truncated, info

        parent_remote.close()
        envs = [env_fn_wrapper() for env_fn_wrapper in env_fn_wrappers.x]
        try:
            while True:
                cmd, data = remote.recv()
                if cmd == 'step':
                    remote.send([step_env(env, action) for env, action in zip(envs, data)])
                elif cmd == 'reset':
                    remote.send([env.reset() for env in envs])
                elif cmd == 'render':
                    remote.send([env.render(data) for env in envs])
                elif cmd == 'state':
                    remote.send([env.state() for env in envs])
                elif cmd == 'get_agent_mask':
                    remote.send([env.get_agent_mask() for env in envs])
                elif cmd == 'close':
                    remote.close()
                    break
                elif cmd == 'get_env_info':
                    env_info = envs[0].env_info
                    remote.send(CloudpickleWrapper(env_info))
                else:
                    raise NotImplementedError
        except KeyboardInterrupt:
            print('SubprocVecEnv worker: got KeyboardInterrupt')
        finally:
            for env in envs:
                env.close()


    class SubprocVecEnv_New_MAS(SubprocVecEnv_Gym):
        """
        VecEnv that runs multiple environments in parallel in subproceses and communicates with them via pipes.
        Recommended to use when num_envs > 1 and step() can be a bottleneck.
        """

        def __init__(self, env_fns, context='spawn', in_series=1):
            """
            Arguments:
            env_fns: iterable of callables -  functions that create environments to run in subprocesses. Need to be cloud-pickleable
            (e.g. when len(env_fns) == 12 and in_series == 3, it will run 4 processes, each running 3 envs in series)
            """
            self.waiting = False
            self.closed = False
            self.n_remotes = num_envs = len(env_fns)
            env_fns = np.array_split(env_fns, self.n_remotes)
            ctx = mp.get_context(context)
            self.remotes, self.work_remotes = zip(*[ctx.Pipe() for _ in range(self.n_remotes)])
            self.ps = [ctx.Process(target=worker, args=(work_remote, remote, CloudpickleWrapper(env_fn)))
                       for (work_remote, remote, env_fn) in zip(self.work_remotes, self.remotes, env_fns)]
            for p in self.ps:
                p.daemon = True  # if the main process crashes, we should not cause things to hang
                with clear_mpi_env_vars():
                    p.start()
            for remote in self.work_remotes:
                remote.close()

            self.remotes[0].send(('get_env_info', None))
            env_info = self.remotes[0].recv().x
            self.viewer = None
            VecEnv.__init__(self, num_envs, env_info["observation_space"], env_info["action_space"])

            self.state_space = env_info["state_space"]
            self.dim_state = self.state_space.shape[-1]
            self.num_agents = env_info["n_agents"]
            self.obs_shape = (self.num_agents, self.observation_space.shape[-1])
            if isinstance(self.action_space, Box):
                self.act_shape = (self.num_agents, self.action_space.shape[-1])
            elif isinstance(self.action_space, Discrete):
                self.act_shape = (self.num_agents,)
            self.rew_shape = (self.num_agents, 1)

            self.buf_obs = np.zeros(combined_shape(self.num_envs, self.obs_shape), dtype=np.float32)
            self.buf_state = np.zeros(combined_shape(self.num_envs, self.dim_state), dtype=np.float32)
            self.buf_agent_mask = np.ones([self.num_envs, self.num_agents], dtype=np.bool_)
            self.buf_terminals = np.zeros((self.num_envs, self.num_agents), dtype=np.bool_)
            self.buf_truncations = np.zeros((self.num_envs, self.num_agents), dtype=np.bool_)
            self.buf_rews = np.zeros((self.num_envs,) + self.rew_shape, dtype=np.float32)
            self.buf_info = [{} for _ in range(self.num_envs)]

            self.max_episode_length = env_info["episode_limit"]
            self.actions = None

        def step_wait(self):
            self._assert_not_closed()
            if not self.waiting:
                raise NotSteppingError
            results = [remote.recv() for remote in self.remotes]
            results = flatten_list(results)
            obs, rews, dones, truncated, infos = zip(*results)
            self.buf_obs, self.buf_rews = np.array(obs), np.array(rews)
            self.buf_terminals, self.buf_truncations, self.buf_infos = np.array(dones), np.array(truncated), list(infos)
            for e in range(self.num_envs):
                if all(dones[e]) or all(truncated[e]):
                    self.remotes[e].send(('reset', None))
                    result = self.remotes[e].recv()
                    obs_reset, _ = flatten_list(result)
                    self.buf_infos[e]["reset_obs"] = obs_reset
                    self.remotes[e].send(('get_agent_mask', None))
                    result = self.remotes[e].recv()
                    self.buf_infos[e]["reset_agent_mask"] = flatten_list(result)
                    self.remotes[e].send(('state', None))
                    result = self.remotes[e].recv()
                    self.buf_infos[e]["reset_state"] = flatten_list(result)
            self.waiting = False
            return self.buf_obs.copy(), self.buf_rews.copy(), self.buf_terminals.copy(), self.buf_truncations.copy(), self.buf_infos.copy()

        def global_state(self):
            self._assert_not_closed()
            for pipe in self.remotes:
                pipe.send(('state', None))
            states = [pipe.recv() for pipe in self.remotes]
            states = flatten_list(states)
            self.buf_state = np.array(states)
            return self.buf_state

        def agent_mask(self):
            self._assert_not_closed()
            for pipe in self.remotes:
                pipe.send(('get_agent_mask', None))
            masks = [pipe.recv() for pipe in self.remotes]
            masks = flatten_list(masks)
            self.buf_agent_mask = np.array(masks)
            return self.buf_agent_mask


    class DummyVecEnv_New_MAS(DummyVecEnv_Gym):
        def __init__(self, env_fns):
            self.waiting = False
            self.envs = [fn() for fn in env_fns]
            env = self.envs[0]
            env_info = env.env_info
            self.viewer = None
            VecEnv.__init__(self, len(env_fns), env_info["observation_space"], env_info["action_space"])

            self.state_space = env_info["state_space"]
            self.dim_state = self.state_space.shape[-1]
            self.num_agents = env_info["n_agents"]
            self.obs_shape = (self.num_agents, self.observation_space.shape[-1])
            if isinstance(self.action_space, Box):
                self.act_shape = (self.num_agents, self.action_space.shape[-1])
            elif isinstance(self.action_space, Discrete):
                self.act_shape = (self.num_agents, )
            self.rew_shape = (self.num_agents, 1)

            self.buf_obs = np.zeros(combined_shape(self.num_envs, self.obs_shape), dtype=np.float32)
            self.buf_state = np.zeros(combined_shape(self.num_envs, self.dim_state), dtype=np.float32)
            self.buf_agent_mask = np.ones([self.num_envs, self.num_agents], dtype=np.bool_)
            self.buf_terminals = np.zeros((self.num_envs, self.num_agents), dtype=np.bool_)
            self.buf_truncations = np.zeros((self.num_envs, self.num_agents), dtype=np.bool_)
            self.buf_rews = np.zeros((self.num_envs,) + self.rew_shape, dtype=np.float32)
            self.buf_info = [{} for _ in range(self.num_envs)]

            self.max_episode_length = env_info["episode_limit"]
            self.actions = None

        def reset(self):
            for i_env, env in enumerate(self.envs):
                obs, infos = env.reset()
                self.buf_obs[i_env], self.buf_info[i_env] = np.array(obs), list(infos)
            self.buf_done = np.zeros((self.num_envs,), dtype=np.bool_)
            return self.buf_obs.copy(), self.buf_info.copy()

        def step_wait(self):
            if not self.waiting:
                raise NotSteppingError
            for e in range(self.num_envs):
                action = self.actions[e]
                obs, rew, done, truncated, infos = self.envs[e].step(action)
                self.buf_obs[e] = obs
                self.buf_rews[e] = rew
                self.buf_terminals[e] = done
                self.buf_truncations[e] = truncated
                self.buf_info[e] = infos
                self.buf_info[e]["individual_episode_rewards"] = infos["episode_score"]
                if all(done) or all(truncated):
                    obs_reset, _ = self.envs[e].reset()
                    self.buf_info[e]["reset_obs"] = obs_reset
                    self.buf_info[e]["reset_agent_mask"] = self.envs[e].get_agent_mask()
                    self.buf_info[e]["reset_state"] = self.envs[e].state()
            self.waiting = False
            return self.buf_obs.copy(), self.buf_rews.copy(), self.buf_terminals.copy(), self.buf_truncations.copy(), self.buf_info.copy()

        def global_state(self):
            for e in range(self.num_envs):
                self.buf_state[e] = self.envs[e].state()
            return self.buf_state

        def agent_mask(self):
            for e in range(self.num_envs):
                self.buf_agent_mask[e] = self.envs[e].get_agent_mask()
            return self.buf_agent_mask


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
        "Dummy_NewEnv_MAS": DummyVecEnv_New_MAS,  # Add the newly defined vectorized environment for multi-agent systems

        # multiprocess #
        "Subproc_Gym": SubprocVecEnv_Gym,
        "Subproc_Pettingzoo": SubprocVecEnv_Pettingzoo,
        "Subproc_StarCraft2": SubprocVecEnv_StarCraft2,
        "Subproc_Football": SubprocVecEnv_GFootball,
        "Subproc_Atari": SubprocVecEnv_Atari,
        "Subproc_NewEnv": SubprocVecEnv_New,  # Add the newly defined vectorized environment
        "Subproc_NewEnv_MAS": SubprocVecEnv_New_MAS,  # Add the newly defined vectorized environment for multi-agent systems
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

        elif config.env_name == "NewEnv_MAS":  # Add the newly defined vectorized environment
            from xuance.environment.new_env_mas.new_env_mas import New_Env_MAS
            env = New_Env_MAS(config.env_id, config.seed)

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

Let's take MADDPG for example, you need to prepare a config file, such as "xuance/configs/maddpg/new_env_mas.yaml".
Finally, you can run the method with new environment by the following commands:

.. code-block:: python

    import argparse
    from xuance import get_runner


    def parse_args():
        parser = argparse.ArgumentParser("Run a demo.")
        parser.add_argument("--method", type=str, default="maddpg")
        parser.add_argument("--env", type=str, default="new_env_mas")
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

