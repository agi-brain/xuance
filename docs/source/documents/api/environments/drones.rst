Drones
==============================================

.. image:: ../../../figures/drones/helix.gif
    :height: 150px
.. image:: ../../../figures/drones/rl.gif
    :height: 150px
.. image:: ../../../figures/drones/marl.gif
    :height: 150px

.. raw:: html

    <br><hr>

This environment is forked from the gym-pybullet-drones,
which is a gym environment with pybullet physics for reinforcement learning of multi-agent quadcopter control.
It supports both single and multiple drones control.
According to the official repository, it provides the following five kinds of action types:

| rpm: rounds per minutes (RPMs);
| pid: PID control;
| vel: Velocity input (using PID control);
| one_d_rpm: 1D (identical input to all motors) with RPMs;
| one_d_pid: 1D (identical input to all motors) with PID control.

You also have permission to customize the scenarios and tasks in this environment for your needs.

| **Official link**: `https://github.com/utiasDSL/gym-pybullet-drones.git <https://github.com/utiasDSL/gym-pybullet-drones.git>`_
| **Paper link**: `https://arxiv.org/pdf/2103.02142.pdf <https://arxiv.org/pdf/2103.02142.pdf>`_

.. raw:: html

    <br><hr>

Installation
-------------------------------------------------

.. tip::

    Before preparing the software packages for this simulator, it is recommended to create a new conda environment with **Python 3.10**.

Open terminal and type the following commands, then a new conda environment for xuance with drones could be built:

.. code-block:: bash

    conda create -n xuance_drones python=3.10
    conda activate xuance_drones
    pip install xuance  # refer to the installation of XuanCe.

    git clone https://github.com/utiasDSL/gym-pybullet-drones.git
    cd gym-pybullet-drones/
    pip install --upgrade pip
    pip install -e .  # if needed, `sudo apt install build-essential` to install `gcc` and build `pybullet`

During the installation of gym-pybullet-drones, you might encounter the errors like:

.. error::

    | gym-pybullet-drones 2.0.0 requires numpy<2.0,>1.24, but you have numpy 1.22.4 which is incompatible.
    | gym-pybullet-drones 2.0.0 requires scipy<2.0,>1.10, but you have scipy 1.7.3 which is incompatible.

**Solution**: Upgrade the above incompatible packages.

.. code-block:: bash

    pip install numpy==1.24.0
    pip install scipy==1.12.0

.. raw:: html

    <br><hr>

Try an Example
-------------------------------------------------

Create a python file named, e.g., "demo_drones.py".

.. code-block:: python

    import argparse
    from xuance import get_runner

    def parse_args():
        parser = argparse.ArgumentParser("Run a demo.")
        parser.add_argument("--method", type=str, default="iddpg")
        parser.add_argument("--env", type=str, default="drones")
        parser.add_argument("--env-id", type=str, default="MultiHoverAviary")
        parser.add_argument("--test", type=int, default=0)
        parser.add_argument("--device", type=str, default="cuda:0")
        parser.add_argument("--parallels", type=int, default=10)
        parser.add_argument("--benchmark", type=int, default=1)
        parser.add_argument("--test-episode", type=int, default=5)

        return parser.parse_args()

    if __name__ == '__main__':
        parser = parse_args()
        runner = get_runner(method=parser.method,
                            env=parser.env,
                            env_id=parser.env_id,
                            parser_args=parser,
                            is_test=parser.test)
        if parser.benchmark:
            runner.benchmark()
        else:
            runner.run()

Open the terminal and type the python command:

.. code-block:: bash

    python demo_drones.py

| Then, you can brew a cup of coffee, and wait for the training process to finish.
| Finally, test the trained model and view the effectiveness.

.. code-block:: bash

    python demo_drones.py --benchmark 0 --test 1


.. raw:: html

    <br><hr>

drones_env.py
-------------------------------------------------

.. py:class::
    xuance.environment.drones.drones_env.Drones_Env(args)

    This is a wrapper class for a Drones_Env environment.

    :param args: An argument object that contains various settings and parameters for initializing the environment.
    :type args: SimpleNamespace
    :param args.continuous: Determines whether the drone operates in a continuous control mode.
    :type args.continuous: bool
    :param args.env_id: Specifies the type of PyBullet Drones environment to instantiate.
    :type args.env_id: str
    :param args.render: Determines whether to render the environment with a graphical interface.
    :type args.render: bool
    :param args.record: Determines whether to record the environment's visual output.
    :type args.record: bool
    :param args.max_episode_steps: Maximum number of steps per episode for the environment.
    :type args.max_episode_steps: int

.. py:function::
    xuance.environment.drones.drones_env.Drones_Env.space_reshape(gym_space)

    Reshape the given Gym space into a new Box space with flattened boundaries.

    :param gym_space: The Gym space that needs to be reshaped.
    :type gym_space: gym.spaces.Space
    :return: A reshaped Box space with flattened boundaries.
    :rtype: gym.spaces.Box

.. py:function::
    xuance.environment.drones.drones_env.Drones_Env.close()

    Close the environment.

.. py:function::
    xuance.environment.drones.drones_env.Drones_Env.render()

    Return the rendering result.

    :return: the rendering result.
    :rtype: np.ndarray

.. py:function::
    xuance.environment.drones.drones_env.Drones_Env.reset()

    Reset the environment.

    :return: The initial observation of the environment as a flattened 1-dimensional array and
             additional information regarding the environment's state.
    :rtype: tuple

.. py:function::
    xuance.environment.drones.drones_env.Drones_Env.step(actions)

    Execute the actions and get next observations, rewards, and other information.

    :param actions: Actions to be executed in the environment. The actions are reshaped to be compatible with the environment's expectations.
    :type actions: np.ndarray
    :return: A tuple containing the flattened initial observation of the environment, the received reward,
             a termination indicator, a truncation indicator, and additional environment-related information.
    :rtype: tuple

.. raw:: html

    <br><hr>

drones_vec__env.py
-------------------------------------------------

.. py:class::
    xuance.environment.drones.drones_vec_env.SubprocVecEnv_Drones(env_fns, context='spawn', in_series=1)

    Extend the functionality of a subprocess-based vectorized environment.

    :param env_fns: environment function.
    :param context:  the method used for creating and managing processes in a multiprocessing environment.
    :param in_series: specifies the number of environments to run in series.
    :type in_series: int

.. py:class::
    xuance.environment.drones.drones_vec_env.DummyVecEnv_Drones(env_fns)

    Extends the functionality of a dummy vectorized environment

    :param env_fns: environment function.

.. raw:: html

    <br><hr>

Source Code
------------------------------------------------

.. tabs::

    .. group-tab:: drones_env.py

        .. code-block:: python

            import numpy as np
            from gym.spaces import Box
            import time


            class Drones_Env:
                def __init__(self, args):
                    # import scenarios of gym-pybullet-drones
                    self.env_id = args.env_id
                    from gym_pybullet_drones.envs.CtrlAviary import CtrlAviary
                    from xuance.environment.drones.customized.HoverAviary import HoverAviary
                    from gym_pybullet_drones.envs.VelocityAviary import VelocityAviary
                    from xuance.environment.drones.customized.MultiHoverAviary import MultiHoverAviary
                    from gym_pybullet_drones.utils.enums import ObservationType, ActionType
                    REGISTRY = {
                        "CtrlAviary": CtrlAviary,
                        "HoverAviary": HoverAviary,
                        "VelocityAviary": VelocityAviary,
                        "MultiHoverAviary": MultiHoverAviary,
                        # you can add your customized scenarios here.
                    }
                    self.gui = args.render  # Note: You cannot render multiple environments in parallel.
                    self.sleep = args.sleep
                    self.env_id = args.env_id

                    kwargs_env = {'gui': self.gui}
                    if self.env_id in ["HoverAviary", "MultiHoverAviary"]:
                        kwargs_env.update({'obs': ObservationType(args.obs_type),
                                           'act': ActionType(args.act_type)})
                    if self.env_id != "HoverAviary":
                        kwargs_env.update({'num_drones': args.num_drones})
                    self.env = REGISTRY[args.env_id](**kwargs_env)

                    self._episode_step = 0
                    self._episode_score = 0.0
                    if self.env_id == "MultiHoverAviary":
                        self.observation_space = self.env.observation_space
                        self.observation_shape = self.env.observation_space.shape
                        self.action_space = self.env.action_space
                        self.action_shape = self.env.action_space.shape
                    else:
                        self.observation_space = self.space_reshape(self.env.observation_space)
                        self.action_space = self.space_reshape(self.env.action_space)
                    self.max_episode_steps = self.max_cycles = args.max_episode_steps

                    self.n_agents = args.num_drones
                    self.env_info = {
                        "n_agents": self.n_agents,
                        "obs_shape": self.env.observation_space.shape,
                        "act_space": self.action_space,
                        "state_shape": 20,
                        "n_actions": self.env.action_space.shape[-1],
                        "episode_limit": self.max_episode_steps,
                    }

                def space_reshape(self, gym_space):
                    low = gym_space.low.reshape(-1)
                    high = gym_space.high.reshape(-1)
                    shape_obs = (gym_space.shape[-1], )
                    return Box(low=low, high=high, shape=shape_obs, dtype=gym_space.dtype)

                def close(self):
                    self.env.close()

                def render(self, *args, **kwargs):
                    return np.zeros([2, 2, 2])

                def reset(self):
                    obs, info = self.env.reset()
                    info["episode_step"] = self._episode_step

                    self._episode_step = 0
                    if self.n_agents > 1:
                        self._episode_score = np.zeros([self.n_agents, 1])
                        obs_return = obs
                    else:
                        self._episode_score = 0.0
                        obs_return = obs.reshape(-1)
                    return obs_return, info

                def step(self, actions):
                    if self.n_agents > 1:
                        obs, reward, terminated, truncated, info = self.env.step(actions)
                        obs_return = obs
                        terminated = [terminated for _ in range(self.n_agents)]
                    else:
                        obs, reward, terminated, truncated, info = self.env.step(actions.reshape([1, -1]))
                        obs_return = obs.reshape(-1)

                    self._episode_step += 1
                    self._episode_score += reward
                    if self.n_agents > 1:
                        truncated = [True for _ in range(self.n_agents)] if (self._episode_step >= self.max_episode_steps) else [False for _ in range(self.n_agents)]
                    else:
                        truncated = True if (self._episode_step >= self.max_episode_steps) else False
                    info["episode_step"] = self._episode_step  # current episode step
                    info["episode_score"] = self._episode_score  # the accumulated rewards

                    if self.gui:
                        time.sleep(self.sleep)

                    return obs_return, reward, terminated, truncated, info

                def get_agent_mask(self):
                    return np.ones(self.n_agents, dtype=np.bool_)  # 1 means available

                def state(self):
                    return np.zeros([20])



    .. group-tab:: drones_vec_env.py

        .. code-block:: python

            from xuance.environment.vector_envs.vector_env import VecEnv, AlreadySteppingError, NotSteppingError
            from xuance.common import space2shape, combined_shape
            from gym.spaces import Dict
            import numpy as np
            import multiprocessing as mp
            from xuance.environment.vector_envs.subproc_vec_env import clear_mpi_env_vars, flatten_list, CloudpickleWrapper
            from xuance.environment.gym.gym_vec_env import SubprocVecEnv_Gym, DummyVecEnv_Gym, worker


            class SubprocVecEnv_Drones(SubprocVecEnv_Gym):
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
                    self.waiting = False
                    self.closed = False
                    self.in_series = in_series
                    num_envs = len(env_fns)
                    assert num_envs % in_series == 0, "Number of envs must be divisible by number of envs to run in series"
                    self.n_remotes = num_envs // in_series
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

                    self.remotes[0].send(('get_spaces', None))
                    observation_space, action_space = self.remotes[0].recv().x
                    VecEnv.__init__(self, len(env_fns), observation_space, action_space)

                    self.obs_shape = space2shape(self.observation_space)
                    if isinstance(self.observation_space, Dict):
                        self.buf_obs = {k: np.zeros(combined_shape(self.num_envs, v)) for k, v in
                                        zip(self.obs_shape.keys(), self.obs_shape.values())}
                    else:
                        self.buf_obs = np.zeros(combined_shape(self.num_envs, self.obs_shape), dtype=np.float32)
                    self.buf_dones = np.zeros((self.num_envs,), dtype=np.bool_)
                    self.buf_trunctions = np.zeros((self.num_envs,), dtype=np.bool_)
                    self.buf_rews = np.zeros((self.num_envs,), dtype=np.float32)
                    self.buf_infos = [{} for _ in range(self.num_envs)]
                    self.actions = None
                    self.remotes[0].send(('get_max_cycles', None))
                    self.max_episode_length = self.remotes[0].recv().x


            class DummyVecEnv_Drones(DummyVecEnv_Gym):
                """
                VecEnv that does runs multiple environments sequentially, that is,
                the step and reset commands are send to one environment at a time.
                Useful when debugging and when num_env == 1 (in the latter case,
                avoids communication overhead)
                """
                def __init__(self, env_fns):
                    self.waiting = False
                    self.closed = False
                    self.envs = [fn() for fn in env_fns]
                    env = self.envs[0]
                    VecEnv.__init__(self, len(env_fns), env.observation_space, env.action_space)
                    self.obs_shape = space2shape(self.observation_space)
                    if isinstance(self.observation_space, Dict):
                        self.buf_obs = {k: np.zeros(combined_shape(self.num_envs, v)) for k, v in
                                        zip(self.obs_shape.keys(), self.obs_shape.values())}
                    else:
                        self.buf_obs = np.zeros(combined_shape(self.num_envs, self.obs_shape), dtype=np.float32)
                    self.buf_dones = np.zeros((self.num_envs,), dtype=np.bool_)
                    self.buf_trunctions = np.zeros((self.num_envs,), dtype=np.bool_)
                    self.buf_rews = np.zeros((self.num_envs,), dtype=np.float32)
                    self.buf_infos = [{} for _ in range(self.num_envs)]
                    self.actions = None
                    try:
                        self.max_episode_length = env.max_episode_steps
                    except AttributeError:
                        self.max_episode_length = 1000

    .. group-tab:: drones_vec_env.py

        .. code-block:: python

            from xuance.environment.vector_envs.vector_env import NotSteppingError
            from xuance.environment.gym.gym_vec_env import DummyVecEnv_Gym, SubprocVecEnv_Gym
            from xuance.common import combined_shape
            from gymnasium.spaces import Box
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


            class SubprocVecEnv_Drones_MAS(SubprocVecEnv_Gym):
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
                    self.dim_obs = env_info["obs_shape"][-1]
                    self.dim_act = self.n_actions = env_info["n_actions"]
                    self.dim_state = env_info["state_shape"]
                    observation_space, action_space = (self.dim_obs,), (self.dim_act,)
                    self.viewer = None
                    VecEnv.__init__(self, num_envs, observation_space, action_space)

                    self.num_agents = env_info["n_agents"]
                    self.obs_shape = env_info["obs_shape"]
                    self.act_shape = (self.num_agents, self.dim_act)
                    self.rew_shape = (self.num_agents, 1)
                    self.dim_reward = self.num_agents
                    self.action_space = env_info["act_space"]
                    self.state_space = Box(low=-np.inf, high=np.inf, shape=[self.dim_state, ], dtype=np.float32)

                    self.buf_obs = np.zeros(combined_shape(self.num_envs, self.obs_shape), dtype=np.float32)
                    self.buf_state = np.zeros(combined_shape(self.num_envs, self.dim_state), dtype=np.float32)
                    self.buf_agent_mask = np.ones([self.num_envs, self.num_agents], dtype=np.bool_)
                    self.buf_terminals = np.zeros((self.num_envs, self.num_agents), dtype=np.bool_)
                    self.buf_truncations = np.zeros((self.num_envs, self.num_agents), dtype=np.bool_)
                    self.buf_rews = np.zeros((self.num_envs,) + self.rew_shape, dtype=np.float32)
                    self.buf_infos = [{} for _ in range(self.num_envs)]

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


            class DummyVecEnv_Drones_MAS(DummyVecEnv_Gym):
                def __init__(self, env_fns):
                    self.waiting = False
                    self.envs = [fn() for fn in env_fns]
                    env = self.envs[0]
                    env_info = env.env_info
                    self.dim_obs = env_info["obs_shape"][-1]
                    self.dim_act = self.n_actions = env_info["n_actions"]
                    self.dim_state = env_info["state_shape"]
                    observation_space, action_space = (self.dim_obs,), (self.dim_act,)
                    self.viewer = None
                    VecEnv.__init__(self, len(env_fns), observation_space, action_space)

                    self.num_agents = env_info["n_agents"]
                    self.obs_shape = env_info["obs_shape"]
                    self.act_shape = (self.num_agents, self.dim_act)
                    self.rew_shape = (self.num_agents, 1)
                    self.dim_reward = self.num_agents
                    self.action_space = env_info["act_space"]
                    self.state_space = Box(low=-np.inf, high=np.inf, shape=[self.dim_state, ])

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

