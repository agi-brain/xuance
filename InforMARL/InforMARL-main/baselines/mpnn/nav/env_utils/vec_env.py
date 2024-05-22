# taken from MPNN author's repo gym_vecenv: https://github.com/agakshat/gym_vecenv
from abc import ABC, abstractmethod


class AlreadySteppingError(Exception):
    """
    Raised when an asynchronous step is running while
    step_async() is called again.
    """

    def __init__(self):
        msg = "already running an async step"
        Exception.__init__(self, msg)


class NotSteppingError(Exception):
    """
    Raised when an asynchronous step is not running but
    step_wait() is called.
    """

    def __init__(self):
        msg = "not running an async step"
        Exception.__init__(self, msg)


class VecEnv(ABC):
    """
    An abstract asynchronous, vectorized environment.
    Used to batch data from multiple copies of an environment, so that
    each observation becomes an batch of observations, and expected
    action is a batch of actions to be applied per-environment.
    """

    closed = False
    viewer = None

    metadata = {"render.modes": ["human", "rgb_array"]}

    def __init__(
        self, num_envs, observation_space, share_observation_space, action_space
    ):
        self.num_envs = num_envs
        self.observation_space = observation_space
        self.share_observation_space = share_observation_space
        self.action_space = action_space

    @abstractmethod
    def reset(self):
        """
        Reset all the environments and return an array of
        observations, or a tuple of observation arrays.

        If step_async is still doing work, that work will
        be cancelled and step_wait() should not be called
        until step_async() is invoked again.
        """
        pass

    @abstractmethod
    def step_async(self, actions):
        """
        Tell all the environments to start taking a step
        with the given actions.
        Call step_wait() to get the results of the step.

        You should not call this if a step_async run is
        already pending.
        """
        pass

    @abstractmethod
    def step_wait(self):
        """
        Wait for the step taken with step_async().

        Returns (obs, rews, dones, infos):
        - obs: an array of observations, or a tuple of
                arrays of observations.
        - rews: an array of rewards
        - dones: an array of "episode done" booleans
        - infos: a sequence of info objects
        """
        pass

    @abstractmethod
    def close(self):
        """
        Clean up the environments' resources.
        """
        pass

    def step(self, actions):
        """
        Step the environments synchronously.

        This is available for backwards compatibility.
        """
        self.step_async(actions)
        return self.step_wait()

    def render(self):
        pass


class VecEnvWrapper(VecEnv):
    def __init__(
        self,
        venv,
        observation_space=None,
        share_observation_space=None,
        action_space=None,
    ):
        self.venv = venv
        VecEnv.__init__(
            self,
            num_envs=venv.num_envs,
            observation_space=observation_space or venv.observation_space,
            share_observation_space=share_observation_space
            or venv.share_observation_space,
            action_space=action_space or venv.action_space,
        )

    def step_async(self, actions):
        self.venv.step_async(actions)

    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def step_wait(self):
        pass

    def close(self):
        return self.venv.close()

    def render(self):
        self.venv.render()


class CloudpickleWrapper(object):
    """
    Uses cloudpickle to serialize contents
    (otherwise multiprocessing tries to use pickle)
    """

    def __init__(self, x):
        self.x = x

    def __getstate__(self):
        import cloudpickle

        return cloudpickle.dumps(self.x)

    def __setstate__(self, ob):
        import pickle

        self.x = pickle.loads(ob)
