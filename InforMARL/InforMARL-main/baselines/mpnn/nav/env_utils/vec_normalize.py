from .vec_env import VecEnvWrapper
from .running_mean_std import RunningMeanStd
import numpy as np


class VecNormalize(VecEnvWrapper):
    """
    Vectorized environment base class
    """

    def __init__(
        self,
        venv,
        ob=True,
        ret=True,
        clipob=10.0,
        cliprew=10.0,
        gamma=0.99,
        epsilon=1e-8,
    ):
        VecEnvWrapper.__init__(self, venv)
        self.ob_rms = RunningMeanStd(shape=self.observation_space.shape) if ob else None
        self.ret_rms = RunningMeanStd(shape=()) if ret else None
        self.clipob = clipob
        self.cliprew = cliprew
        self.ret = np.zeros(self.num_envs)
        self.gamma = gamma
        self.epsilon = epsilon

    def step_wait(self):
        """
        Apply sequence of actions to sequence of environments
        actions -> (observations, rewards, news)

        where 'news' is a boolean vector indicating whether each element is new.
        """
        obs, rews, news, infos = self.venv.step_wait()
        self.ret = self.ret * self.gamma + rews
        obs = self._obfilt(obs)
        if self.ret_rms:
            self.ret_rms.update(self.ret)
            rews = np.clip(
                rews / np.sqrt(self.ret_rms.var + self.epsilon),
                -self.cliprew,
                self.cliprew,
            )
        return obs, rews, news, infos

    def _obfilt(self, obs):
        if self.ob_rms:
            self.ob_rms.update(obs)
            obs = np.clip(
                (obs - self.ob_rms.mean) / np.sqrt(self.ob_rms.var + self.epsilon),
                -self.clipob,
                self.clipob,
            )
            return obs
        else:
            return obs

    def reset(self):
        """
        Reset all environments
        """
        obs = self.venv.reset()
        return self._obfilt(obs)


class MultiAgentVecNormalize(VecEnvWrapper):
    """
    Vectorized environment base class
    """

    def __init__(
        self,
        venv,
        ob=True,
        ret=True,
        clipob=10.0,
        cliprew=10.0,
        gamma=0.99,
        epsilon=1e-8,
    ):
        VecEnvWrapper.__init__(self, venv)
        self.ob_rms = (
            [RunningMeanStd(shape=x.shape) for x in self.observation_space]
            if ob
            else None
        )
        self.ret_rms = (
            RunningMeanStd(shape=(len(self.observation_space),)) if ret else None
        )
        self.clipob = clipob
        self.cliprew = cliprew
        self.ret = np.zeros((self.num_envs, len(self.observation_space)))
        self.gamma = gamma
        self.epsilon = epsilon
        self.n = len(self.observation_space)

    def step_wait(self):
        """
        Apply sequence of actions to sequence of environments
        actions -> (observations, rewards, news)

        where 'news' is a boolean vector indicating whether each element is new.
        """
        obs, rews, news, infos = self.venv.step_wait()
        orig_rews = (
            rews.copy()
        )  # keep original rewards in info dict for comparing same stuff as ours
        self.ret = self.ret * self.gamma + rews
        obs = self._obfilt(obs)
        if self.ret_rms:
            self.ret_rms.update(self.ret)
            rews = np.clip(
                rews / np.sqrt(self.ret_rms.var + self.epsilon),
                -self.cliprew,
                self.cliprew,
            )
        return obs, rews, news, infos, orig_rews

    def _obfilt(self, obs):
        if self.ob_rms:
            obs = np.reshape(obs, (self.num_envs, self.n, -1))
            for j in range(self.n):
                self.ob_rms[j].update(obs[:, j])
                t = np.clip(
                    (np.array(list(obs[:, j]), dtype=np.float) - self.ob_rms[j].mean)
                    / np.sqrt(self.ob_rms[j].var + self.epsilon),
                    -self.clipob,
                    self.clipob,
                )
                t = np.reshape(t, (self.num_envs, -1))
                for k in range(t.shape[0]):
                    obs[:, j][k] = t[k]
            return obs
        else:
            return obs

    def reset(self):
        """
        Reset all environments
        """
        obs = self.venv.reset()
        return self._obfilt(obs)
