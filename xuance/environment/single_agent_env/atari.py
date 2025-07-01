import gymnasium as gym
import numpy as np
from collections import deque
from xuance.environment.single_agent_env.gym import LazyFrames
try:
    import cv2
except ImportError:
    pass

try:
    import ale_py
    gym.register_envs(ale_py)
except ImportError:
    pass


class Atari_Env(gym.Wrapper):
    """Modified Atari environment with training optimizations.

    Implements several enhancements from DeepMind's Nature paper:
    - Episode termination: Treats end-of-life as end-of-episode (resets only on true game over)
    - Frame skipping: Returns only every `skip`-th frame
    - Observation resizing: Warps frames from 210x160 to configurable size
    - Frame stacking: Efficiently stacks last k frames using lazy arrays

    Note:
        All configurations should be set in the provided config object.
    """
    def __init__(self, config, **kwargs):
        """Initializes the Atari environment wrapper.

        Args:
            config: Configuration object containing:
                env_id: Atari environment ID (e.g., "ALE/Breakout-v5")
                env_seed: Random seed for environment
                obs_type: Observation type ("ram"/"rgb"/"grayscale")
                frame_skip: Frame skip interval (int or tuple for stochastic skipping)
                num_stack: Number of frames to stack
                img_size: Target observation dimensions (default: [210, 160])
                noop_max: Maximum no-op actions during reset
                render_mode: Rendering mode (None/"human"/"rgb_array")
                full_action_space: Whether to use full action space
            **kwargs: Additional arguments passed to gym.make()
        """
        full_action_space = config.full_action_space if hasattr(config, 'full_action_space') else False
        self.env = gym.make(config.env_id,
                            render_mode=config.render_mode,
                            obs_type=config.obs_type,
                            frameskip=config.frame_skip,
                            full_action_space=full_action_space)
        self.env.metadata['render_fps'] = config.fps
        self.env.action_space.seed(seed=config.env_seed)
        self.env.reset(seed=config.env_seed)
        self.max_episode_steps = self.env._max_episode_steps if hasattr(self.env, '_max_episode_steps') else 1e5
        super(Atari_Env, self).__init__(self.env)
        # self.env.seed(seed)
        self.num_stack = config.num_stack
        self.obs_type = config.obs_type
        self.frames = deque([], maxlen=self.num_stack)
        self.image_size = [210, 160] if config.img_size is None else config.img_size
        self.noop_max = config.noop_max
        self.lifes = self.env.unwrapped.ale.lives()
        self.was_real_done = True
        self.grayscale, self.rgb = False, False
        if self.obs_type == "rgb":
            self.rgb = True
            self.observation_space = gym.spaces.Box(low=0, high=255,
                                                    shape=(config.img_size[0], config.img_size[1], 3 * self.num_stack),
                                                    dtype=np.uint8)
        elif self.obs_type == "grayscale":
            self.grayscale = True
            self.observation_space = gym.spaces.Box(low=0, high=255,
                                                    shape=(config.img_size[0], config.img_size[1], self.num_stack),
                                                    dtype=np.uint8)
        else:  # ram type
            self.observation_space = self.env.observation_space
        # assert self.env.unwrapped.get_action_meanings()[0] == "NOOP"
        # assert self.env.unwrapped.get_action_meanings()[1] == "FIRE"
        # assert len(self.env.unwrapped.get_action_meanings()) >= 3
        self.action_space = self.env.action_space
        self.metadata = self.env.metadata
        self._render_mode = config.render_mode
        self._episode_step = 0
        self._episode_score = 0.0

    def close(self):
        """Closes the underlying environment and releases resources."""
        self.env.close()

    def render(self, *args, **kwargs):
        """Renders the environment.

        Returns:
            Rendered frame according to specified mode
        """
        return self.env.render()

    def reset(self, *args):
        """Resets the environment with random no-op actions.

        Performs:
        1. Environment reset
        2. Random number of no-op actions
        3. Initial fire action (if available)
        4. Frame stacking initialization

        Returns:
            tuple: (stacked_observations, info_dict)
        """
        info = {}
        if self.was_real_done:
            self.env.reset()
            # Execute NoOp actions
            num_noops = np.random.randint(0, self.noop_max)
            for _ in range(num_noops):
                obs, _, done, _, _ = self.env.step(0)
                if done:
                    self.env.reset()
            # try to fire
            obs, _, done, _, _ = self.env.step(1)
            if done:
                obs = self.env.reset()
            # stack reset observations
            for _ in range(self.num_stack):
                self.frames.append(self.observation(obs))

            self._episode_step = 0
            self._episode_score = 0.0
            info["episode_step"] = 0
        else:
            obs, _, done, _, _ = self.env.step(0)
            for _ in range(self.num_stack):
                self.frames.append(self.observation(obs))

        self.lifes = self.env.unwrapped.ale.lives()
        self.was_real_done = False
        return self._get_obs(), info

    def step(self, actions):
        """Executes one environment step.

        Args:
            actions: Action to execute

        Returns:
            tuple: (observation, reward, terminated, truncated, info)
        """
        observation, reward, terminated, truncated, info = self.env.step(actions)
        self.frames.append(self.observation(observation))
        lives = self.env.unwrapped.ale.lives()
        # avoid environment bug
        if self._episode_step >= self.max_episode_steps:
            terminated = True
        self.was_real_done = terminated
        if (lives < self.lifes) and (lives > 0):
            terminated = True
        truncated = self.was_real_done
        self.lifes = lives
        self._episode_step += 1
        self._episode_score += reward
        info["episode_score"] = self._episode_score
        info["episode_step"] = self._episode_step
        return self._get_obs(), self.reward(reward), terminated, truncated, info

    def _get_obs(self):
        """Returns stacked observations as LazyFrames.

        Returns:
            LazyFrames: Stacked frame observations

        Raises:
            AssertionError: If frame stack is incomplete
        """
        assert len(self.frames) == self.num_stack
        return LazyFrames(list(self.frames))

    def observation(self, frame):
        """Processes raw frame into desired observation format.

        Args:
            frame: Raw environment frame

        Returns:
            Processed observation (resized grayscale/RGB or raw RAM)
        """
        if self.grayscale:
            return np.expand_dims(cv2.resize(frame, self.image_size, interpolation=cv2.INTER_AREA), -1)
        elif self.rgb:
            return cv2.resize(frame, self.image_size, interpolation=cv2.INTER_AREA)
        else:
            return frame

    def reward(self, reward):
        """Applies reward shaping for training.

        Args:
            reward: Original environment reward

        Returns:
            Shaped reward (sign function in this implementation)
        """
        return np.sign(reward)
