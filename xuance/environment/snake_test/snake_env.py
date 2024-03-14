import numpy as np
from gym import spaces
import time


class snake_env:
    def __init__(self, env_id: str, seed: int, *args, **kwargs):
        self.env_id = env_id
        self.remain_step = 10
        self.score = 0.0
        self._episode_score = 0.0
        self._episode_step = 0

        # self.observation_space = spaces.MultiDiscrete([100, 100, 100, 100, 100])  # 小人和苹果的坐标
        self.observation_space = spaces.Box(low=0, high=10, shape=[5,], dtype=np.int32)
        self.action_space = spaces.Discrete(4)
               
        self.apple_position = (0, 0)  # 随机生成苹果的位置
        self.person_position = (50, 50)  # 随机生成小人的位置

    def _generate_apple_position(self):
        return np.random.randint(0, 10, 2)

    def _generate_person_position(self):
        return np.random.randint(0, 10, 2)

    def close(self):
        pass

    # def render(self):
    #     pass
    def render(self, mode='human'):
        if mode == 'human':
            # print("Person: ", self.person_position)
            # print("Apple: ", self.apple_position)
            # print("Remain step: ", self.remain_step)
            print("Score: ", self.score)
        
        elif mode == 'rgb_array':
            pass

    def reset(self):
        self.person_position = self._generate_person_position()  # 重置小人的位置
        self.apple_position = self._generate_apple_position()  # 重置苹果的位置
        self.remain_step = 10  # 重置步数
        self.score = 0.0  # 重置得分
        obs = (self.person_position[0], self.person_position[1], self.apple_position[0], self.apple_position[1], self.remain_step)
        info = {}
        info["remain_step"] = self.remain_step
        return obs, info

    def step(self, action):
        reward = 0
        info = {}
        # print(action)
        
        if action == 0:  # 上
            self.person_position[1] += 1
        elif action == 1:  # 下
            self.person_position[1] -= 1
        elif action == 2:  # 左
            self.person_position[0] -= 1
        elif action == 3:  # 右
            self.person_position[0] += 1

        self.remain_step -= 1  # 每移动一步，步数减1
        # print("动了")

        # 检查是否吃到苹果
        if np.array_equal(self.person_position, self.apple_position):
            self.remain_step += 8  # 吃到苹果，步数加10
            self.score += 1  # 得分加1
            reward += 1
            self.apple_position = self._generate_apple_position()  # 生成新的苹果位置
            # print("吃了!")

        # 检查是否游戏结束
        terminated = False
        truncated = False
        if self.remain_step <= 0:
            terminated = True
            reward -= 50
            # print("累死!")
        elif np.any(self.person_position < 0) or np.any(self.person_position >= 100):
            terminated = True
            reward -= 50
            # print("撞死!")

        self._episode_step += 1
        self._episode_score += reward
        observation = (self.person_position[0], self.person_position[1], self.apple_position[0], self.apple_position[1], self.remain_step)
        info["episode_step"] = self._episode_step  # current episode step
        info["episode_score"] = self._episode_score  # the accumulated rewards
        # print(reward, self.remain_step)
        # time.sleep(0.5)  # 暂停0.5秒
        return observation, reward, terminated, truncated, info
