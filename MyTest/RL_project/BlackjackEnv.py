"""
This is an example of creating a new environment in XuanCe for multi-agent system.
This example
"""
from gymnasium.spaces import Box, Discrete
import numpy as np
import functools
from typing import List
import gymnasium

from pettingzoo import AECEnv
from pettingzoo.utils import agent_selector, wrappers


class BlackjackEnv:
    def __init__(self, env_id: str, seed: int, **kwargs):
        self.n_agents = 1
        self.dim_obs = 3  # dimension of one agent's observation

        self.dim_action = 2  # dimension of actions (continuous)
        # self.n_actions = 2  # number of discrete actions (discrete)

        self.observation_space = Box(low=0, high=30, shape=[self.dim_obs, ], dtype=np.int32, seed=seed)
        self.banker=0 #庄家初始牌
        self.player_initial1= 0 #玩家初始牌面1
        self.player_initial2 = 0  # 玩家初始牌面2
        self.player=0          # 玩家牌面
        self.player_Ace=0              #玩家是否有Ace
        self.action_labels=['twist','stick']
        if kwargs['continuous']:
            self.action_space = Box(low=0, high=1, shape=[self.dim_action, ], dtype=np.float32, seed=seed)
        else:
            self.action_space = Discrete(n=len(self.action_labels), seed=seed)

        self.episode_step = 0
        self.episode_score=0.0
        self.render_mode = kwargs['render_mode']
        self.max_episode_steps = 48 #一共52张牌，reset的时候给庄家和玩家各发2张还剩48
        self.env_info = {
            "n_agents": self.n_agents,
            "observation_space": self.observation_space,
            "action_space": self.action_space,
            "episode_limit": self.max_episode_steps,
        }


    def close(self):
        pass

    def render(self):
        if self.render_mode == 'human':
            print("Score: ", self.episode_score)

        elif self.render_mode == 'rgb_array':
            pass

    def reset(self):

        #庄家初始化
        self.banker=np.random.randint(1, 11)

        #玩家初始化
        self.player_initial1 = np.random.randint(1, 14)
        self.player_initial2 = np.random.randint(1, 14)

        self.player_initial1=10 if self.player_initial1>10 else self.player_initial1
        self.player_initial2=10 if self.player_initial2>10 else self.player_initial2

        if self.player_initial1==1 and self.player_initial2==1:
            self.player_Ace=1
            self.player=12
        elif (self.player_initial1==1 and self.player_initial2!=1 ) or (self.player_initial1!=1 and self.player_initial2==1):
            self.player_Ace=1
            self.player=self.player_initial1 if self.player_initial1!=1 else self.player_initial2
        else:
            self.player=self.player_initial1+self.player_initial2
            self.player_Ace=0

        obs = [self.banker, self.player, self.player_Ace]
        info = {}
        self.episode_step = 0
        self.episode_score = 0.0
        info["episode_step"] = self.episode_step
        info["episode_score"]= self.episode_score
        info['episode_action'] = 0
        return obs, info

    def step(self, actions):
        # Execute the actions and get next observations, rewards, and other information.
        terminated = False
        truncated = False
        reward=0
        #当玩家牌总数小于12时必须要牌
        if self.player<12:
            actions=0
        # if self.player_Ace==1 and self.player+1<12:
        #     actions=0
        if actions==0:#玩家要牌
            player_card=np.random.randint(1,14)
            player_card=10 if player_card>10 else player_card
            if player_card==1 and self.player_Ace==1:#玩家有一张A，又要了一张A
                if self.player+11+1<=21:
                    self.player+=11+1
                else:
                    self.player+=1+1

            elif (player_card!=1 and self.player_Ace ==1) or (player_card==1 and self.player_Ace == 0) : #玩家有一张A，和一张2-10
                if self.player+11+player_card<=21:
                    self.player+=11+player_card
                else:
                    self.player+=1+player_card
                self.player_Ace=1
            else:#玩家只有数牌，没有A
                self.player+=player_card

            if self.player>21:
                reward=-1
                terminated=truncated=True
        elif actions==1:#玩家停牌，现在看庄家的牌
            truncated = terminated = True
            banker_A= 0
            if self.banker==1:#如果一开始庄家亮的牌是A，则首先进行赋值出来
                banker_A = 1
                self.banker =0
            while self.banker<17:
                card=np.random.randint(1,14)
                card=10 if card>10 else card
                if card == 1 and banker_A == 1:  # 、庄家有一张A，又要了一张A
                    if self.banker + 11 + card <= 21:
                        self.banker += 11 + card
                    else:
                        self.banker += 1 + card

                elif (card != 1 and banker_A == 1) or (card == 1 and banker_A == 0):  # 庄家有一张A，和一张2-10
                    if self.banker + 11 + card <= 21:
                        self.banker += 11 + card
                    else:
                        self.banker += 1 + card
                else:
                    self.banker += card
            if self.player>self.banker or self.banker>21:
                reward=1
            elif self.player< self.banker:
                reward=-1
            else:
                reward=0


        obs= [self.banker, self.player, self.player_Ace]
        info ={}

        self.episode_step += 1
        self.episode_score += reward
        info["episode_step"] = self.episode_step  # current episode step
        info["episode_score"]= self.episode_score # the accumulated rewards
        info["episode_action"]=actions
        return obs, reward, terminated, truncated, info
