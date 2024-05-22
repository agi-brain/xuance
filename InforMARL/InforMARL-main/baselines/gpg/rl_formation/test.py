import gym
import gym_flock
import numpy as np
import pdb
import dgl
import torch.nn as nn
import torch.nn.functional as F
import torch as t
from make_g import build_graph
import torch.optim as optim
import dgl
import dgl.function as fn
import math
import pdb

from torch.autograd import Variable
from torch.distributions import Categorical
import torch
import networkx as nx
import pdb
import matplotlib.pyplot as plt
from baselines.gpg.rl_formation.policy import Net
from baselines.gpg.rl_formation.make_g import build_graph
from baselines.gpg.rl_formation.utils import *
from utils import *

import os
import datetime

policy = Net()
optimizer = optim.Adam(policy.parameters(), lr=1e-3)
env = gym.make("FormationFlying-v3")
filename = "./logs/3agents.pt"
policy.load_state_dict(torch.load(filename))

# pdb.set_trace()


def main():
    test_episodes = 5
    running_reward = 10
    plotting_rew = []

    for episode in range(test_episodes):
        reward_over_eps = []
        state = env.reset()  # Reset environment and record the starting state
        g = build_graph(env)
        done = False

        for time in range(200):
            # if episode%50==0:
            env.render()
            # g = build_graph(env)
            action = select_action(state, g, policy)

            action = action.numpy()
            action = np.reshape(action, [-1])

            # Step through environment using chosen action
            action = np.clip(action, -env.max_accel, env.max_accel)

            state, reward, done, _ = env.step(action)

            reward_over_eps.append(reward)
            # Save reward
            policy.reward_episode.append(reward)
            if done:
                break

        # Used to determine when the environment is solved.
        running_reward = (running_reward * 0.99) + (time * 0.01)

        if episode % 50 == 0:
            print(
                "Episode {}\tLast length: {:5d}\tAverage test reward: {:.2f}\tAverage reward over episode: {:.2f}".format(
                    episode, time, running_reward, np.mean(reward_over_eps)
                )
            )

        plotting_rew.append(np.mean(reward_over_eps))

    np.savetxt(
        "Test_Relative_Goal_Reaching_for_%d_agents_rs_rg.txt" % (env.n_agents),
        plotting_rew,
    )
    fig = plt.figure()
    x = np.linspace(0, len(plotting_rew), len(plotting_rew))
    plt.plot(x, plotting_rew)
    plt.savefig("Test_Relative_Goal_Reaching_for_%d_agents_rs_rg.png" % (env.n_agents))
    plt.show()


if __name__ == "__main__":
    main()
