import gym

import os, sys

sys.path.append(os.path.abspath(os.getcwd()))

import baselines.gpg.gym_formation.gym_flock
import numpy as np
import torch.optim as optim

import torch
import matplotlib.pyplot as plt

from baselines.gpg.rl_formation.make_g import build_graph
from baselines.gpg.rl_formation.policy import Net
from baselines.gpg.rl_formation.make_g import build_graph
from baselines.gpg.rl_formation.utils import select_action, update_policy

import os
import datetime

policy = Net()
optimizer = optim.Adam(policy.parameters(), lr=1e-3)
env = gym.make("FormationFlying-v3")

if not os.path.exists("./logs"):
    os.makedirs("./logs")

filename = str(datetime.datetime.now()) + str(
    "_%dagents_fixed_fcnpolicy" % env.n_agents
)
filename = filename + str(".pt")
torch.save(policy.state_dict(), "./logs/%s" % filename)


def main(episodes):
    running_reward = 10
    plotting_rew = []

    for episode in range(episodes):
        reward_over_eps = []
        state = env.reset()  # Reset environment and record the starting state
        g = build_graph(env)
        done = False

        for time in range(200):
            # if episode%50==0:
            # 	env.render()
            # g = build_graph(env)
            action = select_action(state, g, policy)

            action = action.numpy()  # shape [num_agents, action_dim]
            action = np.reshape(action, [-1])

            # Step through environment using chosen action
            action = np.clip(action, -env.max_accel, env.max_accel)

            state, reward, done, _ = env.step(action)
            # state.shape = [num_agents, state_dim]

            reward_over_eps.append(reward)
            # Save reward
            policy.reward_episode.append(reward)
            if done:
                break

        # Used to determine when the environment is solved.
        running_reward = (running_reward * 0.99) + (time * 0.01)

        update_policy(policy, optimizer)

        if episode % 50 == 0:
            print(
                f"Episode {episode}\t "
                f"Last length: {time:5d}\t "
                f"Average running reward: {running_reward:.2f}\t "
                f"Average reward over episode: {np.mean(reward_over_eps):.2f}"
            )

        if episode % 5000 == 0:
            torch.save(policy.state_dict(), "./logs/%s" % filename)

        plotting_rew.append(np.mean(reward_over_eps))
    np.savetxt(
        "Relative_Goal_Reaching_for_%d_agents_rs_rg.txt" % (env.n_agents), plotting_rew
    )
    fig = plt.figure()
    x = np.linspace(0, len(plotting_rew), len(plotting_rew))
    plt.plot(x, plotting_rew)
    plt.savefig("Relative_Goal_Reaching_for_%d_agents_rs_rg.png" % (env.n_agents))
    plt.show()


episodes = 50000
main(episodes)
