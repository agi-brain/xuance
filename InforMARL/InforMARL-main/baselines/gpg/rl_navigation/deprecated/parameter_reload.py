import numpy as np
import torch.optim as optim

import torch
import matplotlib.pyplot as plt

from policy import Net
from make_g import build_graph
from utils import *
import time

# first load old policy
policy = Net()
optimizer = optim.Adam(policy.parameters(), lr=1e-3)
env = gym.make("FormationFlyingInference-v3")
filename = "./logs/2d_3agents.pt"
policy.load_state_dict(torch.load(filename))


# now initialize new policy
policy2 = Net()


def main():
    test_episodes = 5
    running_reward = 10
    plotting_rew = []

    for episode in range(test_episodes):
        reward_over_eps = []
        state = env.reset()  # Reset environment and record the starting state
        g = build_graph(env)

        # need to do one dummy action through the policy to initialize it
        # because pytorch and its dynamic graph things.
        # action = select_action(state,g,policy2)
        # set_weights(policy2,policy)

        # pdb.set_trace()
        done = False
        count = 0
        # for time in range(500):
        while True:
            # if episode%50==0:
            env.render()
            if episode == 0 and count == 0:
                # pdb.set_trace()
                count += 1
                # time.sleep(15)
            # pdb.set_trace()
            g = build_graph(env)
            action = select_action(state, g, policy)

            action = action.numpy()
            action = np.reshape(action, [-1])

            # Step through environment using chosen action
            action = np.clip(action, -env.max_accel, env.max_accel)

            state, reward, done, _ = env.step(action)

            reward_over_eps.append(reward)
            # Save reward
            policy.reward_episode.append(reward)
            # if done:
            # 	break

        # Used to determine when the environment is solved.
        # running_reward = (running_reward * 0.99) + (time * 0.01)

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
