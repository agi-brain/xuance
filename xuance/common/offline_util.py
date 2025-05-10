import d4rl
import numpy as np
import gymnasium as gym


def load_d4rl_dataset(dataset_name: str, max_episode_steps, obsnorm=False, rewnorm=True):
    # create environment
    env = gym.make(dataset_name)
    dataset = d4rl.qlearning_dataset(env)

    if obsnorm:
        state_mean, state_std = compute_mean_std(dataset["observations"], eps=1e-3)
        dataset["observations"] = normalize_states(
            dataset["observations"], state_mean, state_std
        )
        dataset["next_observations"] = normalize_states(
            dataset["next_observations"], state_mean, state_std
        )
    else:
        state_mean = 0.0
        state_std = 1.0

    if rewnorm:
        if any(s in dataset_name for s in ('halfcheetah', 'hopper', 'walker2d')):
            min_ret, max_ret = return_range(dataset, max_episode_steps)
            dataset['rewards'] /= (max_ret - min_ret)
            dataset['rewards'] *= max_episode_steps
        elif 'antmaze' in dataset_name:
            dataset['rewards'] -= 1.

    return dataset, state_mean, state_std


def compute_mean_std(states: np.ndarray, eps: float):
    mean = states.mean(0)
    std = states.std(0) + eps
    return mean, std


def normalize_states(states: np.ndarray, mean: np.ndarray, std: np.ndarray):
    return (states - mean) / std


def return_range(dataset, max_episode_steps):
    returns, lengths = [], []
    ep_ret, ep_len = 0., 0
    for r, d in zip(dataset['rewards'], dataset['terminals']):
        ep_ret += float(r)
        ep_len += 1
        if d or ep_len == max_episode_steps:
            returns.append(ep_ret)
            lengths.append(ep_len)
            ep_ret, ep_len = 0., 0
    # returns.append(ep_ret)    # incomplete trajectory
    lengths.append(ep_len)  # but still keep track of number of steps
    assert sum(lengths) == len(dataset['rewards'])
    return min(returns), max(returns)
