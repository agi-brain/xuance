import numpy as np
from baselines.mpnn.mpe.mape.multiagent.environment import MultiAgentEnv
import baselines.mpnn.mpe.mape.multiagent.scenarios as scenarios
import gym_vecenv


def normalize_obs(obs, mean, std):
    if mean is not None:
        return np.divide((obs - mean), std)
    else:
        return obs


def make_env(env_id, seed, rank, num_agents, dist_threshold, arena_size, identity_size):
    def _thunk():
        env = make_multiagent_env(
            env_id, num_agents, dist_threshold, arena_size, identity_size
        )
        env.seed(seed + rank)  # seed not implemented
        return env

    return _thunk


def make_multiagent_env(env_id, num_agents, dist_threshold, arena_size, identity_size):
    scenario = scenarios.load(env_id + ".py").Scenario(
        num_agents=num_agents,
        dist_threshold=dist_threshold,
        arena_size=arena_size,
        identity_size=identity_size,
    )
    world = scenario.make_world()

    env = MultiAgentEnv(
        world=world,
        reset_callback=scenario.reset_world,
        reward_callback=scenario.reward,
        observation_callback=scenario.observation,
        info_callback=scenario.info if hasattr(scenario, "info") else None,
        discrete_action=True,
        done_callback=scenario.done,
        cam_range=arena_size,
    )
    return env


def make_parallel_envs(args):
    # make parallel environments
    envs = [
        make_env(
            args.env_name,
            args.seed,
            i,
            args.num_agents,
            args.dist_threshold,
            args.arena_size,
            args.identity_size,
        )
        for i in range(args.num_processes)
    ]
    if args.num_processes > 1:
        envs = gym_vecenv.SubprocVecEnv(envs)
    else:
        envs = gym_vecenv.DummyVecEnv(envs)

    envs = gym_vecenv.MultiAgentVecNormalize(envs, ob=False, ret=True)
    return envs


def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module
