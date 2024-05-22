import numpy as np

from multiagent.environment import MultiAgentMPNNEnv
import multiagent.custom_scenarios as scenarios

# import gym_vecenv # instead of this we just copied the relevant
# code from gym_vecenv in mpnn_baseline.nav.env_utils folder
# from multiagent.env_wrappers import DummyVecEnv, SubprocVecEnv
# from onpolicy.envs.env_wrappers import SubprocVecEnv, DummyVecEnv
from baselines.mpnn.nav.env_utils.env_wrappers import SubprocVecEnv, DummyVecEnv
from baselines.mpnn.nav.env_utils.vec_normalize import MultiAgentVecNormalize


def normalize_obs(obs, mean, std):
    if mean is not None:
        return np.divide((obs - mean), std)
    else:
        return obs


def make_env(args, rank):
    def _thunk():
        env = make_multiagent_env(args)
        env.seed(args.seed + rank)  # seed not implemented
        return env

    return _thunk


def make_multiagent_env(args):
    scenario = scenarios.load(args.scenario_name + ".py").Scenario()
    world = scenario.make_world(args=args)

    env = MultiAgentMPNNEnv(
        world=world,
        reset_callback=scenario.reset_world,
        reward_callback=scenario.reward,
        observation_callback=scenario.observation,
        info_callback=scenario.info_callback
        if hasattr(scenario, "info_callback")
        else None,
        scenario_name=args.scenario_name,
        discrete_action=True,
    )
    return env


def make_parallel_envs(args):
    # make parallel environments
    envs = [make_env(args, i) for i in range(args.n_rollout_threads)]
    if args.n_rollout_threads > 1:
        envs = SubprocVecEnv(envs)
    else:
        envs = DummyVecEnv(envs)

    envs = MultiAgentVecNormalize(envs, ob=False, ret=True)
    return envs


def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module


def rearrange_acts(envs, agent_actions):
    # rearrange action according to type of action space
    if envs.action_space[0].__class__.__name__ == "MultiDiscrete":
        for i in range(envs.action_space[0].shape):
            uc_actions_env = np.eye(envs.action_space[0].high[i] + 1)[
                agent_actions[:, :, i]
            ]
            if i == 0:
                agent_actions = uc_actions_env
            else:
                agent_actions = np.concatenate((agent_actions, uc_actions_env), axis=2)
    elif envs.action_space[0].__class__.__name__ == "Discrete":
        agent_actions = np.squeeze(np.eye(envs.action_space[0].n)[agent_actions], 2)
    else:
        raise NotImplementedError
    return agent_actions
