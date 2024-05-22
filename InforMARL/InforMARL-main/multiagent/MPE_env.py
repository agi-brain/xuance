"""
    Code for creating a multiagent environment with one of the scenarios listed
    in ./scenarios/.
    Can be called by using, for example:
        env = MPEEnv('simple_speaker_listener')
    After producing the env object, can be used similarly to an OpenAI gym
    environment.

    A policy using this environment must output actions in the form of a list
    for all agents. Each element of the list should be a numpy array,
    of size (env.world.dim_p + env.world.dim_c, 1). Physical actions precede
    communication actions in this array. See environment.py for more details.
"""

import argparse
from typing import Dict
import numpy as np

from multiagent.custom_scenarios import load


def MPEEnv(args: argparse.Namespace):
    """
    Creates a MultiAgentEnv object as env. This can be used similar to a gym
    environment by calling env.reset() and env.step().
    Use env.render() to view the environment on the screen.

    Input:
        args.scenario_name  :   name of the scenario from ./scenarios/ to be
                            Returns (without the .py extension)
        benchmark       :   whether you want to produce benchmarking data
                        (usually only done during evaluation)

    Some useful env properties (see environment.py):
        .observation_space  :   Returns the observation space for each agent
        .action_space       :   Returns the action space for each agent
        .n                  :   Returns the number of Agents
    """

    # load scenario from script
    scenario = load(args.scenario_name + ".py").Scenario()
    # create world
    world = scenario.make_world(args=args)
    if args.algorithm_name in ["mappo", "rmappo"]:
        from multiagent.environment import MultiAgentPPOEnv as MultiAgentEnv
    else:
        from multiagent.environment import MultiAgentOffPolicyEnv as MultiAgentEnv
    env = MultiAgentEnv(
        world=world,
        reset_callback=scenario.reset_world,
        reward_callback=scenario.reward,
        observation_callback=scenario.observation,
        info_callback=scenario.info_callback
        if hasattr(scenario, "info_callback")
        else None,
        scenario_name=args.scenario_name,
    )
    return env


def GraphMPEEnv(args):
    """
    Same as MPEEnv but for graph environment
    """

    # load scenario from script
    assert "graph" in args.scenario_name, "Only use graph env for graph scenarios"
    scenario = load(args.scenario_name + ".py").Scenario()
    # create world
    world = scenario.make_world(args=args)
    from multiagent.environment import MultiAgentGraphEnv

    # create multiagent environment
    env = MultiAgentGraphEnv(
        world=world,
        reset_callback=scenario.reset_world,
        reward_callback=scenario.reward,
        observation_callback=scenario.observation,
        graph_observation_callback=scenario.graph_observation,
        update_graph=scenario.update_graph,
        id_callback=scenario.get_id,
        info_callback=scenario.info_callback,
        scenario_name=args.scenario_name,
    )

    return env


def GPGMPEEnv(args):
    """
    MPE env but compatible with the GPG baseline code
    """
    # load scenario from script
    scenario = load("navigation_gpg.py").Scenario()
    # create world
    world = scenario.make_world(args=args)
    from multiagent.environment import MultiAgentGPGEnv

    env = MultiAgentGPGEnv(
        world=world,
        reset_callback=scenario.reset_world,
        reward_callback=scenario.reward,
        observation_callback=scenario.observation,
        graph_observation_callback=scenario.graph_observation,
        info_callback=scenario.info_callback,
        id_callback=scenario.get_id,
        update_graph=scenario.update_graph,
        shared_viewer=False,
        scenario_name=args.scenario_name,
    )

    return env


def CADRLMPEEnv(args, phase):
    """
    MPE env but compatible with the CADRL baseline code
    """
    # load scenario from script
    scenario = load("navigation_cadrl.py").Scenario()
    # create world
    world = scenario.make_world(args=args)
    from multiagent.environment import MultiAgentCADRLEnv

    env = MultiAgentCADRLEnv(
        config_args=args,
        phase=phase,
        world=world,
        reset_callback=scenario.reset_world,
        reward_callback=scenario.reward,
        observation_callback=scenario.observation,
        info_callback=scenario.info_callback
        if hasattr(scenario, "info_callback")
        else None,
        shared_viewer=False,
        scenario_name=args.scenario_name,
    )

    return env


def AttentionMPEEnv(args):
    """
    MPE env compatible with the Attention baseline code
    """
    # load scenario from script
    scenario = load("navigation_attention.py").Scenario()
    # create world
    world = scenario.make_world(args=args)
    from multiagent.environment import MultiAgentAttentionEnv

    env = MultiAgentAttentionEnv(
        world=world,
        reset_callback=scenario.reset_world,
        reward_callback=scenario.reward,
        observation_callback=scenario.observation,
        graph_observation_callback=scenario.graph_observation,
        info_callback=scenario.info_callback,
        id_callback=scenario.get_id,
        update_graph=scenario.update_graph,
        shared_viewer=False,
        scenario_name=args.scenario_name,
    )

    return env


def DGNMPEEnv(args):
    """
    MPE env compatible with the DGN baseline code
    """
    # load scenario from script
    scenario = load("navigation_dgn.py").Scenario()
    # create world
    world = scenario.make_world(args=args)
    from multiagent.environment import MultiAgentDGNEnv, MultiAgentDGN_ATOCEnv

    if args.model_name == "dgn":
        env = MultiAgentDGNEnv(
            world=world,
            reset_callback=scenario.reset_world,
            reward_callback=scenario.reward,
            observation_callback=scenario.observation,
            graph_observation_callback=scenario.graph_observation,
            info_callback=scenario.info_callback
            if hasattr(scenario, "info_callback")
            else None,
            id_callback=scenario.get_id,
            update_graph=scenario.update_graph,
            shared_viewer=False,
            scenario_name=args.scenario_name,
        )
    else:
        env = MultiAgentDGN_ATOCEnv(
            world=world,
            reset_callback=scenario.reset_world,
            reward_callback=scenario.reward,
            observation_callback=scenario.observation,
            graph_observation_callback=scenario.graph_observation,
            info_callback=scenario.info_callback
            if hasattr(scenario, "info_callback")
            else None,
            id_callback=scenario.get_id,
            update_graph=scenario.update_graph,
            shared_viewer=False,
            scenario_name=args.scenario_name,
        )

    return env


def make_parallel_env(args: argparse.Namespace):
    """
    args: argparse.Namespace
        Should include the following:
        env_id: str
            The environment name. Example: 'navigation' for 'navigation.py'
        shared_obs_env: bool
            If we want to use the shared_observation environment or not
        n_rollout_threads: int
            Number of parallel envs to run. This will init a SubProcVecEnv
        seed: int
            Seed for environment
    """
    if args.env_type == "shared_obs":
        from multiagent.env_wrappers import ShareDummyVecEnv as DummyVecEnv
        from multiagent.env_wrappers import ShareSubprocVecEnv as SubprocVecEnv
    else:
        from multiagent.env_wrappers import DummyVecEnv, SubprocVecEnv

    def get_env_fn(rank: int):
        def init_env():
            env = MPEEnv(args)
            env.seed(args.seed + rank * 1000)
            np.random.seed(args.seed + rank * 1000)
            return env

        return init_env

    if args.n_rollout_threads == 1:
        return DummyVecEnv([get_env_fn(0)])
    else:
        return SubprocVecEnv([get_env_fn(i) for i in range(args.n_rollout_threads)])
