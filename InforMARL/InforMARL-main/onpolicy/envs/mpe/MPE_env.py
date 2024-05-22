from multiagent.custom_scenarios import load
from multiagent.environment import MultiAgentPPOEnv as MultiAgentEnv
from multiagent.environment import MultiAgentGraphEnv

# from onpolicy.envs.mpe.scenarios import load
# from onpolicy.envs.mpe.environment import MultiAgentEnv
# TODO USE the MPE_Env from multiagent/


def MPEEnv(args):
    """
    Creates a MultiAgentEnv object as env. This can be used similar to a gym
    environment by calling env.reset() and env.step().
    Use env.render() to view the environment on the screen.

    Input:
        scenario_name   :   name of the scenario from ./scenarios/ to be Returns
                            (without the .py extension)
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
    # create multiagent environment
    # if 'navigation' in args.scenario_name:
    env = MultiAgentEnv(
        world=world,
        reset_callback=scenario.reset_world,
        reward_callback=scenario.reward,
        observation_callback=scenario.observation,
        info_callback=scenario.info_callback
        if hasattr(scenario, "info_callback")
        else None,
    )
    # else:
    # env = MultiAgentEnv(world=world, reset_callback=scenario.reset_world,
    # reward_callback=scenario.reward,
    # observation_callback=scenario.observation)

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
    )

    return env
