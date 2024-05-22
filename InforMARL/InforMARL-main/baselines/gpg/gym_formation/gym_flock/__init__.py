from gym.envs.registration import register

register(
    id="LQR-v0",
    entry_point="gym_flock.envs:LQREnv",
    max_episode_steps=200,
)
register(
    id="Consensus-v0",
    entry_point="gym_flock.envs:ConsensusEnv",
    max_episode_steps=500,
)

register(
    id="FormationFlying-v0",
    entry_point="gym_flock.envs:FormationFlyingEnv",
    max_episode_steps=500,
)
register(
    id="FormationFlying-v2",
    entry_point="gym_flock.envs:FormationFlyingEnv2",
    max_episode_steps=500,
)
register(
    id="FormationFlying-v3",
    entry_point="gym_flock.envs:FormationFlyingEnv3",
    max_episode_steps=500,
)
register(
    id="OneAgentFlying-v0",
    entry_point="gym_flock.envs:OneAgentEnv",
    max_episode_steps=500,
)
register(
    id="FormationFlyingInference-v3",
    entry_point="gym_flock.envs:FormationFlyingInferenceEnv3",
    max_episode_steps=500,
)
