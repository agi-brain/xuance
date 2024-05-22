from baselines.offpolicy.algorithms.r_maddpg.algorithm.r_actor_critic import (
    R_MADDPG_Actor,
    R_MADDPG_Critic,
)


class R_MATD3_Actor(R_MADDPG_Actor):
    """R_MATD3 Actor is identical to R_MADDPG Actor, see parent class"""

    pass


class R_MATD3_Critic(R_MADDPG_Critic):
    """R_MATD3 Critic class. Identical to R_MADDPG Critic, but with 2 Q outputs."""

    def __init__(self, args, central_obs_dim, central_act_dim, device):
        super(R_MATD3_Critic, self).__init__(
            args, central_obs_dim, central_act_dim, device, num_q_outs=2
        )
