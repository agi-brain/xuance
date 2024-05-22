from baselines.offpolicy.algorithms.maddpg.algorithm.actor_critic import (
    MADDPG_Actor,
    MADDPG_Critic,
)


class MATD3_Actor(MADDPG_Actor):
    """MATD3 Actor is identical to MADDPG Actor, see parent class"""

    pass


class MATD3_Critic(MADDPG_Critic):
    """MATD3 Critic class. Identical to MADDPG Critic, but with 2 Q output.s"""

    def __init__(self, args, central_obs_dim, central_act_dim, device):
        super(MATD3_Critic, self).__init__(
            args, central_obs_dim, central_act_dim, device, num_q_outs=2
        )
