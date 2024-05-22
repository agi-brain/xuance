from baselines.offpolicy.algorithms.maddpg.algorithm.graph_actor_critic import (
    GMADDPG_Actor,
    GMADDPG_Critic,
)


class GMATD3_Actor(GMADDPG_Actor):
    """MATD3 Actor is identical to MADDPG Actor, see parent class"""

    pass


class GMATD3_Critic(GMADDPG_Critic):
    """MATD3 Critic class. Identical to MADDPG Critic, but with 2 Q output.s"""

    def __init__(self, args, central_obs_dim, central_act_dim, device):
        super(MATD3_Critic, self).__init__(
            args, central_obs_dim, central_act_dim, device, num_q_outs=2
        )
