from baselines.offpolicy.algorithms.r_maddpg.r_maddpg import R_MADDPG


class R_MATD3(R_MADDPG):
    def __init__(
        self,
        args,
        num_agents,
        policies,
        policy_mapping_fn,
        device=None,
        episode_length=None,
    ):
        """See parent class."""
        super(R_MATD3, self).__init__(
            args,
            num_agents,
            policies,
            policy_mapping_fn,
            device=device,
            episode_length=episode_length,
            actor_update_interval=2,
        )
