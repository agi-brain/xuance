from baselines.offpolicy.algorithms.maddpg.maddpg import MADDPG


class MATD3(MADDPG):
    def __init__(self, args, num_agents, policies, policy_mapping_fn, device=None):
        """See parent class."""
        super(MATD3, self).__init__(
            args,
            num_agents,
            policies,
            policy_mapping_fn,
            device=device,
            actor_update_interval=2,
        )
