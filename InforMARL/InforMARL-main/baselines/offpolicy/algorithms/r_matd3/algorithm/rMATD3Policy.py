from baselines.offpolicy.algorithms.r_maddpg.algorithm.rMADDPGPolicy import (
    R_MADDPGPolicy,
)


class R_MATD3Policy(R_MADDPGPolicy):
    def __init__(self, config, policy_config, train=True):
        """See parent class."""
        super(R_MATD3Policy, self).__init__(
            config,
            policy_config,
            target_noise=config["args"].target_action_noise_std,
            td3=True,
            train=train,
        )
