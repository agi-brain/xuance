from baselines.offpolicy.algorithms.maddpg.algorithm.GMADDPGPolicy import GMADDPGPolicy


class GMATD3Policy(GMADDPGPolicy):
    def __init__(self, config, policy_config, train=True):
        """See parent class."""
        super(GMATD3Policy, self).__init__(
            config,
            policy_config,
            target_noise=config["args"].target_action_noise_std,
            td3=True,
            train=train,
        )
