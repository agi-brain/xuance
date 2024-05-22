import torch
from baselines.offpolicy.algorithms.mqmix.mqmix import M_QMix


class M_VDN(M_QMix):
    """See parent class."""

    def __init__(
        self,
        args,
        num_agents,
        policies,
        policy_mapping_fn,
        device=torch.device("cuda:0"),
    ):
        super(M_VDN, self).__init__(
            args, num_agents, policies, policy_mapping_fn, device=device, vdn=True
        )
