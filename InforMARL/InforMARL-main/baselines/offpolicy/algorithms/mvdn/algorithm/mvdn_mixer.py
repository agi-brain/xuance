import torch.nn as nn
from baselines.offpolicy.utils.util import to_torch


class M_VDNMixer(nn.Module):
    """
    Computes total Q values given agent q values and global states.
    :param args: (namespace) contains information about hyperparameters and algorithm configuration (unused).
    :param num_agents: (int) number of agents in env
    :param cent_obs_dim: (int) dimension of the centralized state (unused).
    :param device: (torch.Device) torch device on which to do computations.
    :param multidiscrete_list: (list) list of each action dimension if action space is multidiscrete
    """

    def __init__(self, args, num_agents, cent_obs_dim, device, multidiscrete_list=None):
        """
        init mixer class
        """
        super(M_VDNMixer, self).__init__()
        self.device = device
        self.num_agents = num_agents

        if multidiscrete_list:
            self.num_mixer_q_inps = sum(multidiscrete_list)
        else:
            self.num_mixer_q_inps = self.num_agents

    def forward(self, agent_q_inps, states):
        """
        Computes Q_tot by summing individual agent q values.
        :param agent_q_inps: (torch.Tensor) individual agent q values
        VDN does not use centralized state, so states is unused.
        Ref: https://github.com/hijkzzz/pymarl2/blob/1c691e30e6bcbebbcd8f7bfd92efd50a9a59a5c1/src/modules/mixers/vdn.py#L10

        :return Q_tot: (torch.Tensor) computed Q_tot values
        """
        agent_q_inps = to_torch(agent_q_inps)

        return agent_q_inps.sum(dim=-1).view(-1, 1, 1)
