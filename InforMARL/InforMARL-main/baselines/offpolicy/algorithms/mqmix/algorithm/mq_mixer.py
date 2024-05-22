import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from baselines.offpolicy.utils.util import init


class M_QMixer(nn.Module):
    """
    Computes total Q values given agent q values and global states.
    :param args: (namespace) contains information about hyperparameters and algorithm configuration
    :param num_agents: (int) number of agents in env
    :param cent_obs_dim: (int) dimension of the centralized state
    :param device: (torch.Device) torch device on which to do computations.
    :param multidiscrete_list: (list) list of each action dimension if action space is multidiscrete
    """

    def __init__(self, args, num_agents, cent_obs_dim, device, multidiscrete_list=None):
        """
        init mixer class
        """
        super(M_QMixer, self).__init__()
        self.device = device
        self.num_agents = num_agents
        self.cent_obs_dim = cent_obs_dim
        self.use_orthogonal = args.use_orthogonal

        # dimension of the hidden layer of the mixing net
        self.hidden_layer_dim = args.mixer_hidden_dim
        # dimension of the hidden layer of each hypernet
        self.hypernet_hidden_dim = args.hypernet_hidden_dim

        if multidiscrete_list:
            self.num_mixer_q_inps = sum(multidiscrete_list)
        else:
            self.num_mixer_q_inps = self.num_agents

        if self.use_orthogonal:

            def init_(m):
                return init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0))

        else:

            def init_(m):
                return init(
                    m, nn.init.xavier_uniform_, lambda x: nn.init.constant_(x, 0)
                )

        # hypernets output the weight and bias for the 2 layer MLP which takes in the state and agent Qs and outputs Q_tot
        if args.hypernet_layers == 1:
            # each hypernet only has 1 layer to output the weights
            # hyper_w1 outputs weight matrix which is of dimension (hidden_layer_dim x N)
            self.hyper_w1 = init_(
                nn.Linear(
                    self.cent_obs_dim, self.num_mixer_q_inps * self.hidden_layer_dim
                )
            ).to(self.device)
            # hyper_w2 outputs weight matrix which is of dimension (1 x hidden_layer_dim)
            self.hyper_w2 = init_(
                nn.Linear(self.cent_obs_dim, self.hidden_layer_dim)
            ).to(self.device)
        elif args.hypernet_layers == 2:
            # 2 layer hypernets: output dimensions are same as above case
            self.hyper_w1 = nn.Sequential(
                init_(nn.Linear(self.cent_obs_dim, self.hypernet_hidden_dim)),
                nn.ReLU(),
                init_(
                    nn.Linear(
                        self.hypernet_hidden_dim,
                        self.num_mixer_q_inps * self.hidden_layer_dim,
                    )
                ),
            ).to(self.device)
            self.hyper_w2 = nn.Sequential(
                init_(nn.Linear(self.cent_obs_dim, self.hypernet_hidden_dim)),
                nn.ReLU(),
                init_(nn.Linear(self.hypernet_hidden_dim, self.hidden_layer_dim)),
            ).to(self.device)

        # hyper_b1 outputs bias vector of dimension (1 x hidden_layer_dim)
        self.hyper_b1 = init_(nn.Linear(self.cent_obs_dim, self.hidden_layer_dim)).to(
            self.device
        )
        # hyper_b2 outptus bias vector of dimension (1 x 1)
        self.hyper_b2 = nn.Sequential(
            init_(nn.Linear(self.cent_obs_dim, self.hypernet_hidden_dim)),
            nn.ReLU(),
            init_(nn.Linear(self.hypernet_hidden_dim, 1)),
        ).to(self.device)

    def forward(self, agent_q_inps, states):
        """
        Computes Q_tot using the individual agent q values and global state.
        :param agent_q_inps: (torch.Tensor) individual agent q values
        :param states: (torch.Tensor) state input to the hypernetworks.
        :return Q_tot: (torch.Tensor) computed Q_tot values
        """
        if type(agent_q_inps) == np.ndarray:
            agent_q_inps = torch.FloatTensor(agent_q_inps)
        if type(states) == np.ndarray:
            states = torch.FloatTensor(states)

        agent_q_inps = agent_q_inps.to(self.device)
        states = states.to(self.device)

        batch_size = agent_q_inps.size(0)
        states = states.view(-1, self.cent_obs_dim).float()
        # reshape agent_q_inps into shape (batch_size x 1 x N) to work with torch.bmm
        agent_q_inps = agent_q_inps.view(-1, 1, self.num_mixer_q_inps).float()

        # get the first layer weight matrix batch, apply abs val to ensure nonnegative derivative
        w1 = torch.abs(self.hyper_w1(states))
        # get first bias vector
        b1 = self.hyper_b1(states)
        # reshape to batch_size x N x Hidden Layer Dim (there's a different weight mat for each batch element)
        w1 = w1.view(-1, self.num_mixer_q_inps, self.hidden_layer_dim)
        # reshape to batch_size x 1 x Hidden Layer Dim
        b1 = b1.view(-1, 1, self.hidden_layer_dim)
        # pass the agent qs through first layer defined by the weight matrices, and apply Elu activation
        hidden_layer = F.elu(torch.bmm(agent_q_inps, w1) + b1)
        # get second layer weight matrix batch
        w2 = torch.abs(self.hyper_w2(states))
        # get second layer bias batch
        b2 = self.hyper_b2(states)

        # reshape to shape (batch_size x hidden_layer dim x 1)
        w2 = w2.view(-1, self.hidden_layer_dim, 1)
        # reshape to shape (batch_size x 1 x 1)
        b2 = b2.view(-1, 1, 1)
        # pass the hidden layer results through output layer, with no activataion
        out = torch.bmm(hidden_layer, w2) + b2
        # reshape to (batch_size, 1, 1)
        q_tot = out.view(batch_size, -1, 1)

        q_tot = q_tot.cpu()

        return q_tot
