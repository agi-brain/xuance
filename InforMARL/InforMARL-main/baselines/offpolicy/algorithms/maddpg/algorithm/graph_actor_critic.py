import gym
import argparse
import torch
import torch.nn as nn
from baselines.offpolicy.utils.util import init, to_torch
from baselines.offpolicy.algorithms.utils.mlp import MLPBase
from baselines.offpolicy.algorithms.utils.act import ACTLayer
from baselines.offpolicy.algorithms.utils.gnn import GNNBase
from baselines.offpolicy.utils.util import get_shape_from_obs_space


class GMADDPG_Actor(nn.Module):
    def __init__(
        self,
        args: argparse.Namespace,
        obs_dim: int,
        node_obs_space: gym.Space,
        edge_obs_space: gym.Space,
        act_dim: int,
        device: torch.device,
    ):
        """
        Actor network class for MADDPG/MATD3. Outputs actions given observations.
        :param args: (argparse.Namespace) arguments containing relevant model information.
        :param obs_dim: (int) dimension of the observation vector.
        :param node_obs_space: (gym.Space) Node Observation Space.
        :param edge_obs_space: (gym.Space) Edge Dimension in graphs.
        :param act_dim: (int) dimension of the action vector.
        :param device: (torch.device) specifies the device to run on (cpu/gpu).
        """
        super(GMADDPG_Actor, self).__init__()
        self._use_orthogonal = args.use_orthogonal
        self._gain = args.gain
        self.hidden_size = args.hidden_size
        self.device = device
        self.tpdv = dict(dtype=torch.float32, device=device)

        node_obs_dim = get_shape_from_obs_space(node_obs_space)[
            1
        ]  # returns (num_nodes, num_node_feats)
        edge_dim = get_shape_from_obs_space(edge_obs_space)[0]  # returns (edge_dim,)

        self.gnn = GNNBase(args, node_obs_dim, edge_dim, args.actor_graph_aggr)
        gnn_out_dim = self.gnn.out_dim  # get output shape from gnn

        mlp_in_dim = obs_dim + gnn_out_dim
        # map observation input into input for rnn
        self.mlp = MLPBase(args, obs_dim)

        # get action from rnn hidden state
        self.act = ACTLayer(act_dim, self.hidden_size, self._use_orthogonal, self._gain)

        self.to(device)

    def forward(self, obs, node_obs, adj, agent_id):
        """
        Compute actions using the needed information.
        obs: (np.ndarray / torch.Tensor)
            Observation inputs into network.
        node_obs (np.ndarray / torch.Tensor):
            Local agent graph node features to the actor.
        adj (np.ndarray / torch.Tensor):
            Adjacency matrix for the graph
        agent_id (np.ndarray / torch.Tensor)
            The agent id to which the observation belongs to
        """
        obs = to_torch(obs).to(**self.tpdv)
        node_obs = to_torch(node_obs).to(**self.tpdv)
        adj = to_torch(adj).to(**self.tpdv)
        agent_id = to_torch(agent_id).to(**self.tpdv)

        nbd_features = self.gnn(node_obs, adj, agent_id)
        actor_features = torch.cat([obs, nbd_features], dim=1)
        actor_features = self.mlp(actor_features)
        # pass outputs through linear layer
        action = self.act(actor_features)

        return action


class GMADDPG_Critic(nn.Module):
    """
    Critic network class for MADDPG/MATD3. Outputs actions given observations.
    :param args: (argparse.Namespace) arguments containing relevant model information.
    :param central_obs_dim: (int) dimension of the centralized observation vector.
    :param central_act_dim: (int) dimension of the centralized action vector.
    :param device: (torch.device) specifies the device to run on (cpu/gpu).
    :param num_q_outs: (int) number of q values to output (1 for MADDPG, 2 for MATD3).
    :param use_cent_obs: (bool) whether to use the centralized observations.
    """

    def __init__(
        self,
        args: argparse.Namespace,
        central_obs_dim: int,
        central_act_dim: int,
        node_obs_space: gym.Space,
        edge_obs_space: gym.Space,
        device: torch.device,
        num_q_outs: int = 1,
        use_cent_obs: bool = True,
    ):
        super(GMADDPG_Critic, self).__init__()
        self._use_orthogonal = args.use_orthogonal
        self.hidden_size = args.hidden_size
        self.device = device
        self.use_cent_obs = use_cent_obs
        self.tpdv = dict(dtype=torch.float32, device=device)

        node_obs_dim = get_shape_from_obs_space(node_obs_space)[
            1
        ]  # (num_nodes, num_node_feats)
        edge_dim = get_shape_from_obs_space(edge_obs_space)[0]  # (edge_dim,)

        self.gnn = GNNBase(args, node_obs_dim, edge_dim, args.critic_graph_aggr)
        gnn_out_dim = self.gnn.out_dim  # get output shape from gnn

        input_dim = gnn_out_dim + central_act_dim
        # if using centralized observations, add them to the input
        if use_cent_obs:
            input_dim += central_obs_dim

        self.mlp = MLPBase(args, input_dim)

        init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][
            self._use_orthogonal
        ]

        def init_(m):
            return init(m, init_method, lambda x: nn.init.constant_(x, 0))

        self.q_outs = [init_(nn.Linear(self.hidden_size, 1)) for _ in range(num_q_outs)]

        self.to(device)

    def forward(self, central_obs, central_act, node_obs, adj, agent_id):
        """
        Compute Q-values using the needed information.
        :param central_obs: (np.ndarray)
            Centralized observations with which to compute Q-values.
        :param central_act: (np.ndarray)
            Centralized actions with which to compute Q-values.
        :param node_obs (np.ndarray):
            Local agent graph node features to the actor.
        :param adj (np.ndarray):
            Adjacency matrix for the graph.
        :param agent_id (np.ndarray / torch.Tensor)
            The agent id to which the observation belongs to

        :return q_values: (list) Q-values outputted by each Q-network.
        """
        central_obs = to_torch(central_obs).to(**self.tpdv)
        central_act = to_torch(central_act).to(**self.tpdv)
        node_obs = to_torch(node_obs).to(**self.tpdv)
        adj = to_torch(adj).to(**self.tpdv)
        agent_id = to_torch(agent_id).to(**self.tpdv)

        nbd_features = self.gnn(node_obs, adj, agent_id)
        if self.use_cent_obs:
            critic_features = torch.cat([central_obs, central_act, nbd_features], dim=1)
        else:
            critic_features = torch.cat([central_act, nbd_features], dim=1)

        critic_features = self.mlp(critic_features)
        q_values = [q_out(critic_features) for q_out in self.q_outs]

        return q_values
