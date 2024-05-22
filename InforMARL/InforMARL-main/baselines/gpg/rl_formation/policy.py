import torch.nn as nn
import torch as t
import dgl.function as fn

from torch.autograd import Variable
import torch

gcn_msg = fn.copy_src(src="h", out="m")
gcn_reduce = fn.sum(msg="m", out="h")

###############################################################################
# We then define the node UDF for ``apply_nodes``, which is a fully-connected layer:


class NodeApplyModule(nn.Module):
    def __init__(self, in_feats, out_feats, activation):
        super(NodeApplyModule, self).__init__()
        self.linear = nn.Linear(in_feats, out_feats)
        self.activation = activation

    def forward(self, node):
        h = self.linear(node.data["h"])
        h = self.activation(h)
        return {"h": h}


###############################################################################
# We then proceed to define the GCN module. A GCN layer essentially performs
# message passing on all the nodes then applies the `NodeApplyModule`. Note
# that we omitted the dropout in the paper for simplicity.


class GCN(nn.Module):
    def __init__(self, in_feats, out_feats, activation):
        super(GCN, self).__init__()
        self.apply_mod = NodeApplyModule(in_feats, out_feats, activation)

    def forward(self, g, feature):
        g.ndata["h"] = feature
        g.update_all(gcn_msg, gcn_reduce)
        g.apply_nodes(func=self.apply_mod)
        return g.ndata.pop("h")


###############################################################################


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.gcn1 = GCN(2, 16, t.tanh)
        self.gcn2_mu = GCN(16, 2, t.tanh)
        self.gcn2_sigma = GCN(16, 2, t.tanh)

        self.policy_history = Variable(torch.Tensor())
        self.reward_episode = []
        # Overall reward and loss history
        self.reward_history = []
        self.loss_history = []
        self.gamma = 0.99

    def forward(self, g, features):
        x = self.gcn1(g, features)
        mu = self.gcn2_mu(g, x)
        sigma = self.gcn2_sigma(g, x)
        return mu, sigma
