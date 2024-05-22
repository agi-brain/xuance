import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import dgl.function as fn


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
# The forward function is essentially the same as any other commonly seen NNs
# model in PyTorch.  We can initialize GCN like any ``nn.Module``. For example,
# let's define a simple neural network consisting of two GCN layers. Suppose we
# are training the classifier for the cora dataset (the input feature size is
# 1433 and the number of classes is 7).


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.l1 = nn.Linear(2, 64, bias=False)
        self.l2 = nn.Linear(64, 2, bias=False)
        self.l3 = nn.Linear(64, 2, bias=False)

        # self.gcn1 = GCN(2, 16, t.tanh)
        # self.gcn11 = GCN(16, 16, t.tanh)
        # self.gcn111 = GCN(64, 32, F.relu)
        # self.gcn2 = GCN(16, 2, t.tanh)
        # self.gcn2_ = GCN(16,2,t.tanh)

        # self.gcn2 = GCN(16, 2, t.tanh)
        # self.gcn2_ = GCN(16,2,t.tanh)

        self.policy_history = Variable(torch.Tensor())
        self.reward_episode = []
        # Overall reward and loss history
        self.reward_history = []
        self.loss_history = []
        self.gamma = 0.99

    def forward(self, g, features):
        # x = self.gcn1(g, features)
        # x = self.gcn1(g,x)
        # x = self.gcn11(g,x)
        # x = self.gcn11(g,x)
        x = F.relu(self.l1(features))
        mu = F.relu(self.l2(x))
        sigma = F.relu(self.l3(x))
        # mu = self.gcn2(g, x)
        # sigma = self.gcn2_(g,x)
        return mu, sigma
