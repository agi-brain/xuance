import numpy as np
import torch.nn as nn
import dgl.function as fn

from torch.autograd import Variable
import torch
from baselines.gpg.rl_navigation.modules.act import ACTLayer
from baselines.gpg.rl_navigation.util import get_shape_from_obs_space

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


class Policy(nn.Module):
    def __init__(self, args, obs_space, action_space, device):
        super(Policy, self).__init__()
        obs_shape = get_shape_from_obs_space(obs_space)[0]
        self.action_space = action_space
        self.num_agents = args.num_agents
        self.hidden_size = args.hidden_size

        self.gcn1 = GCN(obs_shape, self.hidden_size, torch.tanh)
        self.gcn2 = GCN(self.hidden_size, self.hidden_size, torch.tanh)
        self.act = ACTLayer(
            action_space,
            self.hidden_size,
            use_orthogonal=args.use_orthogonal,
            gain=args.gain,
        )

        self.policy_history = Variable(torch.Tensor())
        self.reward_episode = []
        # Overall reward history
        self.reward_history = []
        self.gamma = 0.99
        self.to(device)

    def forward(self, g, features):
        x = self.gcn1(g, features)
        x = self.gcn2(g, x)
        actions, action_log_probs = self.act(x)
        return actions, action_log_probs

    def get_actions(self, actions):
        """
        Convert actions from network to environment compatible actions
        """
        actions = actions.detach().cpu().numpy()

        # rearrange actions
        if self.action_space.__class__.__name__ == "MultiDiscrete":
            for i in range(self.envs.action_space.shape):
                uc_actions_env = np.eye(self.action_space.high[i] + 1)[actions[:, :, i]]
                if i == 0:
                    actions_env = uc_actions_env
                else:
                    actions_env = np.concatenate((actions_env, uc_actions_env), axis=2)
        elif self.action_space.__class__.__name__ == "Discrete":
            actions_env = np.squeeze(np.eye(self.action_space.n)[actions], 1)
        else:
            raise NotImplementedError
        return actions_env

    def select_action(self, state, g):
        state = Variable(torch.FloatTensor(state))
        action, action_log_probs = self(g, state)
        action = self.get_actions(action)
        # action shape: (num_agents, act_shape)
        # action_log_probs shape: (num_agents, 1)

        # Add log probability of our chosen action to our history

        if len(self.policy_history) > 0:
            self.policy_history = torch.cat([self.policy_history, (action_log_probs)])
        else:
            self.policy_history = action_log_probs

        return action

    def update_policy(self, optimizer):
        R = 0
        rewards = []

        # Discount future rewards back to the present using gamma
        # FIXME: need to to this for all agents
        for r in self.reward_episode[::-1]:
            R = r + self.gamma * R
            rewards.insert(0, R)

        # Scale rewards
        rewards = torch.FloatTensor(rewards)
        rewards = (rewards - rewards.mean()) / (
            rewards.std() + np.finfo(np.float32).eps
        )
        # rewards has shape [episode_length] we need to make it [episode_length * num_agents]
        # so that it can be multiplied by the gradient of the log probability of the actions
        # and policy.policy_history has shape [episode_length * num_agents]
        rewards = rewards.repeat(
            self.num_agents, 1
        ).T  # shape [episode_length, num_agents]
        rewards = rewards.reshape(-1).unsqueeze(
            1
        )  # shape [episode_length * num_agents, 1]
        # Calculate loss
        # loss = (torch.sum(torch.mul(policy.policy_history, Variable(rewards)).mul(-1), -1))
        loss = -torch.sum(self.policy_history.mul(Variable(rewards)))
        # Update network weights
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Save and intialize episode history counters
        train_info_dict = {"loss": loss.item()}

        self.reward_history.append(np.sum(self.reward_episode))
        self.policy_history = Variable(torch.Tensor())
        self.reward_episode = []
        return train_info_dict
