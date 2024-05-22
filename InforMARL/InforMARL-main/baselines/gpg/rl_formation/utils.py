import numpy as np
import pdb
import torch.nn.functional as F
import torch as t
import math

from torch.autograd import Variable
import torch


pi = Variable(torch.FloatTensor([math.pi]))


def normal(x, mu, sigma_sq):
    a = (-1 * (Variable(x) - mu).pow(2) / (2 * sigma_sq)).exp()
    b = 1 / (2 * sigma_sq * pi.expand_as(sigma_sq)).sqrt()
    return a * b


def select_action(state, g, policy):
    state = Variable(torch.FloatTensor(state))
    mu, sigma = policy(g, state)

    sigma = F.softplus(sigma)

    eps = torch.randn(mu.size())
    action = (mu + sigma.sqrt() * Variable(eps)).data
    prob = normal(action, mu, sigma)

    prob2 = torch.prod(prob.reshape(-1))
    log_prob = prob2.log()

    # Add log probability of our chosen action to our history

    if len(policy.policy_history) > 0:
        policy.policy_history = torch.cat(
            [policy.policy_history, (log_prob.reshape(1))]
        )
    else:
        policy.policy_history = log_prob.reshape(1)

    return action


def update_policy(policy, optimizer):
    R = 0
    rewards = []

    # Discount future rewards back to the present using gamma
    for r in policy.reward_episode[::-1]:
        R = r + policy.gamma * R
        rewards.insert(0, R)

    # Scale rewards
    rewards = torch.FloatTensor(rewards)
    rewards = (rewards - rewards.mean()) / (rewards.std() + np.finfo(np.float32).eps)
    # Calculate loss
    loss = torch.sum(torch.mul(policy.policy_history, Variable(rewards)).mul(-1), -1)
    # pdb.set_trace()
    # Update network weights
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Save and intialize episode history counters

    policy.loss_history.append(loss.item())
    policy.reward_history.append(np.sum(policy.reward_episode))
    policy.policy_history = Variable(torch.Tensor())
    policy.reward_episode = []


def get_weights(policy):
    w = list(policy.parameters())
    w = np.asarray(w)

    weights = {}

    for name, param in policy.named_parameters():
        if param.requires_grad:
            # print (name, param.data)
            weights["%s" % name] = param.data
            # print (name)
    return weights


def set_weights(new_policy, old_policy):
    w = get_weights(old_policy)
    w2 = get_weights(new_policy)
    pdb.set_trace()
