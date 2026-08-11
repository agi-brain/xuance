import torch
import numpy as np
from copy import deepcopy
from xuance.torch import Module, ModuleList, Tensor


class ParameterizedDQN(Module):
    def __init__(self,
                 continuous_actor: Module,
                 q_network: Module,
                 **kwargs):
        super().__init__(**kwargs)
        self.continuous_actor = continuous_actor
        self.q_network = q_network
        self.target_continuous_actor = deepcopy(self.continuous_actor)
        self.target_q_network = deepcopy(self.q_network)

    def Atarget(self, observations: Tensor):
        return self.target_continuous_actor(observations).actions

    def con_action(self, observations: Tensor):
        return self.continuous_actor(observations).actions

    def Qtarget(self, observations: Tensor, actions: Tensor):
        return self.target_q_network(observations, actions).values

    def Qeval(self, observations: Tensor, actions: Tensor):
        return self.q_network(observations, actions).values

    def Qpolicy(self, observations: Tensor):
        continuous_actions = self.continuous_actor(observations).actions
        policy_q = torch.sum(self.q_network(observations, continuous_actions).values)
        return policy_q

    def soft_update(self, tau=0.005):
        for ep, tp in zip(self.continuous_actor.parameters(), self.target_continuous_actor.parameters()):
            tp.data.mul_(1 - tau)
            tp.data.add_(tau * ep.data)
        for ep, tp in zip(self.q_network.parameters(), self.target_q_network.parameters()):
            tp.data.mul_(1 - tau)
            tp.data.add_(tau * ep.data)


class MultipassParameterizedDQN(ParameterizedDQN):
    def __init__(self,
                 continuous_actor: Module,
                 q_network: Module,
                 conact_sizes: np.ndarray,
                 **kwargs):
        super().__init__(continuous_actor, q_network, **kwargs)
        self.offsets = conact_sizes.cumsum()
        self.offsets = np.insert(self.offsets, 0, 0)
        self.num_disact = self.q_network.num_disact

    def Qtarget(self, observations: Tensor, actions: Tensor):
        batch_size = observations.shape[0]
        Q = []
        actions_input = torch.zeros_like(actions)
        actions_input = actions_input.repeat(self.num_disact, 1)
        for i in range(self.num_disact):
            slice_0 = [i * batch_size, (i + 1) * batch_size]
            slice_1 = [self.offsets[i], self.offsets[i + 1]]
            actions_input[slice_0[0]:slice_0[1], slice_1[0]: slice_1[1]] = actions[:, slice_1[0]: slice_1[1]]

        eval_qall = self.target_q_network(observations.repeat(self.num_disact, 1), actions_input).values

        for i in range(self.num_disact):
            eval_q = eval_qall[i * batch_size:(i + 1) * batch_size, i]
            if len(eval_q.shape) == 1:
                eval_q = eval_q.unsqueeze(1)
            Q.append(eval_q)
        Q = torch.cat(Q, dim=1)
        return Q

    def Qeval(self, observations: Tensor, actions: Tensor):
        batch_size = observations.shape[0]
        Q = []
        actions_input = torch.zeros_like(actions)
        actions_input = actions_input.repeat(self.num_disact, 1)
        for i in range(self.num_disact):
            slice_0 = [i * batch_size, (i + 1) * batch_size]
            slice_1 = [self.offsets[i], self.offsets[i + 1]]
            actions_input[slice_0[0]:slice_0[1], slice_1[0]: slice_1[1]] = actions[:, slice_1[0]: slice_1[1]]

        eval_qall = self.q_network(observations.repeat(self.num_disact, 1), actions_input).values
        for i in range(self.num_disact):
            eval_q = eval_qall[i * batch_size:(i + 1) * batch_size, i]
            if len(eval_q.shape) == 1:
                eval_q = eval_q.unsqueeze(1)
            Q.append(eval_q)
        Q = torch.cat(Q, dim=1)
        return Q

    def Qpolicy(self, observations: Tensor):
        conact = self.continuous_actor(observations).actions
        batch_size = observations.shape[0]
        Q = []

        actions_input = torch.zeros_like(conact)
        actions_input = actions_input.repeat(self.num_disact, 1)
        for i in range(self.num_disact):
            slice_0 = [i * batch_size, (i + 1) * batch_size]
            slice_1 = [self.offsets[i], self.offsets[i + 1]]
            actions_input[slice_0[0]:slice_0[1], slice_1[0]: slice_1[1]] = conact[:, slice_1[0]: slice_1[1]]

        eval_qall = self.q_network(observations.repeat(self.num_disact, 1), actions_input).values
        for i in range(self.num_disact):
            eval_q = eval_qall[i * batch_size:(i + 1) * batch_size, i]
            if len(eval_q.shape) == 1:
                eval_q = eval_q.unsqueeze(1)
            Q.append(eval_q)
        Q = torch.cat(Q, dim=1)
        return Q


class SplitParameterisedDQN(MultipassParameterizedDQN):
    def __init__(self,
                 continuous_actor: Module,
                 q_network: ModuleList,
                 conact_sizes: np.ndarray,
                 num_disact: int,
                 **kwargs):
        super(MultipassParameterizedDQN, self).__init__(continuous_actor, q_network, **kwargs)
        self.offsets = conact_sizes.cumsum()
        self.offsets = np.insert(self.offsets, 0, 0)
        self.num_disact = num_disact

    def Qtarget(self, observations: Tensor, actions: Tensor):
        target_Q = []
        for i in range(self.num_disact):
            conact = actions[:, self.offsets[i]:self.offsets[i + 1]]
            eval_q = self.target_q_network[i](observations, conact).values
            target_Q.append(eval_q.unsqueeze(-1))
        target_Q = torch.cat(target_Q, dim=1)
        return target_Q

    def Qeval(self, observations: Tensor, actions: Tensor):
        Q = []
        for i in range(self.num_disact):
            conact = actions[:, self.offsets[i]:self.offsets[i + 1]]
            eval_q = self.q_network[i](observations, conact).values
            Q.append(eval_q.unsqueeze(-1))
        Q = torch.cat(Q, dim=1)
        return Q

    def Qpolicy(self, observations: Tensor):
        conacts = self.continuous_actor(observations).actions
        Q = []
        for i in range(self.num_disact):
            conact = conacts[:, self.offsets[i]:self.offsets[i + 1]]
            eval_q = self.q_network[i](observations, conact).values
            Q.append(eval_q.unsqueeze(-1))
        Q = torch.cat(Q, dim=1)
        return Q
