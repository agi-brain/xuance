import torch
import torch.nn as nn
import torch.nn.functional as F


class VDN_mixer(nn.Module):
    def __init__(self):
        super(VDN_mixer, self).__init__()

    def forward(self, values_n, states=None):
        return values_n.sum(dim=1)


class QMIX_mixer(nn.Module):
    def __init__(self, dim_state, dim_hidden, dim_hypernet_hidden, n_agents, device):
        super(QMIX_mixer, self).__init__()
        self.device = device
        self.dim_state = dim_state
        self.dim_hidden = dim_hidden
        self.dim_hypernet_hidden = dim_hypernet_hidden
        self.n_agents = n_agents
        # self.hyper_w_1 = nn.Linear(self.dim_state, self.dim_hidden * self.n_agents)
        # self.hyper_w_2 = nn.Linear(self.dim_state, self.dim_hidden)
        self.hyper_w_1 = nn.Sequential(nn.Linear(self.dim_state, self.dim_hypernet_hidden),
                                       nn.ReLU(),
                                       nn.Linear(self.dim_hypernet_hidden, self.dim_hidden * self.n_agents)).to(device)
        self.hyper_w_2 = nn.Sequential(nn.Linear(self.dim_state, self.dim_hypernet_hidden),
                                       nn.ReLU(),
                                       nn.Linear(self.dim_hypernet_hidden, self.dim_hidden)).to(device)

        self.hyper_b_1 = nn.Linear(self.dim_state, self.dim_hidden).to(device)
        self.hyper_b_2 = nn.Sequential(nn.Linear(self.dim_state, self.dim_hypernet_hidden),
                                       nn.ReLU(),
                                       nn.Linear(self.dim_hypernet_hidden, 1)).to(device)

    def forward(self, values_n, states):
        states = torch.as_tensor(states, dtype=torch.float32, device=self.device)
        states = states.reshape(-1, self.dim_state)
        agent_qs = values_n.reshape(-1, 1, self.n_agents)
        # First layer
        w_1 = torch.abs(self.hyper_w_1(states))
        w_1 = w_1.view(-1, self.n_agents, self.dim_hidden)
        b_1 = self.hyper_b_1(states)
        b_1 = b_1.view(-1, 1, self.dim_hidden)
        hidden = F.elu(torch.bmm(agent_qs, w_1) + b_1)
        # Second layer
        w_2 = torch.abs(self.hyper_w_2(states))
        w_2 = w_2.view(-1, self.dim_hidden, 1)
        b_2 = self.hyper_b_2(states)
        b_2 = b_2.view(-1, 1, 1)
        # Compute final output
        y = torch.bmm(hidden, w_2) + b_2
        # Reshape and return
        q_tot = y.view(-1, 1)
        return q_tot


class QMIX_FF_mixer(nn.Module):
    def __init__(self, dim_state, dim_hidden, n_agents, device):
        super(QMIX_FF_mixer, self).__init__()
        self.device = device
        self.dim_state = dim_state
        self.dim_hidden = dim_hidden
        self.n_agents = n_agents
        self.dim_input = self.n_agents + self.dim_state
        self.ff_net = nn.Sequential(nn.Linear(self.dim_input, self.dim_hidden),
                                    nn.ReLU(),
                                    nn.Linear(self.dim_hidden, self.dim_hidden),
                                    nn.ReLU(),
                                    nn.Linear(self.dim_hidden, self.dim_hidden),
                                    nn.ReLU(),
                                    nn.Linear(self.dim_hidden, 1)).to(self.device)
        self.ff_net_bias = nn.Sequential(nn.Linear(self.dim_state, self.dim_hidden),
                                         nn.ReLU(),
                                         nn.Linear(self.dim_hidden, 1)).to(self.device)

    def forward(self, values_n, states):
        states = states.reshape(-1, self.dim_state)
        agent_qs = values_n.view([-1, self.n_agents])
        inputs = torch.cat([agent_qs, states], dim=-1).to(self.device)
        out_put = self.ff_net(inputs)
        bias = self.ff_net_bias(states)
        y = out_put + bias
        q_tot = y.view([-1, 1])
        return q_tot


class QTRAN_base(nn.Module):
    def __init__(self, dim_state, dim_action, dim_hidden, n_agents, dim_utility_hidden):
        super(QTRAN_base, self).__init__()
        self.dim_state = dim_state
        self.dim_action = dim_action
        self.dim_hidden = dim_hidden
        self.n_agents = n_agents
        self.dim_q_input = (dim_utility_hidden + self.dim_action) * self.n_agents
        self.dim_v_input = dim_utility_hidden * self.n_agents

        self.Q_jt = nn.Sequential(nn.Linear(self.dim_q_input, self.dim_hidden),
                                  nn.ReLU(),
                                  nn.Linear(self.dim_hidden, self.dim_hidden),
                                  nn.ReLU(),
                                  nn.Linear(self.dim_hidden, 1))
        self.V_jt = nn.Sequential(nn.Linear(self.dim_v_input, self.dim_hidden),
                                  nn.ReLU(),
                                  nn.Linear(self.dim_hidden, self.dim_hidden),
                                  nn.ReLU(),
                                  nn.Linear(self.dim_hidden, 1))

    def forward(self, hidden_states_n, actions_n):
        input_q = torch.cat([hidden_states_n, actions_n], dim=-1).view([-1, self.dim_q_input])
        input_v = hidden_states_n.view([-1, self.dim_v_input])
        q_jt = self.Q_jt(input_q)
        v_jt = self.V_jt(input_v)
        return q_jt, v_jt


class QTRAN_alt(QTRAN_base):
    def __init__(self, dim_state, dim_action, dim_hidden, n_agents, dim_utility_hidden):
        super(QTRAN_alt, self).__init__(dim_state, dim_action, dim_hidden, n_agents, dim_utility_hidden)

    def counterfactual_values(self, q_self_values, q_selected_values):
        q_repeat = q_selected_values.unsqueeze(dim=1).repeat(1, self.n_agents, 1, self.dim_action)
        counterfactual_values_n = q_repeat
        for agent in range(self.n_agents):
            counterfactual_values_n[:, agent, agent] = q_self_values[:, agent, :]
        return counterfactual_values_n.sum(dim=2)

    def counterfactual_values_hat(self, hidden_states_n, actions_n):
        action_repeat = actions_n.unsqueeze(dim=2).repeat(1, 1, self.dim_action, 1)
        action_self_all = torch.eye(self.dim_action).unsqueeze(0)
        action_counterfactual_n = action_repeat.unsqueeze(dim=2).repeat(1, 1, self.n_agents, 1, 1)  # batch * N * N * dim_a * dim_a
        q_n = []
        for agent in range(self.n_agents):
            action_counterfactual_n[:, agent, agent, :, :] = action_self_all
            q_actions = []
            for a in range(self.dim_action):
                input_a = action_counterfactual_n[:, :, agent, a, :]
                q, _ = self.forward(hidden_states_n, input_a)
                q_actions.append(q)
            q_n.append(torch.cat(q_actions, dim=-1).unsqueeze(dim=1))
        return torch.cat(q_n, dim=1)
