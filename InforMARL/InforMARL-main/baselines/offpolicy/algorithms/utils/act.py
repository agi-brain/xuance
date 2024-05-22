import numpy as np
import torch.nn as nn
from baselines.offpolicy.utils.util import init


class ACTLayer(nn.Module):
    def __init__(self, act_dim, hidden_size, use_orthogonal, gain):
        super(ACTLayer, self).__init__()

        self.multi_discrete = False
        init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][use_orthogonal]

        def init_(m):
            return init(m, init_method, lambda x: nn.init.constant_(x, 0), gain)

        if isinstance(act_dim, np.ndarray):
            # MultiDiscrete setting: have n Linear layers for each action
            self.multi_discrete = True
            self.action_outs = nn.ModuleList(
                [init_(nn.Linear(hidden_size, a_dim)) for a_dim in act_dim]
            )
        else:
            self.action_out = init_(nn.Linear(hidden_size, act_dim))

    def forward(self, x, no_sequence=False):
        if self.multi_discrete:
            act_outs = []
            for a_out in self.action_outs:
                act_out = a_out(x)
                if no_sequence:
                    # remove the dummy first time dimension if the input didn't have a time dimension
                    act_out = act_out[0, :, :]
                act_outs.append(act_out)
        else:
            act_outs = self.action_out(x)
            if no_sequence:
                # remove the dummy first time dimension if the input didn't have a time dimension
                act_outs = act_outs[0, :, :]

        return act_outs
