from xuance.torch import Module, Tensor


class BaseComm(Module):
    def __init__(self, n_agents, msg_dims, **kwargs):
        super().__init__()
        self.n_agents = n_agents
        self.msg_dims = msg_dims

    def forward(self, msg: Tensor, **kwargs):
        raise NotImplementedError


class NoneComm(Module):
    def __init__(self):
        super().__init__()

    def forward(self, msg: Tensor, **kwargs):
        return msg


