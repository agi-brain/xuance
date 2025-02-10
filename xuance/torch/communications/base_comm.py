from xuance.torch import Module


class BaseComm(Module):
    def __init__(self, messages_dims, **kwargs):
        super().__init__()

    def forward(self):
        return


class NoneComm(Module):
    def __init__(self):
        super().__init__()

    def forward(self):
        pass


