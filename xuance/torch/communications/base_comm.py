from xuance.torch import Module


class BaseComm(Module):
    def __init__(self, messages_dims, **kwargs):
        super().__init__()

    def forward(self):
        return


class NoneComm():
    def __init__(self):
        pass

    def forward(self):
        pass


    