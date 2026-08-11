import torch

class Harmonizer(torch.nn.Module):
    """Learnable parameter for loss_scale balancing
    Ref: https://github.com/thuml/HarmonyDream/blob/main/dreamerv3-jax/dreamerv3/nets.py
    """
    def __init__(self, device):
        super().__init__()
        self.harmony_s = torch.nn.Parameter(torch.tensor(0.0, device=device))

    def forward(self, loss: torch.Tensor, regularize=True):
        if regularize:
          return loss / (torch.exp(self.harmony_s)) + torch.log(torch.exp(self.harmony_s) + 1)
        else:
          return loss / (torch.exp(self.harmony_s))

    def get_harmony(self):
        return self.harmony_s