"""
Multi-agent Soft Actor-critic (MASAC)
Implementation: Pytorch
"""
import torch
from torch import nn
from typing import Optional, Sequence, Union
from argparse import Namespace
from xuance.torch.learners.multi_agent_rl.isac_learner import ISAC_Learner


class MASAC_Learner(ISAC_Learner):
    def __init__(self,
                 config: Namespace,
                 policy: nn.Module,
                 optimizer: Sequence[torch.optim.Optimizer],
                 scheduler: Sequence[torch.optim.lr_scheduler._LRScheduler] = None,
                 device: Optional[Union[int, str, torch.device]] = None,
                 model_dir: str = "./",
                 **kwargs):
        super(MASAC_Learner, self).__init__(config, policy, optimizer, scheduler, device, model_dir, **kwargs)
