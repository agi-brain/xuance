from typing import Union, Tuple
from xuance.mindspore import Module, Tensor
from xuance.mindspore.rl_models.actors import CategoricalActor, GaussianActor


class VanillaPolicyGradient(Module):
    def __init__(self,
                 actor: Module,
                 **kwargs):
        super().__init__(**kwargs)
        self.actor = actor
        if isinstance(self.actor, CategoricalActor):
            self.is_continuous = False
        elif isinstance(self.actor, GaussianActor):
            self.is_continuous = True
        else:
            raise NotImplementedError

    def construct(self,
                  observation: Union[Tensor, dict],
                  **kwargs) -> Tuple[Tensor, ...]:
        if self.is_continuous:
            a_mean, log_std, _ = self.actor(observation, **kwargs)
            return a_mean, log_std
        else:
            logits, _ = self.actor(observation, **kwargs)
            return logits


