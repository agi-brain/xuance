from typing import Union
from xuance.torch import Module, Tensor
from xuance.torch.rl_models.modules import ModelOutput


class VanillaPolicyGradient(Module):
    def __init__(self,
                 actor: Module,
                 **kwargs):
        super().__init__(**kwargs)
        self.actor = actor

    def forward(self,
                observation: Union[Tensor, dict],
                **kwargs) -> ModelOutput:
        actor_output = self.actor(observation, **kwargs)
        return ModelOutput(distributions=actor_output.distributions,
                           actor_rep_out=actor_output.representations)

    def act(self,
            observation: Union[Tensor, dict],
            deterministic: bool = False,
            **kwargs) -> Tensor:
        actor_output = self.actor(observation, **kwargs)

        if deterministic:
            actions = actor_output.distribution.deterministic_sample()
        else:
            actions = actor_output.distribution.stochastic_sample()
        return actions

