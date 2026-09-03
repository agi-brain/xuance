from abc import abstractmethod
from typing import Dict, Optional
from xuance.common import AgentGrouping
from xuance.tensorflow import Tensor, Module, ModuleDict
from xuance.tensorflow.rl_models.modules import RNN_State, MultiAgentModelOutput


class OffPolicyMultiAgentActorCritic(Module):
    def __init__(self,
                 grouping: AgentGrouping,
                 actors: ModuleDict,
                 critics: ModuleDict,
                 use_rnn: bool = False,
                 use_distributed_training: bool = False,
                 **kwargs):
        super().__init__()

        self.grouping = grouping
        self.groups = grouping.groups
        self.group_keys = grouping.group_keys
        self.agent_keys = grouping.agent_keys
        self.n_agents = len(self.agent_keys)
        self.n_group_agents = {k: len(self.groups[k]) for k in self.group_keys}
        self.use_rnn = use_rnn

        self.actors = actors
        self.target_actors = self.actors.clone(copy_weights=True, trainable=False, name="target_actors")
        self.critics = critics
        self.target_critics = self.critics.clone(copy_weights=True, trainable=False, name="target_critics")

        # Prepare DDP module.
        self.distributed_training = use_distributed_training

    @abstractmethod
    def call(
            self,
            observations: Dict[str, Tensor],
            agent_indices: Dict[str, Tensor],
            group_key: Optional[str] = None,
            rnn_states: Dict[str, RNN_State | dict] = None,
            **kwargs
    ) -> MultiAgentModelOutput:
        raise NotImplementedError

    def Qpolicy(
            self,
            observations: Dict[str, Tensor] | Tensor,
            actions: Dict[str, Tensor] | Tensor,
            agent_indices: Dict[str, Tensor],
            group_key: Optional[str] = None,
            rnn_states: Dict[str, RNN_State | dict] = None,
            **kwargs
    ) -> Dict[str, Tensor]:
        raise NotImplementedError

    def Qtarget(
            self,
            observations: Dict[str, Tensor] | Tensor,
            actions: Dict[str, Tensor] | Tensor,
            agent_indices: Dict[str, Tensor],
            group_key: Optional[str] = None,
            rnn_states: Dict[str, RNN_State | dict] = None,
            **kwargs
    ) -> Dict[str, Tensor]:
        raise NotImplementedError

    def Atarget(
            self,
            observations: Dict[str, Tensor],
            agent_indices: Dict[str, Tensor],
            group_key: Optional[str] = None,
            rnn_states: Dict[str, RNN_State | dict] = None,
            **kwargs
    ) -> Dict[str, Tensor]:
        raise NotImplementedError

    def init_rnn_states(self, batch_size: int) -> Dict[str, RNN_State] | None:
        return self.init_actor_rnn_states(batch_size)

    def init_rnn_states_item(self, i_env: int,
                             rnn_states: Dict[str, RNN_State]) -> Dict[str, RNN_State]:
        return self.init_actor_rnn_states_item(i_env, rnn_states)

    def init_actor_rnn_states(self, batch_size: int) -> Dict[str, RNN_State] | None:
        rnn_states = None
        if self.use_rnn:
            rnn_states = {}
            for group in self.group_keys:
                bs = batch_size * self.n_group_agents[group]
                rnn_states[group] = self.actors[group].representation.obs_representation.init_rnn_states(bs)
        return rnn_states

    def init_actor_rnn_states_item(self, i_env: int,
                                   rnn_states: Dict[str, RNN_State]) -> Dict[str, RNN_State]:
        assert self.use_rnn is True, "This method cannot be called when self.use_rnn is False."
        batch_index = [i_env, ]
        for group in self.group_keys:
            rnn_states[group] = self.actors[group].representation.obs_representation.init_rnn_states_item(
                batch_index, rnn_states[group])
        return rnn_states

    def init_critic_rnn_states(self, batch_size: int) -> Dict[str, RNN_State] | None:
        rnn_states = None
        if self.use_rnn:
            rnn_states = {}
            for group in self.group_keys:
                bs = batch_size * self.n_group_agents[group]
                rnn_states[group] = self.critics[group].representation.obs_representation.init_rnn_states(bs)
        return rnn_states

    def init_critic_rnn_states_item(self, i_env: int,
                                    rnn_states: Dict[str, RNN_State]) -> Dict[str, RNN_State]:
        assert self.use_rnn is True, "This method cannot be called when self.use_rnn is False."
        batch_index = [i_env, ]
        for group in self.group_keys:
            rnn_states[group] = self.critics[group].representation.obs_representation.init_rnn_states_item(
                batch_index, rnn_states[group])
        return rnn_states

    def soft_update(self, tau=0.005):
        for ep, tp in zip(self.actors.variables, self.target_actors.variables):
            tp.assign((1 - tau) * tp + tau * ep)
        for ep, tp in zip(self.critics.variables, self.target_critics.variables):
            tp.assign((1 - tau) * tp + tau * ep)
