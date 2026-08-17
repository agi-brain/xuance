import torch
from dataclasses import dataclass
from typing import Dict
from xuance.common import AgentGrouping
from xuance.torch import Tensor


@dataclass
class AgentGroupedTensor:
    grouped_tensor: Dict[str, Tensor]
    grouping: AgentGrouping

    @property
    def _agent_indices(self):
        return {
            agent: (group, index)
            for group, agents in self.grouping.groups.items()
            for index, agent in enumerate(agents)
        }

    def group(self, group: str) -> Tensor | None:
        """[B, N_group, ...]"""
        if self.grouped_tensor is None:
            return None
        return self.grouped_tensor[group]

    def packed(self, group: str) -> Tensor | None:
        """[B, N_group, ...] -> [B*N_group, ...]"""
        if self.grouped_tensor is None:
            return None
        x = self.grouped_tensor[group]
        return x.flatten(0, 1)

    def agent(self, agent: str) -> Tensor | None:
        """[B, N_group, ...] -> [B, ...]"""
        if self.grouped_tensor is None:
            return None
        group, index = self._agent_indices[agent]
        return self.grouped_tensor[group][:, index]

    @property
    def agent_wise(self) -> Dict[str, Tensor] | None:

        if self.grouped_tensor is None:
            return None

        result = {}

        for group, agents in self.grouping.groups.items():
            value = self.grouped_tensor[group]

            for local_index, agent in enumerate(agents):
                result[agent] = value[:, local_index]

        return result

    @classmethod
    def from_agent_wise(
            cls,
            values: Dict[str, Tensor] | None,
            grouping: AgentGrouping,
    ) -> "AgentGroupedTensor":
        grouped = {}
        if values is None:
            return cls(None, grouping)

        for group, agents in grouping.groups.items():
            grouped[group] = torch.stack(
                [values[agent] for agent in agents],
                dim=1,
            )

        return cls(grouped, grouping)
