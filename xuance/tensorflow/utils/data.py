from dataclasses import dataclass
from typing import Dict, Optional

import tensorflow as tf
from tensorflow import Tensor
from xuance.common import AgentGrouping


@dataclass
class AgentGroupedTensor:
    grouped_tensor: Optional[Dict[str, Tensor]]
    grouping: AgentGrouping

    @property
    def _agent_indices(self):
        return {
            agent: (group, index)
            for group, agents in self.grouping.groups.items()
            for index, agent in enumerate(agents)
        }

    def group(self, group: str) -> Optional[Tensor]:
        """[B, N_group, ...]"""
        if self.grouped_tensor is None:
            return None

        return self.grouped_tensor[group]

    def packed(self, group: str) -> Optional[Tensor]:
        """[B, N_group, ...] -> [B * N_group, ...]"""
        if self.grouped_tensor is None:
            return None

        x = self.grouped_tensor[group]

        shape = tf.shape(x)

        return tf.reshape(
            x,
            tf.concat(
                [[shape[0] * shape[1]], shape[2:]],
                axis=0
            )
        )

    def agent(self, agent: str) -> Optional[Tensor]:
        """[B, N_group, ...] -> [B, ...]"""
        if self.grouped_tensor is None:
            return None

        group, index = self._agent_indices[agent]

        return self.grouped_tensor[group][:, index]

    @property
    def agent_wise(self) -> Optional[Dict[str, Tensor]]:
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
            values: Optional[Dict[str, Tensor]],
            grouping: AgentGrouping,
    ) -> Optional["AgentGroupedTensor"]:

        if values is None:
            return None

        grouped = {}

        for group, agents in grouping.groups.items():
            grouped[group] = tf.stack(
                [values[agent] for agent in agents],
                axis=1
            )

        return cls(
            grouped_tensor=grouped,
            grouping=grouping
        )