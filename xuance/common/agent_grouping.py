from dataclasses import dataclass
from typing import Dict, Mapping, Tuple, Sequence


@dataclass(frozen=True)
class AgentGrouping:
    """Defines which agents share the same model parameters."""

    agent_keys: Tuple[str, ...]
    assignments: Tuple[Tuple[str, str], ...]

    # assignments: ((agent_key, group_key), ...)

    def __post_init__(self):
        if len(set(self.agent_keys)) != len(self.agent_keys):
            raise ValueError("agent_keys contains duplicated agents.")

        assigned_agents = [agent for agent, _ in self.assignments]

        if len(set(assigned_agents)) != len(assigned_agents):
            raise ValueError("An agent cannot belong to multiple groups.")

        missing = set(self.agent_keys) - set(assigned_agents)
        unknown = set(assigned_agents) - set(self.agent_keys)

        if missing:
            raise ValueError(f"Agents without groups: {missing}.")
        if unknown:
            raise ValueError(f"Unknown agents: {unknown}.")

    @property
    def agent_to_group(self) -> Dict[str, str]:
        return dict(self.assignments)

    @property
    def group_keys(self) -> Tuple[str, ...]:
        mapping = self.agent_to_group
        return tuple(dict.fromkeys(mapping[a] for a in self.agent_keys))

    @property
    def groups(self) -> Dict[str, Tuple[str, ...]]:
        mapping = self.agent_to_group
        return {group_key: tuple(agent for agent in self.agent_keys if mapping[agent] == group_key)
                for group_key in self.group_keys}

    def group_of(self, agent_key: str) -> str:
        return self.agent_to_group[agent_key]

    def agents_in(self, group_key: str) -> Tuple[str, ...]:
        return self.groups[group_key]

    def agent_indices(self, group_key: str) -> Tuple[int, ...]:
        members = set(self.agents_in(group_key))
        return tuple(index for index, agent in enumerate(self.agent_keys) if agent in members)

    @property
    def full_shared(self) -> bool:
        return len(self.group_keys) == 1

    @property
    def full_independent(self) -> bool:
        return len(self.group_keys) == len(self.agent_keys)

    @classmethod
    def shared(cls, agent_keys: Sequence[str]) -> "AgentGrouping":
        agent_keys = tuple(agent_keys)
        return cls(
            agent_keys=agent_keys,
            assignments=tuple((agent, "shared") for agent in agent_keys)
        )

    @classmethod
    def independent(cls, agent_keys: Sequence[str]) -> "AgentGrouping":
        agent_keys = tuple(agent_keys)
        return cls(
            agent_keys=agent_keys,
            assignments=tuple((agent, agent) for agent in agent_keys)
        )

    @classmethod
    def from_groups(cls,
                    agent_keys: Sequence[str],
                    groups: Mapping[str, Sequence[str]]) -> "AgentGrouping":
        assignments = []

        for group_key, members in groups.items():
            assignments.extend((agent_key, group_key) for agent_key in members)

        return cls(
            agent_keys=tuple(agent_keys),
            assignments=tuple(assignments)
        )
