agents
-----------------------------------

.. toctree::
    :hidden:
    :maxdepth: 1

    base <agents/base>
    contrastive_unsupervised_rl <agents/contrastive_unsupervised_rl>
    core <agents/core>
    model_based_rl <agents/model_based_rl>
    multi_agent_rl <agents/multi_agent_rl>
    offline_rl <agents/offline_rl>
    policy_gradient <agents/policy_gradient>
    qlearning_family <agents/qlearning_family>

- :doc:`base <agents/base>`.
- :doc:`contrastive_unsupervised_rl <agents/contrastive_unsupervised_rl>`.
- :doc:`core <agents/core>`.
- :doc:`model_based_rl <agents/model_based_rl>`.
- :doc:`multi_agent_rl <agents/multi_agent_rl>`.
- :doc:`offline_rl <agents/offline_rl>`.
- :doc:`policy_gradient <agents/policy_gradient>`.
- :doc:`qlearning_family <agents/qlearning_family>`.

Agent is an entity or a decision-making component that interacts with an environment or other agents to learn and perform tasks.

For DRL algorithms, agent represents the decision-maker that seeks to maximize a reward signal over time by interacting with the environment.
It observes the environment's state, takes actions, and receives feedback (rewards) from the environment.
The goal is to learn a policy :math:`\pi(s)`, which maps observed states :math:`s` to optimal actions :math:`a`,
to maximize the cumulative rewards (often expressed as a discounted sum).

For MARL algorithms, it involves multiple agents, each of which operates independently or cooperatively within the same environment.
Each agent in MARL may have its own policy, objective, and reward signal, depending on the scenario.

The type of agents' interactions could be:

- **Cooperative**: Agents work together to achieve a shared goal, such as in team-based tasks or swarm robotics.
- **Competitive**: Agents compete against each other to maximize their own rewards, such as in adversarial games.
- **Mixed**: A combination of cooperation and competition, where agents have both individual and shared objectives.

Overall, in XuanCe, the ``agent`` module contains some key components, including ``policy``, ``learner``, ``memory``, etc.
Each ``agent`` module inherits from the ``core`` module, while the ``core`` module inherits from the ``base`` module.
