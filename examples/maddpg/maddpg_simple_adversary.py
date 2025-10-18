#!/usr/bin/env python3
"""
MADDPG Simple Adversary with Configurable Good Agents Parameter Sharing

This example demonstrates how to implement configurable parameter sharing for good agents
in the MADDPG adversarial environment. It provides the ability to switch between
parameter sharing and independent parameters for good agents through YAML configuration files.

Key Features:
- Configurable good agents parameter sharing via YAML files
- Maintains adversary agents independence
- Solves gradient conflicts in parameter sharing mode through optimizer restructuring
- Compatible with xuance MADDPG framework


"""

import numpy as np
import torch
import torch.nn as nn
from copy import deepcopy
from argparse import Namespace
from xuance.common import get_configs
from xuance.environment import make_envs
from xuance.torch.agents import REGISTRY_Agents
from xuance.torch.learners.multi_agent_rl.maddpg_learner import MADDPG_Learner
from xuance.torch.agents.multi_agent_rl.maddpg_agents import MADDPG_Agents


class ConfigurableParameterSharingLearner(MADDPG_Learner):
    """
    Enhanced MADDPG Learner with configurable good agents parameter sharing.

    This learner extends the standard MADDPG learner to support configurable
    parameter sharing for good agents while maintaining adversary independence.
    It solves gradient conflicts in shared parameter mode by restructuring
    the optimizer architecture.

    Key Features:
    - Reads 'good_agents_parameter_sharing' from config to control sharing behavior
    - Automatically identifies good agents (containing 'agent' but not 'adversary')
    - Creates shared optimizers for good agents when sharing is enabled
    - Maintains separate optimizers for adversary agents
    - Prevents gradient conflicts through careful optimizer management

    Args:
        config: Configuration object containing training parameters
        model_keys: List of model keys for different agents
        agent_keys: List of agent keys
        policy: Policy object containing actor/critic networks
        callback: Callback function for logging
    """

    def __init__(self, config, model_keys, agent_keys, policy, callback):
        """
        Initialize the configurable parameter sharing learner.

        Args:
            config: Configuration object with training parameters
            model_keys: List of model keys for different agents
            agent_keys: List of agent keys
            policy: Policy object containing actor/critic networks
            callback: Callback function for logging
        """
        super(ConfigurableParameterSharingLearner, self).__init__(
            config, model_keys, agent_keys, policy, callback
        )

        # Identify good agents vs adversary agents
        self.good_agents = [
            key for key in agent_keys
            if 'agent' in key and 'adversary' not in key
        ]
        self.adversary_agents = [
            key for key in agent_keys
            if 'adversary' in key
        ]

        # Read parameter sharing configuration
        self.parameter_sharing_enabled = getattr(
            config, 'good_agents_parameter_sharing', True
        )

        print(f"🔧 Enhanced MADDPG Learner initialized")
        print(f"Good agents: {self.good_agents}")
        print(f"Adversary agents: {self.adversary_agents}")
        print(f"Good agents parameter sharing: {self.parameter_sharing_enabled}")

        # Check if good agents actually share parameters
        if len(self.good_agents) > 1:
            actor_id_0 = id(policy.actor[self.good_agents[0]])
            actor_id_1 = id(policy.actor[self.good_agents[1]])
            self.good_agents_share_params = (actor_id_0 == actor_id_1)
            print(f"Actual parameter sharing status: {self.good_agents_share_params}")

            if self.good_agents_share_params:
                self._rebuild_optimizers_for_sharing()
        else:
            self.good_agents_share_params = False

    def _rebuild_optimizers_for_sharing(self):
        """
        Rebuild optimizers to support parameter sharing without gradient conflicts.

        This method creates a single shared optimizer for all good agents when
        parameter sharing is enabled, while maintaining separate optimizers
        for adversary agents. This prevents gradient conflicts that occur when
        multiple agents try to update the same shared parameters simultaneously.

        The key insight is that shared parameters should only have one optimizer
        instance, even if multiple agents reference the same network.

        Note:
            This method modifies self.optimizer and self.scheduler dictionaries
            to ensure proper parameter sharing without conflicts.
        """
        print("🔧 Rebuilding optimizers for parameter sharing...")

        lr_actor = self.config.learning_rate_actor
        lr_critic = self.config.learning_rate_critic

        self.optimizer = {}
        self.scheduler = {}

        # Create independent optimizers for adversary agents
        for key in self.adversary_agents:
            if key in self.model_keys:
                self.optimizer[key] = {
                    'actor': torch.optim.Adam(
                        self.policy.parameters_actor[key], lr_actor, eps=1e-5
                    ),
                    'critic': torch.optim.Adam(
                        self.policy.parameters_critic[key], lr_critic, eps=1e-5
                    )
                }
                self.scheduler[key] = {
                    'actor': torch.optim.lr_scheduler.LinearLR(
                        self.optimizer[key]['actor'],
                        start_factor=1.0,
                        end_factor=self.end_factor_lr_decay,
                        total_iters=self.config.running_steps
                    ),
                    'critic': torch.optim.lr_scheduler.LinearLR(
                        self.optimizer[key]['critic'],
                        start_factor=1.0,
                        end_factor=self.end_factor_lr_decay,
                        total_iters=self.config.running_steps
                    )
                }

        # Create shared optimizers for good agents
        if self.good_agents:
            shared_key = self.good_agents[0]

            shared_actor_params = list(
                self.policy.actor[shared_key].parameters()
            )
            shared_critic_params = list(
                self.policy.critic[shared_key].parameters()
            )

            shared_optimizer = {
                'actor': torch.optim.Adam(
                    shared_actor_params, lr_actor, eps=1e-5
                ),
                'critic': torch.optim.Adam(
                    shared_critic_params, lr_critic, eps=1e-5
                )
            }

            shared_scheduler = {
                'actor': torch.optim.lr_scheduler.LinearLR(
                    shared_optimizer['actor'],
                    start_factor=1.0,
                    end_factor=self.end_factor_lr_decay,
                    total_iters=self.config.running_steps
                ),
                'critic': torch.optim.lr_scheduler.LinearLR(
                    shared_optimizer['critic'],
                    start_factor=1.0,
                    end_factor=self.end_factor_lr_decay,
                    total_iters=self.config.running_steps
                )
            }

            # Assign same optimizer reference to all good agents
            for key in self.good_agents:
                if key in self.model_keys:
                    self.optimizer[key] = shared_optimizer
                    self.scheduler[key] = shared_scheduler

        print("✅ Optimizer restructuring completed")

    def update(self, sample):
        """
        Update agents using the modified optimizer structure.

        This method performs the standard MADDPG update but with careful
        handling of shared optimizers to prevent duplicate updates.

        Args:
            sample: Training batch containing observations, actions, rewards, etc.

        Returns:
            dict: Dictionary containing training metrics and losses
        """
        self.iterations += 1

        sample_tensor = self.build_training_data(
            sample,
            use_parameter_sharing=self.use_parameter_sharing,
            use_actions_mask=False
        )
        batch_size = sample_tensor['batch_size']
        obs = sample_tensor['obs']
        actions = sample_tensor['actions']
        obs_next = sample_tensor['obs_next']
        rewards = sample_tensor['rewards']
        terminals = sample_tensor['terminals']
        agent_mask = sample_tensor['agent_mask']
        agent_ids = sample_tensor['agent_ids']

        if self.use_parameter_sharing:
            key = self.model_keys[0]
            bs = batch_size * self.n_agents
            obs_joint = obs[key].reshape(batch_size, -1)
            next_obs_joint = obs_next[key].reshape(batch_size, -1)
            actions_joint = actions[key].reshape(batch_size, -1)
            rewards[key] = rewards[key].reshape(batch_size * self.n_agents)
            terminals[key] = terminals[key].reshape(batch_size * self.n_agents)
        else:
            bs = batch_size
            obs_joint = self.get_joint_input(obs, (batch_size, -1))
            next_obs_joint = self.get_joint_input(obs_next, (batch_size, -1))
            actions_joint = self.get_joint_input(actions, (batch_size, -1))

        _, actions_eval = self.policy(observation=obs, agent_ids=agent_ids)
        _, actions_next = self.policy.Atarget(next_observation=obs_next, agent_ids=agent_ids)

        if self.use_parameter_sharing:
            key = self.model_keys[0]
            actions_next_joint = actions_next[key].reshape(
                batch_size, self.n_agents, -1
            ).reshape(batch_size, -1)
        else:
            actions_next_joint = self.get_joint_input(actions_next, (batch_size, -1))

        _, q_eval = self.policy.Qpolicy(
            joint_observation=obs_joint,
            joint_actions=actions_joint,
            agent_ids=agent_ids
        )
        _, q_next = self.policy.Qtarget(
            joint_observation=next_obs_joint,
            joint_actions=actions_next_joint,
            agent_ids=agent_ids
        )

        info = {}

        # Track updated optimizers to avoid duplicate updates
        updated_optimizers = set()

        for key in self.model_keys:
            mask_values = agent_mask[key]

            # Update critic
            q_eval_a = q_eval[key].reshape(bs)
            q_next_i = q_next[key].reshape(bs)
            q_target = rewards[key] + (1 - terminals[key]) * self.gamma * q_next_i
            td_error = (q_eval_a - q_target.detach()) * mask_values
            loss_c = (td_error ** 2).sum() / mask_values.sum()

            critic_optimizer_id = id(self.optimizer[key]['critic'])
            if critic_optimizer_id not in updated_optimizers:
                self.optimizer[key]['critic'].zero_grad()
                loss_c.backward(retain_graph=True)
                if self.use_grad_clip:
                    torch.nn.utils.clip_grad_norm_(
                    self.policy.parameters_critic[key], self.grad_clip_norm
                )
                self.optimizer[key]['critic'].step()
                if self.scheduler[key]['critic'] is not None:
                    self.scheduler[key]['critic'].step()
                updated_optimizers.add(critic_optimizer_id)

            # Update actor
            if self.use_parameter_sharing:
                act_eval = actions_eval[key].reshape(
                    batch_size, self.n_agents, -1
                ).reshape(batch_size, -1)
            else:
                a_joint = {
                    k: actions_eval[k] if k == key else actions[k]
                    for k in self.agent_keys
                }
                act_eval = self.get_joint_input(a_joint, (batch_size, -1))

            _, q_policy = self.policy.Qpolicy(
                joint_observation=obs_joint,
                joint_actions=act_eval,
                agent_ids=agent_ids,
                agent_key=key
            )
            q_policy_i = q_policy[key].reshape(bs)
            loss_a = -(q_policy_i * mask_values).sum() / mask_values.sum()

            actor_optimizer_id = id(self.optimizer[key]['actor'])
            if actor_optimizer_id not in updated_optimizers:
                self.optimizer[key]['actor'].zero_grad()
                loss_a.backward()
                if self.use_grad_clip:
                    torch.nn.utils.clip_grad_norm_(
                        self.policy.parameters_actor[key], self.grad_clip_norm
                    )
                self.optimizer[key]['actor'].step()
                if self.scheduler[key]['actor'] is not None:
                    self.scheduler[key]['actor'].step()
                updated_optimizers.add(actor_optimizer_id)

            learning_rate_actor = self.optimizer[key]['actor'].state_dict()[
                'param_groups'
            ][0]['lr']
            learning_rate_critic = self.optimizer[key]['critic'].state_dict()[
                'param_groups'
            ][0]['lr']

            info.update({
                f"{key}/learning_rate_actor": learning_rate_actor,
                f"{key}/learning_rate_critic": learning_rate_critic,
                f"{key}/loss_actor": loss_a.item(),
                f"{key}/loss_critic": loss_c.item(),
                f"{key}/predictQ": q_eval[key].mean().item()
            })

        self.policy.soft_update(self.tau)
        return info


class ConfigurableParameterSharingAgents(MADDPG_Agents):
    """
    Enhanced MADDPG Agents with configurable good agents parameter sharing.

    This class extends the standard MADDPG agents to support configurable
    parameter sharing for good agents through YAML configuration files.
    It automatically sets up parameter sharing or independence based on
    the 'good_agents_parameter_sharing' configuration parameter.

    Key Features:
    - Reads configuration to determine sharing behavior
    - Automatically identifies and groups good vs adversary agents
    - Sets up shared networks for good agents when enabled
    - Maintains independent networks for adversary agents
    - Provides verification of parameter sharing/independence

    Configuration Parameters:
    - good_agents_parameter_sharing (bool): Whether good agents share parameters
    """

    def _build_policy(self):
        """
        Build policy and configure parameter sharing based on configuration.

        This method builds the standard MADDPG policy and then configures
        parameter sharing for good agents based on the configuration file.

        Returns:
            Policy object with configured parameter sharing
        """
        policy = super()._build_policy()

        parameter_sharing_enabled = getattr(
            self.config, 'good_agents_parameter_sharing', True
        )

        if parameter_sharing_enabled:
            self._setup_good_agent_sharing(policy)
        else:
            print("🔧 Good agents using independent parameters")
            self._verify_independent_parameters(policy)

        return policy

    def _setup_good_agent_sharing(self, policy):
        """
        Set up parameter sharing for good agents.

        This method configures all good agents to share the same network
        instances for both actor and critic networks, enabling parameter
        sharing while keeping adversary agents independent.

        Args:
            policy: Policy object containing all agent networks
        """
        print("🔧 Setting up good agents parameter sharing...")

        all_keys = list(policy.actor.keys())
        good_agents = [
            key for key in all_keys
            if 'agent' in key and 'adversary' not in key
        ]
        adversary_agents = [
            key for key in all_keys
            if 'adversary' in key
        ]

        if len(good_agents) > 1:
            shared_actor = policy.actor[good_agents[0]]
            shared_critic = policy.critic[good_agents[0]]
            shared_target_actor = policy.target_actor[good_agents[0]]
            shared_target_critic = policy.target_critic[good_agents[0]]

            for agent in good_agents:
                policy.actor[agent] = shared_actor
                policy.critic[agent] = shared_critic
                policy.target_actor[agent] = shared_target_actor
                policy.target_critic[agent] = shared_target_critic

            print("✅ Good agents parameter sharing setup completed")

    def _verify_independent_parameters(self, policy):
        """
        Verify that good agents have independent parameters.

        This method checks that all good agents have separate network
        instances, ensuring parameter independence.

        Args:
            policy: Policy object containing all agent networks
        """
        all_keys = list(policy.actor.keys())
        good_agents = [
            key for key in all_keys
            if 'agent' in key and 'adversary' not in key
        ]

        if len(good_agents) > 1:
            actor_ids = {key: id(policy.actor[key]) for key in good_agents}
            unique_ids = set(actor_ids.values())
            if len(unique_ids) == len(good_agents):
                print("✅ Good agents parameter independence verified")
            else:
                print("❌ Good agents parameter independence verification failed")

    def _build_learner(self, config, model_keys, agent_keys, policy, *args, **kwargs):
        """
        Build the enhanced learner with parameter sharing support.

        Returns:
            ConfigurableParameterSharingLearner: Enhanced learner instance
        """
        return ConfigurableParameterSharingLearner(
            config, model_keys, agent_keys, policy, *args, **kwargs
        )


def main():
    """
    Main function demonstrating configurable parameter sharing in MADDPG.

    This function loads configuration from YAML files and demonstrates how to
    use the enhanced MADDPG implementation with configurable good agents
    parameter sharing. It supports both training and testing modes.

    The configuration file determines whether good agents share parameters
    through the 'good_agents_parameter_sharing' setting.
    """
    print("🎯 Configurable Good Agents Parameter Sharing MADDPG Solution")
    print("Supports switching between parameter sharing/independent modes via YAML config")

    config = get_configs(file_dir="maddpg_mpe_configs/simple_adversary_v3_sharing.yaml")
    config = Namespace(**config)

    config.agent = "MADDPG"
    config.policy = "MADDPG_Policy"
    config.use_parameter_sharing = False

    envs = make_envs(config)
    agents = ConfigurableParameterSharingAgents(config, envs)

    print(f"Environment: {config.env_id}")
    print(f"Agents: {list(envs.observation_space.keys())}")

    if not config.test_mode:
        print(f"\n🚀 Starting training...")
        train_steps = config.running_steps // config.parallels
        print(
            f"Training steps: {train_steps:,} "
            f"({config.running_steps:,} total / {config.parallels} parallel envs)"
        )

        try:
            agents.train(train_steps)
            print("✅ Training completed successfully!")
            agents.save_model("maddpg_configurable_sharing_model.pth")
            print("✅ Model saved")

        except Exception as e:
            print(f"❌ Training failed: {e}")

    else:
        print(f"\n🧪 Starting testing...")
        def env_fn():
            """Create test environment with specified episode count."""
            config.parallels = config.test_episode
            return make_envs(config)

        try:
            agents.load_model(path=agents.model_dir_load)
            scores = agents.test(env_fn, config.test_episode)

            print(f"\n📊 Test Results:")
            print(f"Average Score: {np.array(scores).mean():.3f}")
            print(f"Standard Deviation: {np.array(scores).std():.3f}")
        except Exception as e:
            print(f"❌ Testing failed: {e}")

    agents.finish()
    envs.close()
    print("🏁 Program finished")


if __name__ == '__main__':
    main()
