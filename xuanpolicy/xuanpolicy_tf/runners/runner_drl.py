from .runner_basic import *
from xuanpolicy.xuanpolicy_tf.agents import get_total_iters
from xuanpolicy.xuanpolicy_tf.representations import REGISTRY as REGISTRY_Representation
from xuanpolicy.xuanpolicy_tf.agents import REGISTRY as REGISTRY_Agent
from xuanpolicy.xuanpolicy_tf.policies import REGISTRY as REGISTRY_Policy
from xuanpolicy.xuanpolicy_tf.utils.input_reformat import get_repre_in, get_policy_in
import tensorflow.keras as tk
import gym.spaces
import numpy as np


class Runner_DRL(Runner_Base):
    def __init__(self, args):
        self.args = args
        self.agent_name = self.args.agent
        self.env_id = self.args.env_id
        super(Runner_DRL, self).__init__(self.args)

        if self.env_id in ['Platform-v0']:
            self.args.observation_space = self.envs.observation_space.spaces[0]
            old_as = self.envs.action_space
            num_disact = old_as.spaces[0].n
            self.args.action_space = gym.spaces.Tuple(
                (old_as.spaces[0], *(gym.spaces.Box(old_as.spaces[1].spaces[i].low,
                                                    old_as.spaces[1].spaces[i].high, dtype=np.float32) for i in
                                     range(0, num_disact))))
        else:
            self.args.observation_space = self.envs.observation_space
            self.args.action_space = self.envs.action_space

        input_representation = get_repre_in(self.args)
        representation = REGISTRY_Representation[self.args.representation](*input_representation)

        input_policy = get_policy_in(self.args, representation)
        policy = REGISTRY_Policy[self.args.policy](*input_policy)

        if self.agent_name in ["DDPG", "TD3", "SAC", "SACDIS"]:
            # actor_lr_scheduler = MyLinearLR(self.args.actor_learning_rate, start_factor=1.0, end_factor=0.25,
            #                                 total_iters=get_total_iters(self.agent_name, self.args))
            actor_lr_scheduler = tk.optimizers.schedules.ExponentialDecay(self.args.actor_learning_rate,
                                                                          decay_steps=1000, decay_rate=0.9)
            actor_optimizer = tk.optimizers.Adam(actor_lr_scheduler)
            # critic_lr_scheduler = MyLinearLR(self.args.critic_learning_rate, start_factor=1.0, end_factor=0.25,
            #                                  total_iters=get_total_iters(self.agent_name, self.args))
            critic_lr_scheduler = tk.optimizers.schedules.ExponentialDecay(self.args.critic_learning_rate,
                                                                           decay_steps=1000, decay_rate=0.9)
            critic_optimizer = tk.optimizers.Adam(critic_lr_scheduler)
            self.agent = REGISTRY_Agent[self.agent_name](self.args, self.envs, policy,
                                                         [actor_optimizer, critic_optimizer], self.args.device)
        elif self.agent_name in ["PDQN", "MPDQN", "SPDQN"]:
            conactor_lr_scheduler = tk.optimizers.schedules.ExponentialDecay(self.args.learning_rate,
                                                                             decay_steps=1000, decay_rate=0.9)
            conactor_optimizer = tk.optimizers.Adam(conactor_lr_scheduler)
            qnetwork_lr_scheduler = tk.optimizers.schedules.ExponentialDecay(self.args.learning_rate,
                                                                             decay_steps=1000, decay_rate=0.9)
            qnetwork_optimizer = tk.optimizers.Adam(qnetwork_lr_scheduler)
            self.agent = REGISTRY_Agent[self.agent_name](self.args, self.envs, policy,
                                                         [conactor_optimizer, qnetwork_optimizer],
                                                         self.args.device)
        else:
            # lr_scheduler = MyLinearLR(self.args.learning_rate, start_factor=1.0, end_factor=0.25,
            #                           total_iters=get_total_iters(self.agent_name, self.args))
            lr_scheduler = tk.optimizers.schedules.ExponentialDecay(self.args.learning_rate, decay_steps=1000,
                                                                    decay_rate=0.9)
            optimizer = tk.optimizers.Adam(lr_scheduler)
            self.agent = REGISTRY_Agent[self.agent_name](self.args, self.envs, policy, optimizer, self.args.device)

    def run(self):
        self.agent.test() if self.args.test_mode else self.agent.train(self.args.training_steps)
