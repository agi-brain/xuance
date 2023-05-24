from .runner_basic import *
from xuanpolicy.mindspore.agents import get_total_iters
from xuanpolicy.mindspore.representations import REGISTRY as REGISTRY_Representation
from xuanpolicy.mindspore.agents import REGISTRY as REGISTRY_Agent
from xuanpolicy.mindspore.policies import REGISTRY as REGISTRY_Policy
from xuanpolicy.mindspore.utils.input_reformat import get_repre_in, get_policy_in
import itertools
from mindspore.nn import Adam
from mindspore.nn.learning_rate_schedule import ExponentialDecayLR as lr_decay_model
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
            actor_lr_scheduler = lr_decay_model(learning_rate=self.args.actor_learning_rate,
                                                decay_rate=0.5,
                                                decay_steps=get_total_iters(self.agent_name, self.args))
            critic_lr_scheduler = lr_decay_model(learning_rate=self.args.critic_learning_rate,
                                                 decay_rate=0.5,
                                                 decay_steps=get_total_iters(self.agent_name, self.args))
            actor_optimizer = Adam(policy.actor.trainable_params(), actor_lr_scheduler, eps=1e-5)
            if self.agent_name == "TD3":
                critic_optimizer = Adam(itertools.chain(policy.criticA.trainable_params(),
                                                        policy.criticB.trainable_params()),
                                        critic_lr_scheduler, eps=1e-5)
            else:
                critic_optimizer = Adam(policy.critic.trainable_params(), critic_lr_scheduler, eps=1e-5)
            self.agent = REGISTRY_Agent[self.agent_name](self.args, self.envs, policy,
                                                         {'actor': actor_optimizer, 'critic': critic_optimizer},
                                                         {'actor': actor_lr_scheduler, 'critic': critic_lr_scheduler})
        elif self.agent_name in ["PDQN", "MPDQN", "SPDQN"]:
            conactor_lr_scheduler = lr_decay_model(learning_rate=self.args.learning_rate,
                                                decay_rate=0.5,
                                                decay_steps=get_total_iters(self.agent_name, self.args))
            qnetwork_lr_scheduler = lr_decay_model(learning_rate=self.args.learning_rate,
                                                 decay_rate=0.5,
                                                 decay_steps=get_total_iters(self.agent_name, self.args))
            conactor_optimizer = Adam(policy.conactor.trainable_params(), conactor_lr_scheduler, eps=1e-5)
            qnetwork_optimizer = Adam(policy.qnetwork.trainable_params(), qnetwork_lr_scheduler, eps=1e-5)
            self.agent = REGISTRY_Agent[self.agent_name](self.args, self.envs, policy,
                                                         [conactor_optimizer, qnetwork_optimizer],
                                                         [conactor_lr_scheduler, qnetwork_lr_scheduler])
        else:
            lr_scheduler = lr_decay_model(learning_rate=self.args.learning_rate,
                                          decay_rate=0.5,
                                          decay_steps=get_total_iters(self.agent_name, self.args)
                                          )
            optimizer = Adam(policy.trainable_params(), lr_scheduler, eps=1e-5)
            self.agent = REGISTRY_Agent[self.agent_name](self.args, self.envs, policy, optimizer, lr_scheduler)

    def run(self):
        self.agent.test(self.args.test_steps) if self.args.test_mode else self.agent.train(self.args.training_steps)
        self.envs.close()
