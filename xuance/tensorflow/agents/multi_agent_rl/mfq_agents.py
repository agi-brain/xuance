import numpy as np
from tqdm import tqdm
from argparse import Namespace
from xuance.common import Union, Optional, DummyOffPolicyBuffer, DummyOffPolicyBuffer_Atari
from xuance.environment import DummyVecMultiAgentEnv, SubprocVecMultiAgentEnv
from xuance.tensorflow import tk, Module
from xuance.tensorflow.agents import MARLAgents
from xuance.tensorflow.learners import DQN_Learner
from xuance.torch.policies import REGISTRY_Policy
from xuance.torch.agents import OffPolicyMARLAgents, BaseCallback


class MFQ_Agents(MARLAgents):
    """The implementation of Mean-Field Q agents.

        Args:
            config: the Namespace variable that provides hyperparameters and other settings.
            envs: the vectorized environments.
            callback: A user-defined callback function object to inject custom logic during training.
        """

    def __init__(self,
                 config: Namespace,
                 envs: Union[DummyVecMultiAgentEnv, SubprocVecMultiAgentEnv],
                 callback: Optional[BaseCallback] = None):
        super(MFQ_Agents, self).__init__(config, envs, callback)

        self.n_actions_list = [a_space.n for a_space in self.action_space.values()]
        self.n_actions_max = max(self.n_actions_list)
        self.actions_mean = [{k: np.zeros(self.n_actions_max) for k in self.agent_keys} for _ in range(self.n_envs)]

        self.start_greedy, self.end_greedy = config.start_greedy, config.end_greedy
        self.delta_egreedy = (self.start_greedy - self.end_greedy) / config.decay_step_greedy
        self.e_greedy = self.start_greedy

        self.policy = self._build_policy()  # build policy
        self.memory = self._build_memory()  # build memory
        self.learner = self._build_learner(self.config, self.model_keys, self.agent_keys, self.policy, self.callback)

    def act(self, obs_n, *rnn_hidden, test_mode=False, act_mean=None, agent_mask=None, avail_actions=None):
        batch_size = obs_n.shape[0]
        act_mean = np.expand_dims(act_mean, axis=-2).repeat(self.n_agents, axis=1)
        inputs = {"obs": obs_n,
                  "act_mean": act_mean,
                  "ids": np.tile(np.expand_dims(np.eye(self.n_agents), 0), (batch_size, 1, 1))}
        _, greedy_actions, q_output = self.policy(inputs)
        n_alive = np.expand_dims(np.sum(agent_mask, axis=-1), axis=-1).repeat(self.dim_act, axis=1)
        action_n_mask = np.expand_dims(agent_mask, axis=-1).repeat(self.dim_act, axis=-1)
        act_neighbor_sample = self.policy.sample_actions(logits=q_output)
        act_neighbor_onehot = self.learner.onehot_action(act_neighbor_sample, self.dim_act).numpy() * action_n_mask
        act_mean_current = np.sum(act_neighbor_onehot, axis=1) / n_alive
        greedy_actions = greedy_actions.numpy()
        if test_mode:
            return None, greedy_actions, act_mean_current
        else:
            random_actions = np.random.choice(self.dim_act, [self.nenvs, self.n_agents])
            if np.random.rand() < self.egreedy:
                return None, random_actions, act_mean_current
            else:
                return None, greedy_actions, act_mean_current

    def train(self, i_step, n_epochs=1):
        if self.egreedy >= self.end_greedy:
            self.egreedy = self.start_greedy - self.delta_egreedy * i_step
        info_train = {}
        if i_step > self.start_training:
            for i_epoch in range(n_epochs):
                sample = self.memory.sample()
                info_train = self.learner.update(sample)
        info_train["epsilon-greedy"] = self.egreedy
        return info_train
