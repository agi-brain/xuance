import numpy as np
from tqdm import tqdm
from argparse import Namespace
from xuance.common import Union, Optional, DummyOffPolicyBuffer, DummyOffPolicyBuffer_Atari
from xuance.environment import DummyVecMultiAgentEnv, SubprocVecMultiAgentEnv
from xuance.tensorflow import tk, Module
from xuance.tensorflow.policies import REGISTRY_Policy
from xuance.tensorflow.agents import OnPolicyMARLAgents, BaseCallback


class MFAC_Agents(MARLAgents):
    """The implementation of Mean Field Actor-Critic (MFAC) agents.

        Args:
            config: the Namespace variable that provides hyperparameters and other settings.
            envs: the vectorized environments.
            callback: A user-defined callback function object to inject custom logic during training.
        """

    def __init__(self,
                 config: Namespace,
                 envs: Union[DummyVecMultiAgentEnv, SubprocVecMultiAgentEnv],
                 callback: Optional[BaseCallback] = None):
        super(MFAC_Agents, self).__init__(config, envs, callback)

        self.n_actions_list = [a_space.n for a_space in self.action_space.values()]
        self.n_actions_max = max(self.n_actions_list)
        self.actions_mean = [{k: np.zeros(self.n_actions_max) for k in self.agent_keys} for _ in range(self.n_envs)]

        self.policy = self._build_policy()  # build policy
        self.memory = self._build_memory()  # build memory
        self.learner = self._build_learner(self.config, self.model_keys, self.agent_keys, self.policy, self.callback)

    def act(self, obs_n, test_mode, act_mean=None, agent_mask=None):
        batch_size = len(obs_n)
        inputs = {"obs": obs_n,
                  "ids": np.tile(np.expand_dims(np.eye(self.n_agents), 0), (batch_size, 1, 1))}
        _, dists = self.policy(inputs)
        acts = dists.stochastic_sample()

        n_alive = np.expand_dims(np.sum(agent_mask, axis=-1), axis=-1).repeat(self.dim_act, axis=1)
        action_n_mask = np.expand_dims(agent_mask, axis=-1).repeat(self.dim_act, axis=-1)
        act_neighbor_onehot = self.learner.onehot_action(acts, self.dim_act).numpy() * action_n_mask
        act_mean_current = np.sum(act_neighbor_onehot, axis=1) / n_alive

        return acts.numpy(), act_mean_current

    def values(self, obs, actions_mean):
        batch_size = len(obs)
        agents_id = np.tile(np.expand_dims(np.eye(self.n_agents), 0), (batch_size, 1, 1))
        agents_id = tf.convert_to_tensor(agents_id, dtype=tf.float32)
        actions_mean = tf.repeat(tf.expand_dims(tf.convert_to_tensor(actions_mean, dtype=tf.float32), 1),
                                 repeats=self.n_agents, axis=1)
        values_n = self.policy.critic(obs, actions_mean, agents_id)
        hidden_states = None
        return hidden_states, values_n.numpy()

    def train(self, i_step, **kwargs):
        if self.memory.full:
            info_train = {}
            indexes = np.arange(self.buffer_size)
            for _ in range(self.n_epochs):
                np.random.shuffle(indexes)
                for start in range(0, self.buffer_size, self.batch_size):
                    end = start + self.batch_size
                    sample_idx = indexes[start:end]
                    sample = self.memory.sample(sample_idx)
                    info_train = self.learner.update(sample)
            self.learner.lr_decay(i_step)
            self.memory.clear()
            return info_train
        else:
            return {}
