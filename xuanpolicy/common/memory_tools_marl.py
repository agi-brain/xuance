import numpy as np
from abc import ABC, abstractmethod
from xuanpolicy.common.common_tools import discount_cumsum


class BaseBuffer(ABC):
    def __init__(self, obs_space, act_space, rew_space, n_envs, buffer_size, n_minibatch):
        self.obs_space = obs_space
        self.act_space = act_space
        self.rew_space = rew_space
        self.n_envs = n_envs
        self.buffer_size = buffer_size
        self.n_minibatch = n_minibatch
        self.batch_size = n_minibatch
        self.ptr = 0  # last data pointer
        self.size = 0  # current buffer size

    @property
    def full(self):
        return self.size >= self.buffer_size

    @abstractmethod
    def store(self, *args):
        raise NotImplementedError

    def clear(self, *args):
        raise NotImplementedError

    def can_sample(self, *args):
        raise NotImplementedError

    @abstractmethod
    def sample(self, *args):
        raise NotImplementedError

    def finish_ac_path(self, value, i_env):
        return


class CidPreTrainBuffer(ABC):
    def __init__(self, state_space, act_space, rew_space, n_envs, buffer_size, batch_size):
        self.state_space = state_space
        self.act_space = act_space
        self.rew_space = rew_space
        self.n_envs = n_envs
        self.buffer_size = buffer_size
        self.n_minibatch = batch_size
        self.batch_size = batch_size
        self.ptr = 0  # last data pointer
        self.size = 0  # current buffer size

        self.data = {
            'state': np.zeros((self.buffer_size,) + state_space).astype(np.float32),
            'actions': np.zeros((self.buffer_size,) + act_space).astype(np.float32),
            'rewards': np.zeros((self.buffer_size,) + rew_space).astype(np.float32),
        }

        self.keys = self.data.keys()

    def full(self):
        return self.size >= self.buffer_size

    def store(self, step_data):
        ptr_end = self.ptr + self.n_envs

        for k in self.keys:
            self.data[k][self.ptr: ptr_end] = step_data[k]

        self.ptr = (ptr_end) % self.buffer_size
        self.size = np.min([self.size + 1, self.buffer_size])

    def can_sample(self, batch_size):
        return self.size >= batch_size

    def sample(self):
        assert self.can_sample(self.batch_size)
        random_batch_index = np.random.choice(self.size, size=self.batch_size, replace=False)
        samples = {k: self.data[k][random_batch_index] for k in self.keys}
        return samples


class MARL_OffPolicyBuffer(BaseBuffer, ABC):
    def __init__(self, state_space, obs_space, act_space, rew_space, done_space, n_envs, buffer_size, batch_size):
        super(MARL_OffPolicyBuffer, self).__init__(obs_space, act_space, rew_space, n_envs, buffer_size, batch_size)
        self.state_space = state_space
        self.buffer_size = buffer_size * n_envs
        self.env_buffer_size = buffer_size
        self.n_agents = act_space[0]

        self.data = {
            'obs': np.zeros((self.buffer_size,) + obs_space).astype(np.float32),
            'actions': np.zeros((self.buffer_size,) + act_space).astype(np.float32),
            'obs_next': np.zeros((self.buffer_size,) + obs_space).astype(np.float32),
            'rewards': np.zeros((self.buffer_size,) + rew_space).astype(np.float32),
            'terminals': np.zeros((self.buffer_size,) + done_space).astype(np.bool),
            'agent_mask': np.ones((self.buffer_size, self.n_agents)).astype(np.bool)
        }

        if state_space is not None:
            self.data.update({'state': np.zeros((self.buffer_size,) + state_space).astype(np.float32),
                              'state_next': np.zeros((self.buffer_size,) + state_space).astype(np.float32)})
        self.keys = self.data.keys()

    def store(self, step_data):
        ptr_end = self.ptr + self.n_envs

        for k in self.keys:
            self.data[k][self.ptr: ptr_end] = step_data[k]

        self.ptr = (ptr_end) % self.buffer_size
        self.size = np.min([self.size + 1, self.buffer_size])

    def can_sample(self, batch_size):
        return self.size >= batch_size

    def sample(self):
        assert self.can_sample(self.batch_size)

        random_batch_index = np.random.choice(self.size, size=self.batch_size, replace=False)
        samples = {k: self.data[k][random_batch_index] for k in self.keys}
        samples.update({'batch_size': self.batch_size})
        return samples


class MeanField_OffPolicyBuffer(MARL_OffPolicyBuffer):
    def __init__(self, state_space, obs_space, act_space, prob_shape, rew_space, done_space, n_envs, buffer_size,
                 batch_size):
        super(MeanField_OffPolicyBuffer, self).__init__(state_space, obs_space, act_space, rew_space, done_space,
                                                        n_envs, buffer_size, batch_size)
        self.prob_shape = prob_shape
        self.data.update({"act_mean": np.zeros((self.buffer_size,) + prob_shape).astype(np.float32)})
        self.keys = self.data.keys()

    def sample(self):
        assert self.can_sample(self.batch_size)

        random_batch_index = np.random.choice(self.size, size=self.batch_size, replace=False)
        samples = {k: self.data[k][random_batch_index] for k in self.keys}
        samples.update({'batch_size': self.batch_size})

        next_index = (random_batch_index + 1) % self.size
        samples.update({'act_mean_next': self.data['act_mean'][next_index]})

        return samples


class CID_Buffer_OffPolicy(MARL_OffPolicyBuffer):
    def __init__(self, state_space, obs_space, act_space, rew_space, done_space, n_envs, buffer_size, batch_size):
        super(CID_Buffer_OffPolicy, self).__init__(state_space, obs_space, act_space, rew_space, done_space, n_envs,
                                                   buffer_size, batch_size)
        self.data.update({"rewards_assign": np.zeros((self.buffer_size,) + rew_space).astype(np.float32)})
        self.keys = self.data.keys()


class MARL_OnPolicyBuffer(BaseBuffer, ABC):
    def __init__(self, state_space, obs_space, act_space, rew_space, done_space, n_envs,
                 n_steps, n_minibatch, use_gae=True, use_advnorm=False, gamma=0.99, lam=0.95):
        super(MARL_OnPolicyBuffer, self).__init__(obs_space, act_space, rew_space, n_envs, n_steps, n_minibatch)
        self.state_space = state_space
        self.done_space = done_space
        self.n_steps = n_steps
        self.n_agents = act_space[0]
        self.use_gae = use_gae
        self.use_advantage_norm = use_advnorm
        self.gamma, self.gae_lambda = gamma, lam
        self.env_batch_size = self.n_steps // self.n_minibatch
        self.batch_size = self.n_envs * self.n_steps // self.n_minibatch

        self.start_ids = np.zeros(self.n_envs)  # the start index of the last episode for each env.

        self.data = {}
        self.clear()
        self.keys = self.data.keys()
        self.data_shapes = {k: self.data[k].shape for k in self.keys}

    def clear(self):
        self.data.update({
            'obs': np.zeros((self.n_envs, self.n_steps,) + self.obs_space).astype(np.float32),
            'state': np.zeros((self.n_envs, self.n_steps,) + self.state_space).astype(np.float32),
            'actions': np.zeros((self.n_envs, self.n_steps,) + self.act_space).astype(np.float32),
            'rewards': np.zeros((self.n_envs, self.n_steps,) + self.rew_space).astype(np.float32),
            'values': np.zeros((self.n_envs, self.n_steps,) + self.rew_space).astype(np.float32),
            'log_pi_old': np.zeros((self.n_envs, self.n_steps, self.n_agents,)).astype(np.float32),
            'pi_dist_old': np.zeros((self.n_envs, self.n_steps, self.n_agents,)).astype(np.object),
            'advantages': np.zeros((self.n_envs, self.n_steps,) + self.rew_space).astype(np.float32),
            'terminals': np.zeros((self.n_envs, self.n_steps,) + self.done_space).astype(np.bool),
            'agent_mask': np.ones((self.n_envs, self.n_steps, self.n_agents)).astype(np.bool),
        })

        self.ptr = 0  # current pointer
        self.size = 0  # current buffer size
        self.start_ids = np.zeros(self.n_envs)

    def store(self, step_data):
        for k in self.keys:
            if k == "advantages": continue
            if k in step_data.keys():
                if k == "pi_dist_old":
                    self.data[k][:, self.ptr] = np.array(step_data[k])
                else:
                    self.data[k][:, self.ptr] = step_data[k]

        self.ptr = (self.ptr + 1) % self.n_steps
        self.size = np.min([self.size + 1, self.n_steps])

    def finish_ac_path(self, value, i_env):  # when an episode is finished
        if self.size == 0:
            return
        end_id = self.n_steps if (self.ptr == 0) else self.ptr
        path_slice = np.arange(self.start_ids[i_env], end_id).astype(np.int32)
        if self.full:
            path_slice = np.arange(self.start_ids[i_env], self.buffer_size).astype(np.int32)
        rewards = np.append(np.array(self.data['rewards'][i_env, path_slice]), [value], axis=0)
        returns = np.append(np.array(self.data['values'][i_env, path_slice]), [value], axis=0)
        deltas = rewards[:-1] + self.gamma * returns[1:] - returns[:-1]
        advantages = discount_cumsum(deltas, self.gamma * self.gae_lambda) if self.use_gae else deltas
        returns = discount_cumsum(rewards, self.gamma)[:-1]
        self.data['values'][i_env, path_slice] = returns
        self.data['advantages'][i_env, path_slice] = advantages
        self.start_ids[i_env] = self.ptr

    def can_sample(self):
        return self.size >= self.batch_size

    def sample(self):
        random_env_index = np.random.choice(self.n_envs, size=self.batch_size)
        random_step_index = np.random.choice(self.n_steps, size=self.batch_size)
        samples = {}
        for k in self.keys:
            samples[k] = self.data[k][random_env_index, random_step_index]
        samples.update({'batch_size': self.batch_size})

        return samples


class MARL_OnPolicyBuffer_MindSpore(MARL_OnPolicyBuffer):
    def __init__(self, state_space, obs_space, act_space, rew_space, done_space, n_envs,
                 n_steps, n_minibatch, use_gae=True, use_advnorm=False, gamma=0.99, lam=0.95, n_actions=None):
        self.n_actions = n_actions
        super(MARL_OnPolicyBuffer_MindSpore, self).__init__(state_space, obs_space, act_space, rew_space, done_space,
                                                            n_envs, n_steps, n_minibatch, use_gae, use_advnorm, gamma,
                                                            lam)
        self.keys = self.data.keys()
        self.data_shapes = {k: self.data[k].shape for k in self.keys}

    def clear(self):
        self.data.update({
            'obs': np.zeros((self.n_envs, self.n_steps,) + self.obs_space).astype(np.float32),
            'state': np.zeros((self.n_envs, self.n_steps,) + self.state_space).astype(np.float32),
            'actions': np.zeros((self.n_envs, self.n_steps,) + self.act_space).astype(np.float32),
            'rewards': np.zeros((self.n_envs, self.n_steps,) + self.rew_space).astype(np.float32),
            'values': np.zeros((self.n_envs, self.n_steps,) + self.rew_space).astype(np.float32),
            'log_pi_old': np.zeros((self.n_envs, self.n_steps, self.n_agents,)).astype(np.float32),
            'act_prob_old': np.zeros((self.n_envs, self.n_steps, self.n_agents, self.n_actions)).astype(np.float32),
            'advantages': np.zeros((self.n_envs, self.n_steps,) + self.rew_space).astype(np.float32),
            'terminals': np.zeros((self.n_envs, self.n_steps,) + self.done_space).astype(np.bool),
            'agent_mask': np.ones((self.n_envs, self.n_steps, self.n_agents)).astype(np.bool),
        })

        self.ptr = 0  # current pointer
        self.size = 0  # current buffer size
        self.start_ids = np.zeros(self.n_envs)

    def store(self, step_data):
        for k in self.keys:
            if k == "advantages": continue
            if k in step_data.keys():
                self.data[k][:, self.ptr] = step_data[k]

        self.ptr = (self.ptr + 1) % self.n_steps
        self.size = np.min([self.size + 1, self.n_steps])


class MeanField_OnPolicyBuffer(MARL_OnPolicyBuffer):
    def __init__(self, state_space, obs_space, act_space, prob_space, rew_space, done_space, n_envs,
                 n_steps, n_minibatch, use_gae=True, use_advnorm=False, gamma=0.99, lam=0.95):
        self.prob_space = prob_space
        super(MeanField_OnPolicyBuffer, self).__init__(state_space, obs_space, act_space, rew_space, done_space, n_envs,
                                                       n_steps, n_minibatch, use_gae, use_advnorm, gamma, lam)

    def clear(self):
        self.data.update({
            'obs': np.zeros((self.n_envs, self.n_steps,) + self.obs_space).astype(np.float32),
            'obs_next': np.zeros((self.n_envs, self.n_steps,) + self.obs_space).astype(np.float32),
            'state': np.zeros((self.n_envs, self.n_steps,) + self.state_space).astype(np.float32),
            'state_next': np.zeros((self.n_envs, self.n_steps,) + self.state_space).astype(np.float32),
            'actions': np.zeros((self.n_envs, self.n_steps,) + self.act_space).astype(np.float32),
            'act_mean': np.zeros((self.n_envs, self.n_steps,) + self.prob_space).astype(np.float32),
            'rewards': np.zeros((self.n_envs, self.n_steps,) + self.rew_space).astype(np.float32),
            'terminals': np.zeros((self.n_envs, self.n_steps,) + self.done_space).astype(np.bool),
            'agent_mask': np.ones((self.n_envs, self.n_steps, self.n_agents)).astype(np.bool),
        })

        self.ptr = 0  # current pointer
        self.size = 0  # current buffer size
        self.start_ids = np.zeros(self.n_envs)

    def finish_ac_path(self, value, i_env):  # when an episode is finished
        if self.size == 0:
            return
        self.start_ids[i_env] = self.ptr


class COMA_Buffer(BaseBuffer, ABC):
    def __init__(self, state_space, obs_space, act_space, act_onehot_space, rew_space, done_space,
                 n_envs, buffer_size, batch_size, max_seq_length):
        super(COMA_Buffer, self).__init__(obs_space, act_space, rew_space, n_envs, buffer_size, batch_size)
        self.state_space = state_space
        self.act_onehot_space = act_onehot_space
        self.done_space = done_space
        self.max_seq_len = max_seq_length
        self.n_agents = act_space[0]
        self.batch_size = batch_size
        self.buffer_size_env = self.buffer_size
        self.buffer_size = self.buffer_size_env * self.n_envs

        self.end_ids = np.zeros(self.n_envs)  # the end index of the episode for each env.

        self.data = {}
        self.clear()
        self.keys = self.data.keys()
        self.data_shapes = {k: self.data[k].shape for k in self.keys}

    def clear(self):
        self.data.update({
            'obs': np.zeros((self.n_envs, self.buffer_size_env, self.max_seq_len,) + self.obs_space).astype(np.float32),
            'state': np.zeros((self.n_envs, self.buffer_size_env, self.max_seq_len,) + self.state_space).astype(
                np.float32),
            'actions': np.zeros((self.n_envs, self.buffer_size_env, self.max_seq_len,) + self.act_space).astype(
                np.float32),
            'actions_onehot': np.zeros(
                (self.n_envs, self.buffer_size_env, self.max_seq_len,) + self.act_onehot_space).astype(np.float32),
            'rewards': np.zeros((self.n_envs, self.buffer_size_env, self.max_seq_len,) + self.rew_space).astype(
                np.float32),
            'terminals': np.zeros((self.n_envs, self.buffer_size_env, self.max_seq_len,) + self.done_space).astype(
                np.bool),
            'agent_mask': np.ones((self.n_envs, self.buffer_size_env, self.max_seq_len, self.n_agents)).astype(np.bool),
        })

        self.buffer_ptrs = np.array(np.zeros(self.n_envs, np.int))
        self.step_ptr = 0  # current pointer
        self.size = 0  # current buffer size
        self.end_ids = np.array(np.zeros(self.n_envs, np.int))
        self.env_index = np.arange(self.n_envs)

    def store(self, step_data):
        for k in self.keys:
            self.data[k][self.env_index, self.buffer_ptrs, self.step_ptr] = step_data[k]
        self.step_ptr = (self.step_ptr + 1) % self.max_seq_len

    def finish_ac_path(self, value=None, i_env=None):
        self.buffer_ptrs[i_env] = (self.buffer_ptrs[i_env] + 1) % self.buffer_size_env
        self.end_ids[i_env] = self.max_seq_len if self.step_ptr == 0 else self.step_ptr
        self.size = np.min([self.size + 1, self.buffer_size])

    def can_sample(self):
        return self.size >= self.batch_size

    def sample(self):
        random_env_index = np.random.choice(self.n_envs, size=self.batch_size)
        random_buffer_index = np.random.choice(self.buffer_size_env, size=self.batch_size)
        samples = {k: self.data[k][random_env_index, random_buffer_index] for k in self.keys}
        samples.update({'batch_size': self.batch_size})
        return samples


class CID_Buffer(MARL_OnPolicyBuffer):
    def __init__(self, state_space, obs_space, act_space, act_prob_space, rew_space, done_space, n_envs,
                 n_steps, n_minibatch, use_gae=True, use_advnorm=False, gamma=0.99, lam=0.95):
        super(CID_Buffer, self).__init__(state_space, obs_space, act_space, act_prob_space, rew_space, done_space,
                                         n_envs,
                                         n_steps, n_minibatch, use_gae, use_advnorm, gamma, lam)
        self.data = {}
        self.clear()
        self.keys = self.data.keys()
        self.data_shapes = {k: self.data[k].shape for k in self.keys}

    def clear(self):
        self.data.update({
            'obs': np.zeros((self.n_envs, self.n_steps,) + self.obs_space).astype(np.float32),
            'state': np.zeros((self.n_envs, self.n_steps,) + self.state_space).astype(np.float32),
            'actions': np.zeros((self.n_envs, self.n_steps,) + self.act_space).astype(np.float32),
            'rewards': np.zeros((self.n_envs, self.n_steps,) + self.rew_space).astype(np.float32),
            'rewards_assign': np.zeros((self.n_envs, self.n_steps,) + self.rew_space).astype(np.float32),
            'values': np.zeros((self.n_envs, self.n_steps,) + self.rew_space).astype(np.float32),
            'log_pi_old': np.zeros((self.n_envs, self.n_steps, self.n_agents,)).astype(np.float32),
            'advantages': np.zeros((self.n_envs, self.n_steps,) + self.rew_space).astype(np.float32),
            'terminals': np.zeros((self.n_envs, self.n_steps,) + self.done_space).astype(np.bool),
            'agent_mask': np.ones((self.n_envs, self.n_steps, self.n_agents)).astype(np.bool),
            'act_mean': np.zeros((self.n_envs, self.n_steps,) + self.act_prob_space).astype(np.float32)
        })

        self.ptr = 0  # current pointer
        self.size = 0  # current buffer size
        self.start_ids = np.zeros(self.n_envs)

    def finish_ac_path(self, value, i_env):  # when an episode is finished
        if self.size == 0:
            return
        end_id = self.n_steps if (self.ptr == 0) else self.ptr
        path_slice = np.arange(self.start_ids[i_env], end_id).astype(np.int32)
        if self.full:
            path_slice = np.arange(self.start_ids[i_env], self.buffer_size).astype(np.int32)
        rewards = np.append(np.array(self.data['rewards_assign'][i_env, path_slice]), [value], axis=0)
        returns = np.append(np.array(self.data['values'][i_env, path_slice]), [value], axis=0)
        deltas = rewards[:-1] + self.gamma * returns[1:] - returns[:-1]
        advantages = discount_cumsum(deltas, self.gamma * self.gae_lambda) if self.use_gae else deltas
        returns = discount_cumsum(rewards, self.gamma)[:-1]
        self.data['values'][i_env, path_slice] = returns
        self.data['advantages'][i_env, path_slice] = advantages
        self.start_ids[i_env] = self.ptr
