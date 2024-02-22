from xuance.mindspore.agents import *


class MFAC_Agents(MARLAgents):
    def __init__(self,
                 config: Namespace,
                 envs: DummyVecEnv_Pettingzoo):
        self.gamma = config.gamma
        self.n_envs = envs.num_envs
        self.n_size = config.buffer_size
        self.n_epoch = config.n_epoch
        self.n_minibatch = config.n_minibatch
        if config.state_space is not None:
            config.dim_state, state_shape = config.state_space.shape, config.state_space.shape
        else:
            config.dim_state, state_shape = None, None

        input_representation = get_repre_in(config)
        representation = REGISTRY_Representation[config.representation](*input_representation)
        input_policy = get_policy_in_marl(config, representation)
        policy = REGISTRY_Policy[config.policy](*input_policy, gain=config.gain)
        scheduler = lr_decay_model(learning_rate=config.learning_rate, decay_rate=0.5,
                                   decay_steps=get_total_iters(config.agent_name, config))
        optimizer = Adam(policy.trainable_params(), scheduler, eps=1e-5)
        self.observation_space = envs.observation_space
        self.action_space = envs.action_space
        self.representation_info_shape = policy.representation.output_shapes
        self.auxiliary_info_shape = {}

        if config.state_space is not None:
            config.dim_state, state_shape = config.state_space.shape, config.state_space.shape
        else:
            config.dim_state, state_shape = None, None
        memory = MeanField_OnPolicyBuffer(config.n_agents,
                                          state_shape,
                                          config.obs_shape,
                                          config.act_shape,
                                          config.rew_shape,
                                          config.done_shape,
                                          envs.num_envs,
                                          config.buffer_size,
                                          config.use_gae, config.use_advnorm, config.gamma, config.gae_lambda,
                                          prob_space=config.act_prob_shape)
        self.buffer_size = memory.buffer_size
        self.batch_size = self.buffer_size // self.n_minibatch
        learner = MFAC_Learner(config, policy, optimizer, scheduler, config.model_dir, config.gamma)
        super(MFAC_Agents, self).__init__(config, envs, policy, memory, learner, config.log_dir, config.model_dir)
        self._concat = ops.Concat(axis=-1)
        self.on_policy = True

    def act(self, obs_n, test_mode, act_mean=None, agent_mask=None):
        batch_size = len(obs_n)
        agents_id = ops.broadcast_to(self.expand_dims(self.eye(self.n_agents, self.n_agents, ms.float32), 0),
                                     (batch_size, -1, -1))
        obs_n = Tensor(obs_n)

        _, act_probs = self.policy(obs_n, agents_id)
        acts = self.policy.actor.sample(act_probs)

        n_alive = ops.broadcast_to(self.expand_dims(Tensor(agent_mask).sum(axis=-1), -1), (-1, int(self.dim_act)))
        action_n_mask = ops.broadcast_to(self.expand_dims(Tensor(agent_mask), -1), (-1, -1, int(self.dim_act)))
        act_neighbor_onehot = self.learner.onehot_action(acts, self.dim_act) * action_n_mask
        act_mean_current = act_neighbor_onehot.sum(axis=1) / n_alive
        act_mean_current = act_mean_current.asnumpy()

        return acts.asnumpy(), act_mean_current

    def values(self, obs, actions_mean):
        batch_size = len(obs)
        actions_mean = ops.broadcast_to(Tensor(actions_mean).unsqueeze(1), (-1, self.n_agents, -1))
        agents_id = ops.broadcast_to(self.expand_dims(self.eye(self.n_agents, self.n_agents, ms.int32), 0),
                                     (batch_size, -1, -1)).astype(ms.float32)
        values_n = self.policy.get_values(Tensor(obs), actions_mean, agents_id)
        hidden_states = None
        return hidden_states, values_n.asnumpy()

    def train(self, i_step, **kwargs):
        if self.memory.full:
            info_train = {}
            indexes = np.arange(self.buffer_size)
            for _ in range(self.n_epoch):
                np.random.shuffle(indexes)
                for start in range(0, self.buffer_size, self.batch_size):
                    end = start + self.batch_size
                    sample_idx = indexes[start:end]
                    sample = self.memory.sample(sample_idx)
                    info_train = self.learner.update(sample)
            self.memory.clear()
            return info_train
        else:
            return {}
