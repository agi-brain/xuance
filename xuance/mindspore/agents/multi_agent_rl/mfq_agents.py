from xuance.mindspore.agents import *
from xuance.mindspore.agents.agents_marl import linear_decay_or_increase


class MFQ_Agents(MARLAgents):
    def __init__(self,
                 config: Namespace,
                 envs: DummyVecEnv_Pettingzoo):
        self.gamma = config.gamma

        self.start_greedy, self.end_greedy = config.start_greedy, config.end_greedy
        self.egreedy = self.start_greedy
        self.delta_egreedy = (self.start_greedy - self.end_greedy) / config.decay_step_greedy
        self.use_recurrent, self.rnn = config.use_recurrent, config.rnn
        self.rnn_hidden = None

        input_representation = get_repre_in(config)
        representation = REGISTRY_Representation[config.representation](*input_representation)
        input_policy = get_policy_in_marl(config, representation)
        policy = REGISTRY_Policy[config.policy](*input_policy)
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
        memory = MeanField_OffPolicyBuffer(config.n_agents,
                                           state_shape,
                                           config.obs_shape,
                                           config.act_shape,
                                           config.act_prob_shape,
                                           config.rew_shape,
                                           config.done_shape,
                                           envs.num_envs,
                                           config.buffer_size,
                                           config.batch_size)
        learner = MFQ_Learner(config, policy, optimizer, scheduler,
                              config.model_dir, config.gamma, config.sync_frequency)
        super(MFQ_Agents, self).__init__(config, envs, policy, memory, learner, config.log_dir, config.model_dir)
        self.on_policy = False

    def act(self, obs_n, *rnn_hidden, test_mode=False, act_mean=None, agent_mask=None, avail_actions=None):
        batch_size = obs_n.shape[0]
        agents_id = ops.broadcast_to(self.expand_dims(self.eye(self.n_agents, self.n_agents, ms.float32), 0),
                                     (batch_size, -1, -1))
        obs_in = Tensor(obs_n)
        act_mean = ops.broadcast_to(self.expand_dims(Tensor(act_mean).astype(ms.float32), -2), (-1, self.n_agents, -1))

        if self.use_recurrent:  # awaiting to be tested
            batch_agents = batch_size * self.n_agents
            hidden_state, greedy_actions, q_output = self.policy(obs_in.view(batch_agents, 1, -1),
                                                                 act_mean.view(batch_agents, 1, -1),
                                                                 agents_id.view(batch_agents, 1, -1),
                                                                 *rnn_hidden,
                                                                 avail_actions=avail_actions)
        else:
            hidden_state, greedy_actions, q_output = self.policy(obs_in, act_mean, agents_id)
        n_alive = ops.broadcast_to(self.expand_dims(Tensor(agent_mask).sum(axis=-1), -1), (-1, int(self.dim_act)))
        action_n_mask = ops.broadcast_to(self.expand_dims(Tensor(agent_mask), -1), (-1, -1, int(self.dim_act)))
        act_neighbor_sample = self.policy.sample_actions(logits=q_output)
        act_neighbor_onehot = self.learner.onehot_action(act_neighbor_sample, self.dim_act) * action_n_mask
        act_mean_current = act_neighbor_onehot.sum(axis=1) / n_alive
        act_mean_current = act_mean_current.asnumpy()
        greedy_actions = greedy_actions.asnumpy()
        if test_mode:
            return hidden_state, greedy_actions, act_mean_current
        else:
            random_actions = np.random.choice(self.dim_act, [self.nenvs, self.n_agents])
            if np.random.rand() < self.egreedy:
                return hidden_state, random_actions, act_mean_current
            else:
                return hidden_state, greedy_actions, act_mean_current

    def train(self, i_step, n_epoch=1):
        if self.egreedy >= self.end_greedy:
            self.egreedy = self.start_greedy - self.delta_egreedy * i_step
        info_train = {}
        if i_step > self.start_training:
            for i_epoch in range(n_epoch):
                sample = self.memory.sample()
                info_train = self.learner.update(sample)
        info_train["epsilon-greedy"] = self.egreedy
        return info_train
