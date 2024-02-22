from xuance.tensorflow.agents import *
from xuance.tensorflow.agents.agents_marl import linear_decay_or_increase


class MFQ_Agents(MARLAgents):
    def __init__(self,
                 config: Namespace,
                 envs: DummyVecEnv_Pettingzoo,
                 device: str = "cpu:0"):
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
        lr_scheduler = MyLinearLR(config.learning_rate, start_factor=1.0, end_factor=0.5,
                                  total_iters=get_total_iters(config.agent_name, config))
        optimizer = tk.optimizers.Adam(lr_scheduler)
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
        learner = MFQ_Learner(config, policy, optimizer,
                              config.device, config.model_dir, config.gamma, config.sync_frequency)
        super(MFQ_Agents, self).__init__(config, envs, policy, memory, learner, device,
                                         config.log_dir, config.model_dir)
        self.on_policy = False

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
