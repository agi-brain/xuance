from xuance.tensorflow.agents import *
from xuance.tensorflow.agents.agents_marl import linear_decay_or_increase


class COMA_Agents(MARLAgents):
    def __init__(self,
                 config: Namespace,
                 envs: DummyVecEnv_Pettingzoo,
                 device: str = "cpu:0"):
        self.gamma = config.gamma
        self.start_greedy, self.end_greedy = config.start_greedy, config.end_greedy
        self.egreedy = self.start_greedy
        self.delta_egreedy = (self.start_greedy - self.end_greedy) / config.decay_step_greedy

        self.n_envs = envs.num_envs
        self.n_size = config.n_size
        self.n_epoch = config.n_epoch
        self.n_minibatch = config.n_minibatch
        if config.state_space is not None:
            config.dim_state, state_shape = config.state_space.shape[0], config.state_space.shape
        else:
            config.dim_state, state_shape = None, None

        # create representation for COMA actor
        input_representation = get_repre_in(config)
        self.use_recurrent = config.use_recurrent
        self.use_global_state = config.use_global_state
        kwargs_rnn = {"N_recurrent_layers": config.N_recurrent_layers,
                      "dropout": config.dropout,
                      "rnn": config.rnn} if self.use_recurrent else {}
        representation = REGISTRY_Representation[config.representation](*input_representation, **kwargs_rnn)
        # create policy
        input_policy = get_policy_in_marl(config, representation)
        policy = REGISTRY_Policy[config.policy](*input_policy,
                                                use_recurrent=config.use_recurrent,
                                                rnn=config.rnn,
                                                gain=config.gain,
                                                use_global_state=self.use_global_state,
                                                dim_state=config.dim_state)
        lr_scheduler = [MyLinearLR(config.learning_rate_actor, start_factor=1.0, end_factor=0.5,
                                   total_iters=get_total_iters(config.agent_name, config)),
                        MyLinearLR(config.learning_rate_critic, start_factor=1.0, end_factor=0.5,
                                   total_iters=get_total_iters(config.agent_name, config))]
        optimizer = [tk.optimizers.Adam(lr_scheduler[0]),
                     tk.optimizers.Adam(lr_scheduler[1])]
        self.observation_space = envs.observation_space
        self.action_space = envs.action_space
        self.representation_info_shape = policy.representation.output_shapes
        self.auxiliary_info_shape = {}

        if config.state_space is not None:
            config.dim_state, state_shape = config.state_space.shape, config.state_space.shape
        else:
            config.dim_state, state_shape = None, None
        config.act_onehot_shape = config.act_shape + tuple([config.dim_act])

        buffer = COMA_Buffer_RNN if self.use_recurrent else COMA_Buffer
        input_buffer = (config.n_agents, config.state_space.shape, config.obs_shape, config.act_shape, config.rew_shape,
                        config.done_shape, envs.num_envs, config.n_size,
                        config.use_gae, config.use_advnorm, config.gamma, config.gae_lambda)
        memory = buffer(*input_buffer, max_episode_length=envs.max_episode_length,
                        dim_act=config.dim_act, td_lambda=config.td_lambda)
        self.buffer_size = memory.buffer_size
        self.batch_size = self.buffer_size // self.n_minibatch

        learner = COMA_Learner(config, policy, optimizer,
                               config.device, config.model_dir, config.gamma, config.sync_frequency)
        super(COMA_Agents, self).__init__(config, envs, policy, memory, learner, device,
                                          config.log_dir, config.model_dir)
        self.on_policy = True

    def act(self, obs_n, *rnn_hidden, avail_actions=None, test_mode=False):
        batch_size = len(obs_n)
        with tf.device(self.device):
            # build critic input
            agents_id = tf.tile(tf.expand_dims(tf.eye(self.n_agents), axis=0), multiples=(batch_size, 1, 1))
            inputs_policy = {"obs": tf.convert_to_tensor(obs_n), "ids": agents_id}


            # actions_n = torch.Tensor(actions_n).unsqueeze(-1).to(self.device)
            # actions_in = torch.Tensor(actions_onehot).unsqueeze(1).to(self.device)
            # actions_in = actions_in.view(batch_size, 1, -1).repeat(1, self.n_agents, 1)
            # agent_mask = 1 - torch.eye(self.n_agents, device=self.device)
            # agent_mask = agent_mask.view(-1, 1).repeat(1, self.dim_act).view(self.n_agents, -1)
            # actions_in = actions_in * agent_mask.unsqueeze(0)
            # if self.use_global_state:
            #     state = torch.Tensor(state).unsqueeze(1).to(self.device).repeat(1, self.n_agents, 1)
            #     critic_in = torch.concat([state, obs_n, actions_in], dim=-1)
            # else:
            #     critic_in = torch.concat([obs_n, actions_in], dim=-1)
            # # get critic values
            # hidden_state, values_n = self.policy.get_values(critic_in, target=True)
            #
            # target_values = values_n.gather(-1, actions_n.long())
            # return hidden_state, target_values.detach().cpu().numpy()


        agents_id = tf.tile(tf.expand_dims(tf.eye(self.n_agents), axis=0), multiples=(batch_size, 1, 1))
        with tf.device(self.device):
            agents_id = tf.tile(tf.expand_dims(tf.eye(self.n_agents), axis=0), multiples=(batch_size, 1, 1))
            inputs_policy = {"obs": tf.convert_to_tensor(obs_n), "ids": agents_id}
            states, dists = self.policy(inputs_policy)
            # acts = dists.stochastic_sample()  # stochastic policy
            epsilon = 1.0 if test_mode else self.epsilon_decay.epsilon
            greedy_actions = tf.argmax(dists.logits, axis=-1)
        if noise:
            random_variable = np.random.random(greedy_actions.shape)
            action_pick = np.int32((random_variable < epsilon))
            random_actions = np.array([[self.args.action_space[agent].sample() for agent in self.agent_keys]])
            actions_select = action_pick * greedy_actions.numpy() + (1 - action_pick) * random_actions
            actions_onehot = self.learner.onehot_action(actions_select, self.dim_act)
            return actions_select, actions_onehot.numpy()
        else:
            actions_onehot = self.learner.onehot_action(greedy_actions, self.dim_act)
            return greedy_actions.numpy(), actions_onehot.numpy()

    def train(self, i_step, **kwargs):
        if self.egreedy >= self.end_greedy:
            self.egreedy = self.start_greedy - self.delta_egreedy * i_step
        info_train = {}
        if self.memory.full:
            indexes = np.arange(self.buffer_size)
            for _ in range(self.n_epoch):
                np.random.shuffle(indexes)
                for start in range(0, self.buffer_size, self.batch_size):
                    end = start + self.batch_size
                    sample_idx = indexes[start:end]
                    sample = self.memory.sample(sample_idx)
                    if self.use_recurrent:
                        info_train = self.learner.update_recurrent(sample, self.egreedy)
                    else:
                        info_train = self.learner.update(sample, self.egreedy)
            self.memory.clear()
        info_train["epsilon-greedy"] = self.egreedy
        return info_train
