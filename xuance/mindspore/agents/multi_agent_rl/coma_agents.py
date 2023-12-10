from xuance.mindspore.agents import *


class COMA_Agents(MARLAgents):
    def __init__(self,
                 config: Namespace,
                 envs: DummyVecEnv_Pettingzoo):
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
        scheduler = [lr_decay_model(learning_rate=config.learning_rate_actor, decay_rate=0.5,
                                    decay_steps=get_total_iters(config.agent_name, config)),
                     lr_decay_model(learning_rate=config.learning_rate_critic, decay_rate=0.5,
                                    decay_steps=get_total_iters(config.agent_name, config))]
        optimizer = [Adam(policy.parameters_actor, scheduler[0], eps=1e-5),
                     Adam(policy.parameters_critic, scheduler[1], eps=1e-5)]
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

        learner = COMA_Learner(config, policy, optimizer, scheduler,
                               config.model_dir, config.gamma, config.sync_frequency)

        super(COMA_Agents, self).__init__(config, envs, policy, memory, learner, config.log_dir, config.model_dir)
        self.on_policy = True

    def act(self, obs_n, *rnn_hidden, avail_actions=None, test_mode=False):
        batch_size = len(obs_n)
        agents_id = ops.broadcast_to(self.expand_dims(self.eye(self.n_agents, self.n_agents, ms.float32), 0),
                                     (batch_size, -1, -1))
        obs_in = Tensor(obs_n).view(batch_size, self.n_agents, -1)
        epsilon = 0.0 if test_mode else self.end_greedy
        if self.use_recurrent:
            batch_agents = batch_size * self.n_agents
            hidden_state, action_probs = self.policy(obs_in.view(batch_agents, 1, -1),
                                                     agents_id.view(batch_agents, 1, -1),
                                                     *rnn_hidden,
                                                     avail_actions=avail_actions.reshape(batch_agents, 1, -1),
                                                     epsilon=epsilon)
            action_probs = action_probs.view(batch_size, self.n_agents, self.dim_act)
        else:
            hidden_state, action_probs = self.policy(obs_in, agents_id,
                                                     avail_actions=avail_actions,
                                                     epsilon=epsilon)
        picked_actions = Categorical(action_probs).sample()
        onehot_actions = self.learner.onehot_action(picked_actions, self.dim_act)
        return hidden_state, picked_actions.asnumpy(), onehot_actions.asnumpy()

    def values(self, obs_n, *rnn_hidden, state=None, actions_n=None, actions_onehot=None):
        batch_size = len(obs_n)
        # build critic input
        obs_n = Tensor(obs_n)
        actions_n = self.expand_dims(Tensor(actions_n), -1)
        actions_in = self.expand_dims(Tensor(actions_onehot), 1)
        actions_in = ops.broadcast_to(actions_in.view(batch_size, 1, -1), (-1, self.n_agents, -1))
        agent_mask = 1 - self.eye(self.n_agents, self.n_agents, ms.float32)
        agent_mask = ops.broadcast_to(agent_mask.view(-1, 1), (-1, int(self.dim_act))).view(self.n_agents, -1)
        actions_in = actions_in * self.expand_dims(agent_mask, 0)
        if self.use_global_state:
            state = ops.broadcast_to(self.expand_dims(Tensor(state), 1), (-1, self.n_agents, -1))
            critic_in = self.policy._concat([state, obs_n, actions_in])
        else:
            critic_in = self.policy._concat([obs_n, actions_in])
        # get critic values
        hidden_state, values_n = self.policy.get_values(critic_in, target=True)

        target_values = values_n.gather(actions_n, -1, -1)
        return hidden_state, target_values.asnumpy()

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

