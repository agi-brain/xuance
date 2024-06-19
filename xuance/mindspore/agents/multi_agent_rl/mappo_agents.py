from xuance.mindspore.agents import *


class MAPPO_Agents(MARLAgents):
    def __init__(self,
                 config: Namespace,
                 envs: DummyVecMultiAgentEnv):
        self.gamma = config.gamma
        self.n_envs = envs.num_envs
        self.n_epochs = config.n_epochs
        self.n_minibatch = config.n_minibatch
        if config.state_space is not None:
            config.dim_state, state_shape = config.state_space.shape[0], config.state_space.shape
        else:
            config.dim_state, state_shape = None, None

        input_representation = get_repre_in(config)
        self.use_rnn = config.use_rnn
        self.use_global_state = config.use_global_state
        # create representation for actor
        kwargs_rnn = {"N_recurrent_layers": config.N_recurrent_layers,
                      "dropout": config.dropout,
                      "rnn": config.rnn} if self.use_rnn else {}
        representation = REGISTRY_Representation[config.representation](*input_representation, **kwargs_rnn)
        # create representation for critic
        if self.use_global_state:
            input_representation[0] = (config.dim_state + config.dim_obs * config.n_agents,)
        else:
            input_representation[0] = (config.dim_obs * config.n_agents,)
        representation_critic = REGISTRY_Representation[config.representation](*input_representation, **kwargs_rnn)
        # create policy
        input_policy = get_policy_in_marl(config, (representation, representation_critic))
        policy = REGISTRY_Policy[config.policy](*input_policy,
                                                use_rnn=config.use_rnn,
                                                rnn=config.rnn,
                                                gain=config.gain)
        scheduler = lr_decay_model(learning_rate=config.learning_rate, decay_rate=0.5,
                                   decay_steps=get_total_iters(config.agent_name, config))
        optimizer = Adam(policy.trainable_params(), config.learning_rate, eps=1e-5)
        self.observation_space = envs.observation_space
        self.action_space = envs.action_space
        self.auxiliary_info_shape = {}

        buffer = MARL_OnPolicyBuffer_RNN if self.use_rnn else MARL_OnPolicyBuffer
        input_buffer = (config.n_agents, config.state_space.shape, config.obs_shape, config.act_shape, config.rew_shape,
                        config.done_shape, envs.num_envs, config.n_size,
                        config.use_gae, config.use_advnorm, config.gamma, config.gae_lambda)
        memory = buffer(*input_buffer, max_episode_steps=envs.max_episode_steps, dim_act=config.dim_act)
        self.buffer_size = memory.buffer_size
        self.batch_size = self.buffer_size // self.n_minibatch

        learner = MAPPO_Learner(config, policy, optimizer, scheduler, config.model_dir, config.gamma)
        super(MAPPO_Agents, self).__init__(config, envs, policy, memory, learner, config.log_dir, config.model_dir)
        self.on_policy = True
        self._concat = ms.ops.Concat(axis=-1)

    def act(self, obs_n, *rnn_hidden, avail_actions=None, state=None, test_mode=False):
        batch_size = len(obs_n)
        agents_id = ops.broadcast_to(self.expand_dims(self.eye(self.n_agents, self.n_agents, ms.float32), 0),
                                     (batch_size, -1, -1))
        obs_in = Tensor(obs_n).view(batch_size, self.n_agents, -1)
        if self.use_rnn:
            batch_agents = batch_size * self.n_agents
            hidden_state, act_probs = self.policy(obs_in.view(batch_agents, 1, -1),
                                                  agents_id.view(batch_agents, 1, -1),
                                                  *rnn_hidden,
                                                  avail_actions=avail_actions.reshape(batch_agents, 1, -1))
            actions = self.policy.actor.sample(act_probs)
            log_pi_a = self.policy.actor.log_prob(value=actions, probs=act_probs)
            actions = actions.reshape(batch_size, self.n_agents)
        else:
            hidden_state, act_probs = self.policy(obs_in, agents_id, avail_actions=avail_actions)
            actions = self.policy.actor.sample(act_probs)
            log_pi_a = self.policy.actor.log_prob(value=actions, probs=act_probs)
        return hidden_state, actions.asnumpy(), log_pi_a.asnumpy()

    def values(self, obs_n, *rnn_hidden, state=None):
        batch_size = len(obs_n)
        agents_id = ops.broadcast_to(self.expand_dims(self.eye(self.n_agents, self.n_agents, ms.float32), 0),
                                     (batch_size, -1, -1))
        # build critic input
        if self.use_global_state:
            state = Tensor(state).unsqueeze(1)
            obs_n = Tensor(obs_n).view(batch_size, 1, -1)
            critic_in = self._concat([ops.broadcast_to(obs_n, (-1, self.n_agents, -1)),
                                      ops.broadcast_to(state, (-1, self.n_agents, -1))])
        else:
            critic_in = Tensor(obs_n).view(batch_size, 1, -1)
            critic_in = ops.broadcast_to(critic_in, (-1, self.n_agents, -1))
        # get critic values
        if self.use_rnn:
            hidden_state, values_n = self.policy.get_values(critic_in.unsqueeze(2),  # add a sequence length axis.
                                                            agents_id.unsqueeze(2),
                                                            *rnn_hidden)
            values_n = values_n.squeeze(2)
        else:
            hidden_state, values_n = self.policy.get_values(critic_in, agents_id)

        return hidden_state, values_n.asnumpy()

    def train(self, i_step, **kwargs):
        info_train = {}
        if self.memory.full:
            indexes = np.arange(self.buffer_size)
            for _ in range(self.n_epochs):
                np.random.shuffle(indexes)
                for start in range(0, self.buffer_size, self.batch_size):
                    end = start + self.batch_size
                    sample_idx = indexes[start:end]
                    sample = self.memory.sample(sample_idx)
                    if self.use_rnn:
                        info_train = self.learner.update_recurrent(sample)
                    else:
                        info_train = self.learner.update(sample)
            self.learner.lr_decay(i_step)
            self.memory.clear()
        return info_train
