from xuance.mindspore.agents import *


class VDAC_Agents(MARLAgents):
    def __init__(self,
                 config: Namespace,
                 envs: DummyVecEnv_Pettingzoo):
        self.gamma = config.gamma
        self.n_envs = envs.num_envs
        self.n_size = config.n_size
        self.n_epoch = config.n_epoch
        self.n_minibatch = config.n_minibatch
        if config.state_space is not None:
            config.dim_state, state_shape = config.state_space.shape, config.state_space.shape
        else:
            config.dim_state, state_shape = None, None

        input_representation = get_repre_in(config)
        self.use_recurrent = config.use_recurrent
        # create representation for actor
        kwargs_rnn = {"N_recurrent_layers": config.N_recurrent_layers,
                      "dropout": config.dropout,
                      "rnn": config.rnn} if self.use_recurrent else {}
        representation = REGISTRY_Representation[config.representation](*input_representation, **kwargs_rnn)
        # create policy
        if config.mixer == "VDN":
            mixer = VDN_mixer()
        elif config.mixer == "QMIX":
            mixer = QMIX_mixer(config.dim_state[0], config.hidden_dim_mixing_net, config.hidden_dim_hyper_net,
                               config.n_agents)
        elif config.mixer == "Independent":
            mixer = None
        else:
            raise f"Mixer named {config.mixer} is not defined!"
        input_policy = get_policy_in_marl(config, representation, mixer=mixer)
        policy = REGISTRY_Policy[config.policy](*input_policy,
                                                use_recurrent=config.use_recurrent,
                                                rnn=config.rnn,
                                                gain=config.gain)
        scheduler = lr_decay_model(learning_rate=config.learning_rate, decay_rate=0.5,
                                   decay_steps=get_total_iters(config.agent_name, config))
        optimizer = nn.Adam(policy.trainable_params(), scheduler, eps=1e-5)
        self.observation_space = envs.observation_space
        self.action_space = envs.action_space
        self.auxiliary_info_shape = {}

        buffer = MARL_OnPolicyBuffer_RNN if self.use_recurrent else MARL_OnPolicyBuffer
        input_buffer = (config.n_agents, config.state_space.shape, config.obs_shape, config.act_shape, config.rew_shape,
                        config.done_shape, envs.num_envs, config.n_size,
                        config.use_gae, config.use_advnorm, config.gamma, config.gae_lambda)
        memory = buffer(*input_buffer, max_episode_length=envs.max_episode_length, dim_act=config.dim_act)
        self.buffer_size = memory.buffer_size
        self.batch_size = self.buffer_size // self.n_minibatch

        learner = VDAC_Learner(config, policy, optimizer, scheduler, config.model_dir, config.gamma)
        super(VDAC_Agents, self).__init__(config, envs, policy, memory, learner, config.log_dir, config.model_dir)
        self.share_values = True if config.rew_shape[0] == 1 else False
        self.on_policy = True

    def act(self, obs_n, *rnn_hidden, avail_actions=None, state=None, test_mode=False):
        batch_size = len(obs_n)
        agents_id = ops.broadcast_to(self.expand_dims(self.eye(self.n_agents, self.n_agents, ms.float32), 0),
                                     (batch_size, -1, -1))
        obs_in = Tensor(obs_n).view(batch_size, self.n_agents, -1)
        if state is not None:
            state = Tensor(state)
        if self.use_recurrent:
            batch_agents = batch_size * self.n_agents
            hidden_state, act_probs, values_tot = self.policy(obs_in.view(batch_agents, 1, -1),
                                                              agents_id.unsqueeze(2),
                                                              *rnn_hidden,
                                                              avail_actions=avail_actions[:, :, np.newaxis],
                                                              state=state.unsqueeze(2))
            actions = self.policy.actor.sample(act_probs)
            actions = actions.reshape(batch_size, self.n_agents)
            values_tot = values_tot.reshape([batch_size, self.n_agents, 1])
        else:
            hidden_state, act_probs, values_tot = self.policy(obs_in, agents_id,
                                                              avail_actions=avail_actions,
                                                              state=state)
            actions = self.policy.actor.sample(act_probs)
        return hidden_state, actions.asnumpy(), values_tot.asnumpy()

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
                    if self.use_recurrent:
                        info_train = self.learner.update_recurrent(sample)
                    else:
                        info_train = self.learner.update(sample)
            self.learner.lr_decay(i_step)
            self.memory.clear()
            return info_train
        else:
            return {}
