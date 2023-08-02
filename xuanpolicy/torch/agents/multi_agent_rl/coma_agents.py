from xuanpolicy.torch.agents import *
from xuanpolicy.torch.agents.agents_marl import linear_decay_or_increase


class COMA_Agents(MARLAgents):
    def __init__(self,
                 config: Namespace,
                 envs: DummyVecEnv_Pettingzoo,
                 device: Optional[Union[int, str, torch.device]] = None):
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
        if self.use_recurrent:
            kwargs_rnn = {"N_recurrent_layers": config.N_recurrent_layers,
                          "dropout": config.dropout,
                          "rnn": config.rnn}
            representation = REGISTRY_Representation[config.representation](*input_representation, **kwargs_rnn)
        else:
            representation = REGISTRY_Representation[config.representation](*input_representation)

        input_policy = get_policy_in_marl(config, representation, config.agent_keys, None)
        policy = REGISTRY_Policy[config.policy](*input_policy)
        optimizer = [torch.optim.Adam(policy.parameters_actor, config.learning_rate_actor, eps=1e-5),
                     torch.optim.Adam(policy.parameters_critic, config.learning_rate_critic, eps=1e-5)]
        scheduler = [torch.optim.lr_scheduler.LinearLR(optimizer[0], start_factor=1.0, end_factor=0.5,
                                                       total_iters=get_total_iters(config.agent_name, config)),
                     torch.optim.lr_scheduler.LinearLR(optimizer[1], start_factor=1.0, end_factor=0.5,
                                                       total_iters=get_total_iters(config.agent_name, config))]
        self.observation_space = envs.observation_space
        self.action_space = envs.action_space
        self.representation_info_shape = policy.representation.output_shapes
        self.auxiliary_info_shape = {}

        if config.state_space is not None:
            config.dim_state, state_shape = config.state_space.shape, config.state_space.shape
        else:
            config.dim_state, state_shape = None, None
        config.act_onehot_shape = config.act_shape + tuple([config.dim_act])

        # memory = COMA_Buffer(state_shape, config.obs_shape, config.act_shape, config.act_onehot_shape,
        #                      config.rew_shape, config.done_shape, envs.num_envs,
        #                      config.buffer_size, config.batch_size, envs.envs[0].max_cycles)
        buffer = MARL_OnPolicyBuffer_RNN if self.use_recurrent else MARL_OnPolicyBuffer
        input_buffer = (config.n_agents, config.state_space.shape, config.obs_shape, config.act_shape, config.rew_shape,
                        config.done_shape, envs.num_envs, config.n_size,
                        config.use_gae, config.use_advnorm, config.gamma, config.gae_lambda)
        memory = buffer(*input_buffer, max_episode_length=envs.max_episode_length, dim_act=config.dim_act)
        self.buffer_size = memory.buffer_size
        self.batch_size = self.buffer_size // self.n_minibatch

        learner = COMA_Learner(config, policy, optimizer, scheduler,
                               config.device, config.model_dir, config.gamma, config.sync_frequency)

        super(COMA_Agents, self).__init__(config, envs, policy, memory, learner, device,
                                          config.log_dir, config.model_dir)

    def act(self, obs_n, episode, test_mode, noise=False):
        batch_size = len(obs_n)
        agents_id = torch.eye(self.n_agents).unsqueeze(0).expand(batch_size, -1, -1).to(self.device)
        states, dists = self.policy(obs_n, agents_id)
        # acts = dists.stochastic_sample()  # stochastic policy
        epsilon = 1.0 if test_mode else self.epsilon_decay.epsilon
        greedy_actions = dists.logits.argmax(dim=-1, keepdims=False)
        if noise:
            random_variable = np.random.random(greedy_actions.shape)
            action_pick = np.int32((random_variable < epsilon))
            random_actions = np.array([[self.args.action_space[agent].sample() for agent in self.agent_keys]])
            actions_select = action_pick * greedy_actions.cpu().numpy() + (1 - action_pick) * random_actions
            actions_onehot = self.learner.onehot_action(torch.Tensor(actions_select), self.dim_act)
            return actions_select, actions_onehot.detach().cpu().numpy()
        else:
            actions_onehot = self.learner.onehot_action(greedy_actions, self.dim_act)
            return greedy_actions.detach().cpu().numpy(), actions_onehot.detach().cpu().numpy()

    def train(self, i_episode):
        self.epsilon_decay.update()
        if self.memory.full:
            sample = self.memory.sample()
            info_train = self.learner.update(sample)
            return info_train
        else:
            return {}
