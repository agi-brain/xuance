from xuance.torch.agents import *
from xuance.torch.agents.agents_marl import linear_decay_or_increase


class QTRAN_Agents(MARLAgents):
    def __init__(self,
                 config: Namespace,
                 envs: DummyVecEnv_Pettingzoo,
                 device: Optional[Union[int, str, torch.device]] = None):
        self.gamma = config.gamma

        self.start_greedy = config.start_greedy
        self.end_greedy = config.end_greedy
        self.egreedy = config.start_greedy
        self.delta_egreedy = (self.start_greedy - self.end_greedy) / config.decay_step_greedy
        if config.state_space is not None:
            config.dim_state, state_shape = config.state_space.shape, config.state_space.shape
        else:
            config.dim_state, state_shape = None, None

        input_representation = get_repre_in(config)
        representation = REGISTRY_Representation[config.representation](*input_representation)
        mixer = VDN_mixer()
        if config.agent == "QTRAN_base":
            qtran_net = QTRAN_base(config.dim_state[0], config.dim_act, config.qtran_net_hidden_dim,
                                   config.n_agents, config.q_hidden_size[0]).to(device)
        elif config.agent == "QTRAN_alt":
            qtran_net = QTRAN_alt(config.dim_state[0], config.dim_act, config.qtran_net_hidden_dim,
                                  config.n_agents, config.q_hidden_size[0]).to(device)
        else:
            raise ValueError("Mixer {} not recognised.".format(config.agent))

        input_policy = get_policy_in_marl(config, representation, config.agent_keys, mixer, qtran_mixer=qtran_net)
        policy = REGISTRY_Policy[config.policy](*input_policy)
        optimizer = torch.optim.Adam(policy.parameters(), config.learning_rate, eps=1e-5)
        scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.5,
                                                      total_iters=get_total_iters(config.agent_name, config))
        self.observation_space = envs.observation_space
        self.action_space = envs.action_space
        self.representation_info_shape = policy.representation.output_shapes
        self.auxiliary_info_shape = {}

        memory = MARL_OffPolicyBuffer(state_shape,
                                      config.obs_shape,
                                      config.act_shape,
                                      config.rew_shape,
                                      config.done_shape,
                                      envs.num_envs,
                                      config.buffer_size,
                                      config.batch_size)
        learner = QTRAN_Learner(config, policy, optimizer, scheduler,
                                config.device, config.model_dir, config.gamma,
                                config.sync_frequency)

        self.epsilon_decay = linear_decay_or_increase(config.start_greedy, config.end_greedy,
                                                      config.greedy_update_steps)
        super(QTRAN_Agents, self).__init__(config, envs, policy, memory, learner, device,
                                           config.log_dir, config.model_dir)
        self.on_policy = False

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
