from xuanpolicy.torch.agents import *
from xuanpolicy.torch.agents.agents_marl import linear_decay_or_increase


class VDN_Agents(MARLAgents):
    def __init__(self,
                 config: Namespace,
                 envs: DummyVecEnv_Pettingzoo,
                 device: Optional[Union[int, str, torch.device]] = None):
        self.gamma = config.gamma

        self.start_greedy = config.start_greedy
        self.end_greedy = config.end_greedy
        self.egreedy = config.start_greedy

        input_representation = get_repre_in(config)
        representation = REGISTRY_Representation[config.representation](*input_representation)
        mixer = VDN_mixer()
        input_policy = get_policy_in_marl(config, representation, config.agent_keys, mixer)
        policy = REGISTRY_Policy[config.policy](*input_policy)
        optimizer = torch.optim.Adam(policy.parameters(), config.learning_rate, eps=1e-5)
        scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.5,
                                                      total_iters=get_total_iters(config.agent_name, config))
        self.observation_space = envs.observation_space
        self.action_space = envs.action_space
        self.representation_info_shape = policy.representation.output_shapes
        self.auxiliary_info_shape = {}

        if config.state_space is not None:
            config.dim_state, state_shape = config.state_space.shape, config.state_space.shape
        else:
            config.dim_state, state_shape = None, None
        memory = MARL_OffPolicyBuffer(state_shape,
                                      config.obs_shape,
                                      config.act_shape,
                                      config.rew_shape,
                                      config.done_shape,
                                      envs.num_envs,
                                      config.buffer_size,
                                      config.batch_size)
        learner = VDN_Learner(config, policy, optimizer, scheduler,
                              config.device, config.modeldir, config.gamma,
                              config.sync_frequency)
        self.start_greedy, self.end_greedy = config.start_greedy, config.end_greedy
        self.egreedy = self.start_greedy
        self.delta_egreedy = (self.start_greedy - self.end_greedy) / (config.decay_step_greedy / envs.num_envs / envs.max_episode_length)
        super(VDN_Agents, self).__init__(config, envs, policy, memory, learner, device,
                                         config.logdir, config.modeldir)

    def act(self, obs_n, test_mode=False):
        batch_size = obs_n.shape[0]
        agents_id = torch.eye(self.n_agents).unsqueeze(0).expand(batch_size, -1, -1).to(self.device)
        obs_in = torch.Tensor(obs_n).view([batch_size, self.n_agents, -1]).to(self.device)
        _, greedy_actions, _ = self.policy(obs_in, agents_id)
        greedy_actions = greedy_actions.cpu().detach().numpy()

        if test_mode:
            return greedy_actions
        else:
            random_actions = np.random.choice(self.dim_act, [self.nenvs, self.n_agents])
            if np.random.rand() < self.egreedy:
                return random_actions
            else:
                return greedy_actions

    def train(self, i_step):
        if self.egreedy >= self.end_greedy:
            self.egreedy -= self.delta_egreedy

        if i_step > self.start_training:
            sample = self.memory.sample()
            info_train = self.learner.update(sample)
            info_train["epsilon-greedy"] = self.egreedy
            return info_train
        else:
            return {}
