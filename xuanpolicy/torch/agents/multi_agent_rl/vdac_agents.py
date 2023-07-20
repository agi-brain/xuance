from xuanpolicy.torch.agents import *


class VDAC_Agents(MARLAgents):
    def __init__(self,
                 config: Namespace,
                 envs: DummyVecEnv_Pettingzoo,
                 device: Optional[Union[int, str, torch.device]] = None):
        self.device = torch.device("cuda" if (torch.cuda.is_available() and config.device in ["gpu", "cuda:0"]) else "cpu")
        self.gamma = config.gamma

        if config.state_space is not None:
            config.dim_state, state_shape = config.state_space.shape, config.state_space.shape
        else:
            config.dim_state, state_shape = None, None

        input_representation = get_repre_in(config)
        representation = REGISTRY_Representation[config.representation](*input_representation)
        if config.mixer == "VDN":
            mixer = VDN_mixer()
        elif config.mixer == "QMIX":
            mixer = QMIX_mixer(config.dim_state[0], config.hidden_dim_mixing_net, config.hidden_dim_hyper_net,
                               config.n_agents, self.device)
        else:
            mixer = None

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
        memory = MARL_OnPolicyBuffer(state_shape, config.obs_shape, config.act_shape, config.rew_shape,
                                     config.done_shape, envs.num_envs, config.nsteps, config.nminibatch,
                                     config.use_gae, config.use_advnorm, config.gamma, config.lam)
        learner = VDAC_Learner(config, policy, optimizer, scheduler,
                               config.device, config.model_dir, config.gamma)
        super(VDAC_Agents, self).__init__(config, envs, policy, memory, learner, device,
                                          config.log_dir, config.model_dir)

    def act(self, obs_n, episode, test_mode, state=None, noise=False):
        batch_size = len(obs_n)
        agents_id = torch.eye(self.n_agents).unsqueeze(0).expand(batch_size, -1, -1).to(self.device)
        states, dists, vs = self.policy(obs_n, agents_id)
        if self.args.mixer == "VDN":
            vs_tot = self.policy.value_tot(vs).repeat(1, self.n_agents).unsqueeze(-1)
        else:
            vs_tot = self.policy.value_tot(vs, state).repeat(1, self.n_agents).unsqueeze(-1)
        acts = dists.stochastic_sample()
        return acts.detach().cpu().numpy(), vs_tot.detach().cpu().numpy()

    def value(self, obs, state):
        batch_size = len(state)
        agents_id = torch.eye(self.n_agents).unsqueeze(0).expand(batch_size, -1, -1).to(self.device)
        repre_out = self.policy.representation(obs)
        critic_input = torch.concat([torch.Tensor(repre_out['state']), agents_id], dim=-1)
        values_n = self.policy.critic(critic_input)
        values = self.policy.value_tot(values_n, global_state=state).view(-1, 1).repeat(1, self.n_agents).unsqueeze(-1)
        return values.detach().cpu().numpy()

    def train(self, i_episode):
        if self.memory.full:
            info_train = {}
            for _ in range(self.args.nminibatch * self.args.nepoch):
                sample = self.memory.sample()
                info_train = self.learner.update(sample)
            self.memory.clear()
            return info_train
        else:
            return {}
