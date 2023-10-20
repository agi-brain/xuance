from xuance.torch.agents import *


class MFAC_Agents(MARLAgents):
    def __init__(self,
                 config: Namespace,
                 envs: DummyVecEnv_Pettingzoo,
                 device: Optional[Union[int, str, torch.device]] = None):
        self.gamma = config.gamma
        if config.state_space is not None:
            config.dim_state, state_shape = config.state_space.shape, config.state_space.shape
        else:
            config.dim_state, state_shape = None, None

        input_representation = get_repre_in(config)
        representation = REGISTRY_Representation[config.representation](*input_representation)
        input_policy = get_policy_in_marl(config, representation, config.agent_keys)
        policy = REGISTRY_Policy[config.policy](*input_policy)
        optimizer = [torch.optim.Adam(policy.parameters_actor, config.lr_a, eps=1e-5),
                     torch.optim.Adam(policy.parameters_critic, config.lr_c, eps=1e-5)]
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
        memory = MeanField_OnPolicyBuffer(state_shape,
                                          config.obs_shape,
                                          config.act_shape,
                                          config.act_prob_shape,
                                          config.rew_shape,
                                          config.done_shape,
                                          envs.num_envs,
                                          config.nsteps,
                                          config.nminibatch,
                                          config.use_gae, config.use_advnorm, config.gamma, config.lam)
        learner = MFAC_Learner(config, policy, optimizer, scheduler,
                               config.device, config.model_dir, config.gamma)
        super(MFAC_Agents, self).__init__(config, envs, policy, memory, learner, device,
                                          config.log_dir, config.model_dir)
        self.on_policy = True

    def act(self, obs_n, episode, test_mode, act_mean=None, agent_mask=None, noise=False):
        batch_size = len(obs_n)
        agents_id = torch.eye(self.n_agents).unsqueeze(0).expand(batch_size, -1, -1).to(self.device)
        obs_n = torch.Tensor(obs_n).to(self.device)

        _, dists = self.policy(obs_n, agents_id)
        acts = dists.stochastic_sample()

        n_alive = torch.Tensor(agent_mask).sum(dim=-1).unsqueeze(-1).repeat(1, self.dim_act).to(self.device)
        action_n_mask = torch.Tensor(agent_mask).unsqueeze(-1).repeat(1, 1, self.dim_act).to(self.device)
        act_neighbor_onehot = self.learner.onehot_action(acts, self.dim_act) * action_n_mask
        act_mean_current = act_neighbor_onehot.float().sum(dim=1) / n_alive
        act_mean_current = act_mean_current.cpu().detach().numpy()

        return acts.detach().cpu().numpy(), act_mean_current

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
