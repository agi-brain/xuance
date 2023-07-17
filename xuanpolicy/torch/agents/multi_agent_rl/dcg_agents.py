import torch.nn
from xuanpolicy.torch.agents import *


class DCG_Agents(MARLAgents):
    def __init__(self,
                 config: Namespace,
                 envs: DummyVecEnv_Pettingzoo,
                 device: Optional[Union[int, str, torch.device]] = None):
        self.gamma = config.gamma
        self.start_greedy, self.end_greedy = config.start_greedy, config.end_greedy
        self.egreedy = self.start_greedy
        self.delta_egreedy = (self.start_greedy - self.end_greedy) / (
                config.decay_step_greedy / envs.num_envs / envs.max_episode_length)

        input_representation = get_repre_in(config)
        self.use_recurrent = config.use_recurrent
        if self.use_recurrent:
            kwargs_rnn = {"N_recurrent_layers": config.N_recurrent_layers,
                          "dropout": config.dropout,
                          "rnn": config.rnn}
            representation = REGISTRY_Representation[config.representation](*input_representation, **kwargs_rnn)
        else:
            representation = REGISTRY_Representation[config.representation](*input_representation)
        repre_state_dim = config.representation_hidden_size[-1]
        from xuanpolicy.torch.policies.coordination_graph import DCG_utility, DCG_payoff, Coordination_Graph
        utility = DCG_utility(repre_state_dim, config.hidden_utility_dim, config.dim_act).to(device)
        payoffs = DCG_payoff(repre_state_dim * 2, config.hidden_payoff_dim, config.dim_act, config).to(device)
        dcgraph = Coordination_Graph(config.n_agents, config.graph_type)
        dcgraph.set_coordination_graph(device)
        if config.env_name == "StarCraft2":
            action_space = config.action_space
        else:
            action_space = config.action_space[config.agent_keys[0]]
        if config.agent == "DCG_S":
            policy = REGISTRY_Policy[config.policy](action_space,
                                                    config.state_space.shape[0], representation,
                                                    utility, payoffs, dcgraph, config.hidden_bias_dim,
                                                    None, None, torch.nn.ReLU, device,
                                                    use_recurrent=config.use_recurrent,
                                                    rnn=config.rnn)
        else:
            policy = REGISTRY_Policy[config.policy](action_space,
                                                    config.state_space.shape[0], representation,
                                                    utility, payoffs, dcgraph, None,
                                                    None, None, torch.nn.ReLU, device,
                                                    use_recurrent=config.use_recurrent,
                                                    rnn=config.rnn)
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
        from xuanpolicy.torch.learners.multi_agent_rl.dcg_learner import DCG_Learner
        learner = DCG_Learner(config, policy, optimizer, scheduler,
                              config.device, config.modeldir, config.gamma,
                              config.sync_frequency)
        super(DCG_Agents, self).__init__(config, envs, policy, memory, learner, device, config.logdir, config.modeldir)

    def act(self, obs_n, *rnn_hidden, avail_actions=None, test_mode=False):
        return self.learner.act(obs_n, *rnn_hidden, avail_actions=avail_actions, test_mode=test_mode)

    def train(self, i_step):
        if self.egreedy >= self.end_greedy:
            self.egreedy -= self.delta_egreedy

        if i_step > self.start_training:
            sample = self.memory.sample()
            if self.use_recurrent:
                info_train = self.learner.update_recurrent(sample)
            else:
                info_train = self.learner.update(sample)
            info_train["epsilon-greedy"] = self.egreedy
            return info_train
        else:
            return {}
