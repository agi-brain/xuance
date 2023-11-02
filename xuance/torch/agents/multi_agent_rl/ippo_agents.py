import torch

from xuance.torch.agents import *


class IPPO_Agents(MARLAgents):
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
            config.dim_state, state_shape = config.state_space.shape[0], config.state_space.shape
        else:
            config.dim_state, state_shape = None, None

        input_representation = get_repre_in(config)
        self.use_recurrent = config.use_recurrent
        self.use_global_state = config.use_global_state
        # create representation for actor
        kwargs_rnn = {"N_recurrent_layers": config.N_recurrent_layers,
                      "dropout": config.dropout,
                      "rnn": config.rnn} if self.use_recurrent else {}
        representation = REGISTRY_Representation[config.representation](*input_representation, **kwargs_rnn)
        # create representation for critic
        input_representation[0] = (config.dim_state,) if self.use_global_state else (config.dim_obs,)
        representation_critic = REGISTRY_Representation[config.representation](*input_representation, **kwargs_rnn)
        # create policy
        input_policy = get_policy_in_marl(config, (representation, representation_critic))
        policy = REGISTRY_Policy[config.policy](*input_policy,
                                                use_recurrent=config.use_recurrent,
                                                rnn=config.rnn,
                                                gain=config.gain)
        optimizer = torch.optim.Adam(policy.parameters(),
                                     lr=config.learning_rate, eps=1e-5,
                                     weight_decay=config.weight_decay)
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

        learner = IPPO_Learner(config, policy, optimizer, None, config.device, config.model_dir, config.gamma)
        super(IPPO_Agents, self).__init__(config, envs, policy, memory, learner, device,
                                          config.log_dir, config.model_dir)
        self.share_values = True if config.rew_shape[0] == 1 else False
        self.on_policy = True

    def act(self, obs_n, *rnn_hidden, avail_actions=None, state=None, test_mode=False):
        batch_size = len(obs_n)
        agents_id = torch.eye(self.n_agents).unsqueeze(0).expand(batch_size, -1, -1).to(self.device)
        obs_in = torch.Tensor(obs_n).view([batch_size, self.n_agents, -1]).to(self.device)
        if self.use_recurrent:
            batch_agents = batch_size * self.n_agents
            hidden_state, dists = self.policy(obs_in.view(batch_agents, 1, -1),
                                              agents_id.view(batch_agents, 1, -1),
                                              *rnn_hidden,
                                              avail_actions=avail_actions.reshape(batch_agents, 1, -1))
            actions = dists.stochastic_sample()
            log_pi_a = dists.log_prob(actions).reshape(batch_size, self.n_agents)
            actions = actions.reshape(batch_size, self.n_agents)
        else:
            hidden_state, dists = self.policy(obs_in, agents_id, avail_actions=avail_actions)
            actions = dists.stochastic_sample()
            log_pi_a = dists.log_prob(actions)
        return hidden_state, actions.detach().cpu().numpy(), log_pi_a.detach().cpu().numpy()

    def values(self, obs_n, *rnn_hidden, state=None):
        batch_size = len(obs_n)
        agents_id = torch.eye(self.n_agents).unsqueeze(0).expand(batch_size, -1, -1).to(self.device)
        # build critic input
        if self.use_global_state:
            state = torch.Tensor(state).unsqueeze(1).to(self.device)
            critic_in = state.expand(-1, self.n_agents, -1)
        else:
            critic_in = torch.Tensor(obs_n).to(self.device)
        # get critic values
        if self.use_recurrent:
            hidden_state, values_n = self.policy.get_values(critic_in.unsqueeze(2),  # add a sequence length axis.
                                                            agents_id.unsqueeze(2),
                                                            *rnn_hidden)
            values_n = values_n.squeeze(2)
        else:
            hidden_state, values_n = self.policy.get_values(critic_in, agents_id)

        return hidden_state, values_n.detach().cpu().numpy()

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
