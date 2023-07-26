import numpy as np

from xuanpolicy.torch.agents import *


class MAPPO_Agents(MARLAgents):
    def __init__(self,
                 config: Namespace,
                 envs: DummyVecEnv_Pettingzoo,
                 device: Optional[Union[int, str, torch.device]] = None):
        self.gamma = config.gamma
        self.n_envs = envs.num_envs
        self.n_size = config.n_size
        self.n_epoch = config.n_epoch
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

        input_policy = get_policy_in_marl(config, representation)
        policy = REGISTRY_Policy[config.policy](*input_policy,
                                                use_centralized_V=config.use_centralized_V,
                                                use_recurrent=config.use_recurrent,
                                                rnn=config.rnn)
        optimizer = torch.optim.Adam(policy.parameters(), config.learning_rate, eps=1e-5)
        scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.5,
                                                      total_iters=get_total_iters(config.agent_name, config))
        self.observation_space = envs.observation_space
        self.action_space = envs.action_space
        self.representation_info_shape = policy.representation.output_shapes
        self.auxiliary_info_shape = {}

        buffer = MARL_OnPolicyBuffer_RNN if self.use_recurrent else MARL_OnPolicyBuffer
        input_buffer = (config.n_agents, config.state_space.shape, config.obs_shape, config.act_shape, config.rew_shape,
                        config.done_shape, envs.num_envs, config.n_size,
                        config.use_gae, config.use_advnorm, config.gamma, config.gae_lambda)
        memory = buffer(*input_buffer, max_episode_length=envs.max_episode_length, dim_act=config.dim_act)
        self.buffer_size = memory.buffer_size
        self.batch_size = self.buffer_size // self.n_epoch
        learner = MAPPO_Clip_Learner(config, policy, optimizer, scheduler,
                                     config.device, config.model_dir, config.gamma)
        super(MAPPO_Agents, self).__init__(config, envs, policy, memory, learner, device,
                                           config.log_dir, config.model_dir)
        self.share_values = True if config.rew_shape[0] == 1 else False

    def act(self, obs_n, *rnn_hidden, avail_actions=None, state=None, test_mode=False):
        batch_size = len(obs_n)
        agents_id = torch.eye(self.n_agents).unsqueeze(0).expand(batch_size, -1, -1).to(self.device)
        obs_in = torch.Tensor(obs_n).view([batch_size, self.n_agents, -1]).to(self.device)
        if self.use_recurrent:
            hidden_state, dists = self.policy(obs_in, agents_id, *rnn_hidden, avail_actions=avail_actions)
        else:
            hidden_state, dists = self.policy(obs_in, agents_id, avail_actions=avail_actions)
        actions = dists.stochastic_sample()
        log_pi_a = dists.log_prob(actions)
        return hidden_state, actions.detach().cpu().numpy(), log_pi_a.detach().cpu().numpy()

    def values(self, obs_n, *rnn_hidden, state=None):
        batch_size = len(obs_n)
        agents_id = torch.eye(self.n_agents).unsqueeze(0).expand(batch_size, -1, -1).to(self.device)
        obs_in = torch.Tensor(obs_n).view([batch_size, self.n_agents, -1]).to(self.device)
        if self.use_recurrent:
            hidden_state, values_n = self.policy.get_values(obs_in, agents_id, *rnn_hidden)
        else:
            hidden_state, values_n = self.policy.get_values(obs_in, agents_id)
        values = values_n.mean(dim=1) if self.share_values else values_n

        return hidden_state, values.detach().cpu().numpy()

    def train(self, i_episode):
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
            self.memory.clear()
            return info_train
        else:
            return {}
