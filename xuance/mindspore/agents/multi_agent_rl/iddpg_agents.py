from xuance.mindspore.agents import *


class IDDPG_Agents(MARLAgents):
    def __init__(self,
                 config: Namespace,
                 envs: DummyVecEnv_Pettingzoo):
        self.gamma = config.gamma

        input_representation = get_repre_in(config)
        representation = REGISTRY_Representation[config.representation](*input_representation)
        input_policy = get_policy_in_marl(config, representation)
        policy = REGISTRY_Policy[config.policy](*input_policy)
        scheduler = [lr_decay_model(learning_rate=config.lr_a, decay_rate=0.5,
                                    decay_steps=get_total_iters(config.agent_name, config)),
                     lr_decay_model(learning_rate=config.lr_c, decay_rate=0.5,
                                    decay_steps=get_total_iters(config.agent_name, config))]
        optimizer = [Adam(policy.parameters_actor, scheduler[0], eps=1e-5),
                     Adam(policy.parameters_critic, scheduler[1], eps=1e-5)]
        self.observation_space = envs.observation_space
        self.action_space = envs.action_space
        self.representation_info_shape = policy.representation.output_shapes
        self.auxiliary_info_shape = {}

        if config.state_space is not None:
            config.dim_state, state_shape = config.state_space.shape, config.state_space.shape
        else:
            config.dim_state, state_shape = None, None
        memory = MARL_OffPolicyBuffer(config.n_agents,
                                      state_shape,
                                      config.obs_shape,
                                      config.act_shape,
                                      config.rew_shape,
                                      config.done_shape,
                                      envs.num_envs,
                                      config.buffer_size,
                                      config.batch_size)
        learner = IDDPG_Learner(config, policy, optimizer, scheduler, config.model_dir, config.gamma)
        super(IDDPG_Agents, self).__init__(config, envs, policy, memory, learner, config.log_dir, config.model_dir)
        self.on_policy = False

    def act(self, obs_n, test_mode):
        batch_size = len(obs_n)
        agents_id = ops.broadcast_to(self.expand_dims(self.eye(self.n_agents, self.n_agents, ms.float32), 0),
                                     (batch_size, -1, -1))
        _, actions = self.policy(Tensor(obs_n), agents_id)
        actions = actions.asnumpy()
        if not test_mode:
            actions += np.random.normal(0, self.args.sigma, size=actions.shape)
        return None, actions

    def train(self, i_episode):
        sample = self.memory.sample()
        info_train = self.learner.update(sample)
        return info_train
