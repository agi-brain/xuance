from xuance.tensorflow.agents import *


class MATD3_Agents(MARLAgents):
    def __init__(self,
                 config: Namespace,
                 envs: DummyVecEnv_Pettingzoo,
                 device: str = "cpu:0"):
        self.gamma = config.gamma

        input_representation = get_repre_in(config)
        representation = REGISTRY_Representation[config.representation](*input_representation)
        input_policy = get_policy_in_marl(config, representation)
        policy = REGISTRY_Policy[config.policy](*input_policy)
        lr_scheduler = [MyLinearLR(config.lr_a, start_factor=1.0, end_factor=0.5,
                                   total_iters=get_total_iters(config.agent_name, config)),
                        MyLinearLR(config.lr_c, start_factor=1.0, end_factor=0.5,
                                   total_iters=get_total_iters(config.agent_name, config))]
        optimizer = [tk.optimizers.Adam(lr_scheduler[0]),
                     tk.optimizers.Adam(lr_scheduler[1])]
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
        learner = MATD3_Learner(config, policy, optimizer,
                                config.device, config.model_dir, config.gamma)
        super(MATD3_Agents, self).__init__(config, envs, policy, memory, learner, device,
                                           config.log_dir, config.model_dir)
        self.on_policy = False

    def act(self, obs_n, test_mode):
        batch_size = len(obs_n)
        with tf.device(self.device):
            agents_id = tf.tile(tf.expand_dims(tf.eye(self.n_agents), axis=0), multiples=(batch_size, 1, 1))
            inputs_policy = {"obs": tf.convert_to_tensor(obs_n), "ids": agents_id}
            _, actions = self.policy(inputs_policy)
        actions = actions.numpy()
        if test_mode:
            return None, actions
        else:
            actions += np.random.normal(0, self.args.sigma, size=actions.shape)
            return None, actions

    def train(self, i_episode):
        sample = self.memory.sample()
        info_train = self.learner.update(sample)
        return info_train
