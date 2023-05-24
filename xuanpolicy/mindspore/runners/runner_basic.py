import time

from xuanpolicy.environment import make_envs


class Runner_Base(object):
    def __init__(self, args):
        # build environments
        if hasattr(args, 'continuous_action'):
            self.envs = make_envs(args.env_name, args.env_id, args.seed, args.vectorize, args.parallels,
                                  args.continuous_action, args.render_mode)
        else:
            self.envs = make_envs(args.env_name, args.env_id, args.seed, args.vectorize, args.parallels,
                                  args.render_mode)

        if args.vectorize != 'NOREQUIRED':
            self.n_envs = self.envs.num_envs

        self.train_at_step = args.train_at_step

    def run(self):
        pass

    def tb_load(self, data):
        return


class Runner_Base_MARL(Runner_Base):
    def __init__(self, args):
        if args.test_mode:
            args.render_mode = 'human'
        super(Runner_Base_MARL, self).__init__(args)
        # build environments
        self.n_handles = len(self.envs.handles)
        self.loss = [[0.0] for _ in range(self.n_handles)]

        self.agent_keys = self.envs.agent_keys
        self.agent_ids = self.envs.agent_ids
        self.agent_keys_all = self.envs.keys
        self.n_agents_all = len(self.agent_keys_all)
        self.render = args.render
        self.render_delay = args.render_delay

        self.train_at_step = args.train_at_step
        self.n_episodes = args.training_steps
        self.n_tests = args.n_tests
        self.test_period = args.test_period
        self.test_mode = args.test_mode
        self.marl_agents = []
        self.marl_names = []

    def combine_env_actions(self, actions):
        actions_envs = []
        for e in range(self.n_envs):
            act_handle = {}
            for h, keys in enumerate(self.agent_keys):
                act_handle.update({agent_name: actions[h][e][i] for i, agent_name in enumerate(keys)})
            actions_envs.append(act_handle)
        return actions_envs

    def get_actions(self, obs_n, episode, test_mode, act_mean_last, agent_mask, state):
        actions_n, log_pi_n, values_n, actions_n_onehot = [], [], [], []
        act_mean_current = act_mean_last
        for h, mas_group in enumerate(self.marl_agents):
            if self.marl_names[h] == "MFQ":
                a, a_mean = mas_group.act(obs_n[h], episode, test_mode, act_mean_last[h], agent_mask[h],
                                          noise=(not test_mode))
                act_mean_current[h] = a_mean
            elif self.marl_names[h] == "MFAC":
                a, a_mean = mas_group.act(obs_n[h], episode, test_mode, act_mean_last[h], agent_mask[h],
                                                  noise=(not test_mode))
                act_mean_current[h] = a_mean
            elif self.marl_names[h] in ["MAPPO_KL", "MAPPO_Clip", "CID_Simple"]:
                a, log_pi, values = mas_group.act(obs_n[h], episode, test_mode, state=state, noise=(not test_mode))
                log_pi_n.append(log_pi)
                values_n.append(values)
            elif self.marl_names[h] in ["VDAC"]:
                a, values = mas_group.act(obs_n[h], episode, test_mode, state=state, noise=(not test_mode))
                values_n.append(values)
                log_pi_n.append(None)
            elif self.marl_names[h] in ["COMA"]:
                a, a_onehot = mas_group.act(obs_n[h], episode, test_mode, noise=(not test_mode))
                actions_n_onehot.append(a_onehot)
            else:
                a = mas_group.act(obs_n[h], episode, test_mode, noise=(not test_mode))
            actions_n.append(a)
        return {'actions_n': actions_n, 'log_pi': log_pi_n, 'act_mean': act_mean_current,
                'act_n_onehot': actions_n_onehot, 'values': values_n}

    def run_episode(self, episode, test_mode=False):
        return

    def print_infos(self, args):
        infos = []
        for h, arg in enumerate(args):
            agent_name = self.envs.agent_keys[h][0][0:-2]
            if arg.n_agents == 1:
                infos.append(agent_name + ": {} agent".format(arg.n_agents) + ", {}".format(arg.agent))
            else:
                infos.append(agent_name + ": {} agents".format(arg.n_agents) + ", {}".format(arg.agent))
        print(infos)
        time.sleep(0.01)
