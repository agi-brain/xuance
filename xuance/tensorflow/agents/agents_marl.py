from xuance.tensorflow.agents import *


class MARLAgents(object):
    def __init__(self,
                 config: Namespace,
                 envs: DummyVecEnv_Pettingzoo,
                 policy: tk.Model,
                 memory: BaseBuffer,
                 learner: LearnerMAS,
                 device: str = "cpu:0",
                 logdir: str = "./logs/",
                 modeldir: str = "./models/"):
        self.args = config
        self.handle = config.handle
        self.n_agents = config.n_agents
        self.agent_keys = config.agent_keys
        self.agent_index = config.agent_ids
        self.dim_obs = self.args.dim_obs
        self.dim_act = self.args.dim_act
        self.dim_id = self.n_agents
        self.device = device

        self.envs = envs
        self.render = config.render
        self.nenvs = envs.num_envs
        self.policy = policy
        self.memory = memory
        self.learner = learner
        self.device = device
        self.logdir = logdir
        self.modeldir = modeldir
        create_directory(logdir)
        create_directory(modeldir)

    def save_model(self):
        self.learner.save_model()

    def load_model(self, path):
        self.learner.load_model(path)

    def act(self, obs_n, episode, test_mode, noise=False):
        if not test_mode:
            epsilon = self.epsilon_decay.epsilon
        else:
            epsilon = 1.0
        batch_size = obs_n.shape[0]
        # agents_id = np.tile(np.expand_dims(np.eye(self.n_agents), 0), (batch_size, 1, 1)).reshape([-1, self.n_agents])
        # obs_in = obs_n.reshape([batch_size * self.n_agents, -1])
        inputs = {"obs": obs_n,
                  "ids": np.tile(np.expand_dims(np.eye(self.n_agents), 0), (batch_size, 1, 1))}
        _, greedy_actions, _ = self.policy(inputs)

        greedy_actions = greedy_actions.numpy()
        if noise:
            random_variable = np.random.random(greedy_actions.shape)
            action_pick = np.int32((random_variable < epsilon))
            random_actions = np.array([[self.args.action_space[agent].sample() for agent in self.agent_keys]])
            return action_pick * greedy_actions + (1 - action_pick) * random_actions
        else:
            return greedy_actions

    def train(self, i_episode):
        return


class linear_decay_or_increase(object):
    def __init__(self, start, end, step_length):
        self.start = start
        self.end = end
        self.step_length = step_length
        if self.start > self.end:
            self.is_decay = True
            self.delta = (self.start - self.end) / self.step_length
        else:
            self.is_decay = False
            self.delta = (self.end - self.start) / self.step_length
        self.epsilon = start

    def update(self):
        if self.is_decay:
            self.epsilon = max(self.epsilon - self.delta, self.end)
        else:
            self.epsilon = min(self.epsilon + self.delta, self.end)


def get_total_iters(agent_name, args):
    if agent_name in ["A2C", "PG", "PPO_Clip", "PPO_KL", "PPG", "VDAC", "COMA", "MFAC", "MAPPO_Clip", "MAPPO_KL"]:
        return int(args.training_steps * args.nepoch * args.nminibatch / args.nsteps)
    else:
        return int(args.training_steps / args.training_frequency)


class RandomAgents(MARLAgents):
    def __init__(self, args):
        super(RandomAgents, self).__init__(args)

    def act(self, obs_n, episode, test_mode, noise=False):
        random_actions = np.array([[self.args.action_space[agent].sample() for agent in self.agent_keys]])
        return random_actions
