from xuanpolicy.torch.agents import *
import socket
import time
from pathlib import Path


class MARLAgents(object):
    def __init__(self,
                 config: Namespace,
                 envs: DummyVecEnv_Pettingzoo,
                 policy: nn.Module,
                 memory: BaseBuffer,
                 learner: LearnerMAS,
                 device: Optional[Union[str, int, torch.device]] = None,
                 logdir: str = "./logs/",
                 modeldir: str = "./models/",
                 ):
        self.args = config
        self.handle = config.handle
        self.n_agents = config.n_agents
        self.agent_keys = config.agent_keys
        self.agent_index = config.agent_ids
        self.dim_obs = self.args.dim_obs
        self.dim_act = self.args.dim_act
        self.dim_id = self.n_agents
        self.device = torch.device("cuda" if (torch.cuda.is_available() and config.device in ["gpu", "cuda:0"]) else "cpu")
        self.envs = envs

        if config.logger == "tensorboard":
            time_string = time.asctime().replace(" ", "").replace(":", "_")
            log_dir = os.path.join(os.getcwd(), config.logdir) + "/" + time_string
            if not os.path.exists(log_dir):
                os.makedirs(log_dir)
            self.writer = SummaryWriter(log_dir)
            self.use_wandb = False
        elif config.logger == "wandb":
            config_dict = vars(config)
            wandb_dir = Path(os.path.join(os.getcwd(), config.logdir))
            if not wandb_dir.exists():
                os.makedirs(str(wandb_dir))
            wandb.init(config=config_dict,
                       project=config.project_name,
                       entity=config.wandb_user_name,
                       notes=socket.gethostname(),
                       dir=wandb_dir,
                       group=config.env,
                       job_type=config.agent,
                       name=time.asctime(),
                       reinit=True
                       )
            os.environ["WANDB_SILENT"] = "True"
            self.use_wandb = True
        else:
            raise "No logger is implemented."

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

    def log_infos(self, info: dict, x_index: int):
        """
        info: (dict) information to be visualized
        n_steps: current step
        """
        if self.use_wandb:
            for k, v in info.items():
                wandb.log({k: v}, step=x_index)
        else:
            for k, v in info.items():
                try:
                    self.writer.add_scalar(k, v, x_index)
                except:
                    self.writer.add_scalars(k, v, x_index)

    def save_model(self):
        self.learner.save_model()

    def load_model(self, path):
        self.learner.load_model(path)

    def act(self, obs_n, test_mode, noise=False):
        if not test_mode:
            epsilon = self.epsilon_decay.epsilon
        else:
            epsilon = 1.0
        batch_size = obs_n.shape[0]
        agents_id = torch.eye(self.n_agents).unsqueeze(0).expand(batch_size, -1, -1).to(self.device)

        obs_in = torch.Tensor(obs_n).view([batch_size, self.n_agents, -1]).to(self.device)

        _, greedy_actions, _ = self.policy(obs_in, agents_id)
        greedy_actions = greedy_actions.cpu().detach().numpy()
        if noise:
            random_variable = np.random.random(greedy_actions.shape)
            action_pick = np.int32((random_variable < epsilon))
            random_actions = np.array([[self.args.action_space[agent].sample() for agent in self.agent_keys]])
            return action_pick * greedy_actions + (1 - action_pick) * random_actions
        else:
            return greedy_actions

    def train(self, i_episode):
        raise NotImplementedError


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


class RandomAgents(object):
    def __init__(self, args, envs, device=None):
        self.args = args
        self.n_agents = self.args.n_agents
        self.agent_keys = args.agent_keys
        self.action_space = self.args.action_space
        self.nenvs = envs.num_envs

    def act(self, obs_n, episode, test_mode, noise=False):
        rand_a = [[self.action_space[agent].sample() for agent in self.agent_keys] for e in range(self.nenvs)]
        random_actions = np.array(rand_a)
        return random_actions

    def load_model(self, model_dir):
        return
