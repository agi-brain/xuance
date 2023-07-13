import copy
import os
import socket
from pathlib import Path
from .runner_basic import Runner_Base, make_envs
from xuanpolicy.torch.agents import REGISTRY as REGISTRY_Agent
import wandb
from torch.utils.tensorboard import SummaryWriter
import time
import numpy as np
from copy import deepcopy


class SC2_Runner(Runner_Base):
    def __init__(self, args):
        super(SC2_Runner, self).__init__(args)
        self.args = args
        self.render = args.render
        if args.logger == "tensorboard":
            time_string = time.asctime().replace(" ", "").replace(":", "_")
            log_dir = os.path.join(os.getcwd(), args.logdir) + "/" + time_string
            if not os.path.exists(log_dir):
                os.makedirs(log_dir)
            self.writer = SummaryWriter(log_dir)
            self.use_wandb = False
        elif args.logger == "wandb":
            config_dict = vars(args)
            wandb_dir = Path(os.path.join(os.getcwd(), args.logdir))
            if not wandb_dir.exists():
                os.makedirs(str(wandb_dir))
            wandb.init(config=config_dict,
                       project=args.project_name,
                       entity=args.wandb_user_name,
                       notes=socket.gethostname(),
                       dir=wandb_dir,
                       group=args.env_id,
                       job_type=args.agent,
                       name=time.asctime(),
                       reinit=True
                       )
            # os.environ["WANDB_SILENT"] = "True"
            self.use_wandb = True
        else:
            raise "No logger is implemented."

        self.current_step = 0
        self.current_episode = np.zeros((self.envs.num_envs,), np.int32)
        self.episode_length = self.envs.max_episode_length
        self.rnn_hidden = None
        self.num_agents = args.n_agents = self.envs.num_agents
        self.dim_obs, self.dim_act, self.dim_state = self.envs.dim_obs, self.envs.dim_act, self.envs.dim_state
        args.dim_obs, args.dim_act = self.dim_obs, self.dim_act
        args.obs_shape = (self.num_agents, self.dim_obs)
        args.act_shape = (self.num_agents, )
        args.rew_shape, args.done_shape, args.act_prob_shape = (self.num_agents, 1), (self.num_agents,), (args.dim_act,)
        args.action_space = self.envs.action_space
        args.state_space = self.envs.state_space

        # environment details, representations, policies, optimizers, and agents.
        self.agents = REGISTRY_Agent[args.agent](args, self.envs, args.device)

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

    def log_videos(self, info: dict, fps: int, x_index: int=0):
        if self.use_wandb:
            for k, v in info.items():
                wandb.log({k: wandb.Video(v, fps=fps, format='gif')}, step=x_index)
        else:
            for k, v in info.items():
                self.writer.add_video(k, v, fps=fps, global_step=x_index)

    def get_actions(self, obs_n, avail_actions, test_mode):
        log_pi_n, values_n, actions_n_onehot = [], [], []
        self.rnn_hidden, actions_n = self.agents.act(obs_n, *self.rnn_hidden,
                                                     avail_actions=avail_actions, test_mode=test_mode)
        return {'actions_n': actions_n, 'log_pi': log_pi_n,
                'act_n_onehot': actions_n_onehot, 'values': values_n}

    def store_data(self, obs_n, actions_dict, state, rewards, filled, avail_actions):
        data_step = {
            "obs": obs_n,
            "actions": actions_dict['actions_n'],
            "state": state,
            "reward": rewards,
            "terminals": filled,
            "avai_actions": avail_actions
        }
        self.agents.memory.store(data_step)

    def train_episode(self, n_episodes):
        episode_score = np.zeros([self.n_envs, 1], dtype=np.float32)
        episode_info, train_info = {}, {}
        for i_episode in range(n_episodes):
            obs_n, state, _ = self.envs.reset()
            self.rnn_hidden = self.agents.policy.representation.init_hidden(self.n_envs)
            filled = np.zeros([self.n_envs, 1], dtype=np.bool)
            for step in range(self.episode_length):
                available_actions = self.envs.get_avail_actions()
                actions_dict = self.get_actions(obs_n, available_actions, False)
                next_obs_n, next_state, rewards, terminated, truncated, info = self.envs.step(actions_dict['actions_n'])

                self.store_data(obs_n, actions_dict, state, rewards, filled, available_actions)

                obs_n, state = deepcopy(next_obs_n), deepcopy(next_state)
                for i_env in range(self.n_envs):
                    if terminated[i_env] or truncated[i_env]:
                        filled[i_env] = True


    def test_episode(self, n_episodes):
        return 0

    def run(self):
        if self.args.test_mode:
            def env_fn():
                arg_test = copy.deepcopy(self.args)
                arg_test.parallels = arg_test.test_episode
                return make_envs(arg_test)
            self.render = True
            self.agents.load_model(self.agents.modeldir)
            self.test_episode(env_fn)
            print("Finish testing.")
        else:
            n_train_episodes = self.args.running_steps // self.episode_length // self.n_envs
            self.train_episode(n_train_episodes)
            print("Finish training.")
            self.agents.save_model("final_train_model.pth")

        self.envs.close()
        if self.use_wandb:
            wandb.finish()
        else:
            self.writer.close()

    def benchmark(self):
        def env_fn():
            arg_test = copy.deepcopy(self.args)
            arg_test.parallels = arg_test.test_episode
            return make_envs(arg_test)

        n_train_episodes = self.args.running_steps // self.episode_length // self.n_envs
        n_eval_interval = self.args.eval_interval // self.episode_length // self.n_envs
        num_epoch = int(n_train_episodes / n_eval_interval)

        test_scores = self.test_episode(env_fn)
        best_scores = {
            "mean": np.mean(test_scores, axis=1),
            "std": np.std(test_scores, axis=1),
            "step": self.current_step
        }

        for i_epoch in range(num_epoch):
            print("Epoch: %d/%d:" % (i_epoch, num_epoch))
            self.train_episode(n_episodes=n_eval_interval)
            test_scores = self.test_episode(env_fn)

            mean_test_scores = np.mean(test_scores)
            if mean_test_scores > best_scores["mean"]:
                best_scores = {
                    "mean": mean_test_scores,
                    "std": np.std(test_scores),
                    "step": self.current_step
                }
            # save best model
            self.agents.save_model("best_model.pth")

        # end benchmarking
        print("Finish benchmarking.")
        print("Best Score: ", best_scores["mean"], "Std: ", best_scores["std"])

        self.envs.close()
        if self.use_wandb:
            wandb.finish()
        else:
            self.writer.close()

