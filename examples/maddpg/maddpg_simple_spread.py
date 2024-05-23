import os
import socket
import time
from pathlib import Path
import wandb
import argparse
from copy import deepcopy
import numpy as np
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from xuance import get_arguments
from xuance.environment import make_envs
from xuance.torch.utils.operations import set_seed
from xuance.common import get_time_string


def parse_args():
    parser = argparse.ArgumentParser("Example of XuanCe: MADDPG.")
    parser.add_argument("--method", type=str, default="maddpg")
    parser.add_argument("--env", type=str, default="mpe")
    parser.add_argument("--env-id", type=str, default="simple_spread_v3")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--test", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--benchmark", type=int, default=1)
    parser.add_argument("--config", type=str, default="./maddpg_simple_spread_config.yaml")

    return parser.parse_args()


class Runner(object):
    def __init__(self, args):
        # set random seeds
        set_seed(args.seed)

        # prepare directories
        self.args = args
        self.args.agent_name = args.agent
        time_string = get_time_string()
        folder_name = f"seed_{args.seed}_" + time_string
        self.args.model_dir_load = self.args.model_dir
        self.args.model_dir_save = os.path.join(os.getcwd(), self.args.model_dir, folder_name)

        # Logger
        if self.args.logger == "tensorboard":
            log_dir = os.path.join(os.getcwd(), self.args.log_dir, folder_name)
            if not os.path.exists(log_dir):
                os.makedirs(log_dir)
            self.writer = SummaryWriter(log_dir)
            self.use_wandb = False
        else:
            config_dict = vars(self.args)
            wandb_dir = Path(os.path.join(os.getcwd(), self.args.log_dir))
            if not wandb_dir.exists():
                os.makedirs(str(wandb_dir))
            wandb.init(config=config_dict,
                       project=self.args.project_name,
                       entity=self.args.wandb_user_name,
                       notes=socket.gethostname(),
                       dir=wandb_dir,
                       group=self.args.env_id,
                       job_type=self.args.agent,
                       name=time_string,
                       reinit=True)
            self.use_wandb = True

        # build environments
        self.envs = make_envs(args)
        self.n_envs = self.envs.num_envs
        self.fps = 20
        self.agent_keys = self.args.agent_keys = self.envs.agent_keys[0]
        self.episode_length = self.envs.max_episode_length
        self.render = self.args.render

        # environment details, representations, policies, optimizers, and agents.\
        self.args.n_agents = self.envs.n_agents[0]
        self.args.observation_space = self.envs.observation_space
        self.args.obs_shape = self.envs.observation_space[self.agent_keys[0]].shape
        self.args.dim_obs = self.args.obs_shape[0]
        self.args.dim_act = self.envs.action_space[self.agent_keys[0]].shape[0]
        self.args.act_shape = (self.args.dim_act,)
        self.args.action_space = self.envs.action_space
        self.args.state_space = None
        self.args.rew_shape, self.args.done_shape = (self.args.n_agents, 1), (self.args.n_agents,)
        from xuance.torch.agents import MADDPG_Agents
        self.agents = MADDPG_Agents(self.args, self.envs, self.args.device)
        self.current_step, self.current_episode = 0, np.zeros((self.envs.num_envs,), np.int32)

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

    def log_videos(self, info: dict, fps: int, x_index: int = 0):
        if self.use_wandb:
            for k, v in info.items():
                wandb.log({k: wandb.Video(v, fps=fps, format='gif')}, step=x_index)
        else:
            for k, v in info.items():
                self.writer.add_video(k, v, fps=fps, global_step=x_index)

    def combine_env_actions(self, actions):
        num_envs = actions.shape[0]
        actions_envs = [{k: actions[e][i] for i, k in enumerate(self.agent_keys)} for e in range(num_envs)]
        return actions_envs

    def get_actions(self, obs_n, test_mode):
        _, a = self.agents.act(obs_n, test_mode=test_mode)
        return {'actions_n': a}

    def store_data(self, obs_n, next_obs_n, actions_dict, agent_mask, rew_n, done_n):
        data_step = {'obs': obs_n[0], 'obs_next': next_obs_n[0], 'actions': actions_dict['actions_n'],
                     'rewards': rew_n[0], 'agent_mask': agent_mask[0], 'terminals': done_n[0]}
        self.agents.memory.store(data_step)

    def train_episode(self, n_episodes):
        episode_score = np.zeros([self.n_envs, 1], dtype=np.float32)
        for _ in tqdm(range(n_episodes)):
            obs_n = self.envs.buf_obs
            for step in range(self.episode_length):
                actions_dict = self.get_actions(obs_n[0], test_mode=False)
                actions_execute = self.combine_env_actions(actions_dict['actions_n'])
                next_obs_n, rew_n, terminated_n, truncated_n, infos = self.envs.step(actions_execute)
                agent_mask = self.envs.agent_mask()
                self.store_data(obs_n, next_obs_n, actions_dict, agent_mask, rew_n, terminated_n)
                obs_n = deepcopy(next_obs_n)

                episode_score += np.mean(rew_n[0] * agent_mask[0][:, :, np.newaxis], axis=1)
                terminal_handle = terminated_n[0].all(axis=-1)
                truncate_handle = truncated_n[0].all(axis=-1)

                for i_env in range(self.n_envs):
                    if terminal_handle[i_env] or truncate_handle[i_env]:
                        self.current_episode[i_env] += 1
                        obs_n[0][i_env] = infos[i_env]["reset_obs"][0]
                        agent_mask[0][i_env] = infos[i_env]["reset_agent_mask"][0]
                        episode_score[i_env] = np.mean(infos[i_env]["individual_episode_rewards"])
                self.current_step += self.n_envs

            # train the model for each episode
            train_info = self.agents.train(self.current_step)
            episode_info = {"Train_Episode_Score": episode_score[0].mean()}
            self.log_infos(episode_info, self.current_step)
            self.log_infos(train_info, self.current_step)

    def test_episode(self, env_fn):
        test_envs = env_fn()
        num_envs = test_envs.num_envs
        episode_videos = []
        obs_n, infos = test_envs.reset()
        if self.args.render_mode == "rgb_array" and self.render:
            images = test_envs.render(self.args.render_mode)
            episode_videos.append(images[0])
        episode_score = np.zeros([num_envs, 1], dtype=np.float32)

        for step in range(self.episode_length):
            actions_dict = self.get_actions(obs_n[0], test_mode=True)
            actions_execute = self.combine_env_actions(actions_dict['actions_n'])
            next_obs_n, rew_n, terminated_n, truncated_n, infos = test_envs.step(actions_execute)
            if self.args.render_mode == "rgb_array" and self.render:
                images = test_envs.render(self.args.render_mode)
                episode_videos.append(images[0])

            agent_mask = test_envs.agent_mask()
            obs_n = deepcopy(next_obs_n)

            episode_score += np.mean(rew_n[0] * agent_mask[0][:, :, np.newaxis], axis=1)
            terminal_handle = terminated_n[0].all(axis=-1)
            truncate_handle = truncated_n[0].all(axis=-1)

            for i_env in range(num_envs):
                if terminal_handle[i_env] or truncate_handle[i_env]:
                    obs_n[0][i_env] = infos[i_env]["reset_obs"][0]
                    agent_mask[0][i_env] = infos[i_env]["reset_agent_mask"][0]
        scores = episode_score.mean(axis=1)
        if self.args.test_mode:
            print("Mean score: ", scores)

        if self.args.render_mode == "rgb_array" and self.render:
            # time, height, width, channel -> time, channel, height, width
            videos_info = {"Videos_Test": np.array([episode_videos], dtype=np.uint8).transpose((0, 1, 4, 2, 3))}
            self.log_videos(info=videos_info, fps=self.fps, x_index=self.current_step)

        test_info = {"Test-Episode-Rewards": scores[0]}
        self.log_infos(test_info, self.current_step)
        test_envs.close()
        return episode_score

    def run(self):
        if self.args.test_mode:
            def env_fn():
                args_test = deepcopy(self.args)
                args_test.parallels = args_test.test_episode
                return make_envs(args_test)

            self.render = True
            self.agents.load_model(path=self.args.model_dir_load)
            self.test_episode(env_fn)
            print("Finish testing.")
        else:
            n_train_episodes = self.args.running_steps // self.episode_length // self.n_envs
            self.train_episode(n_train_episodes)
            print("Finish training.")
            self.agents.save_model("final_train_model.pth")

    def benchmark(self):
        def env_fn():
            args_test = deepcopy(self.args)
            args_test.parallels = args_test.test_episode
            return make_envs(args_test)

        n_train_episodes = self.args.running_steps // self.episode_length // self.n_envs
        n_eval_interval = self.args.eval_interval // self.episode_length // self.n_envs
        num_epoch = int(n_train_episodes / n_eval_interval)

        test_scores = self.test_episode(env_fn)
        best_scores = {
            "mean": np.mean(test_scores),
            "std": np.std(test_scores),
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
        print("Best Score, Mean: ", best_scores["mean"], "Std: ", best_scores["std"])

    def finish(self):
        self.envs.close()
        if self.use_wandb:
            wandb.finish()
        else:
            self.writer.close()


if __name__ == "__main__":
    parser = parse_args()
    args = get_arguments(method=parser.method,
                         env=parser.env,
                         env_id=parser.env_id,
                         config_path=parser.config,
                         parser_args=parser,
                         is_test=parser.test)
    runner = Runner(args)

    if args.benchmark:
        runner.benchmark()
    else:
        runner.run()

    runner.finish()
