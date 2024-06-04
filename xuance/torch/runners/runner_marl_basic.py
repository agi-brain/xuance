"""
This is demo of runner for cooperative multi-agent reinforcement learning.
"""
import os
import socket
from pathlib import Path
import wandb
from torch.utils.tensorboard import SummaryWriter
from .runner_basic import Runner_Base, make_envs
from xuance.torch.agents import REGISTRY_Agents
from xuance.common import get_time_string
from tqdm import tqdm
import numpy as np
from gymnasium.spaces import Box, Discrete
from copy import deepcopy


class Runner_MARL(Runner_Base):
    def __init__(self, args):
        super(Runner_MARL, self).__init__(args)
        self.args = args
        self.render = args.render
        self.fps = args.fps

        time_string = get_time_string()
        seed = f"seed_{self.args.seed}_"
        self.args.model_dir_load = args.model_dir
        self.args.model_dir_save = os.path.join(os.getcwd(), args.model_dir, seed + time_string)

        if args.logger == "tensorboard":
            log_dir = os.path.join(os.getcwd(), args.log_dir, seed + time_string)
            if not os.path.exists(log_dir):
                os.makedirs(log_dir)
            self.writer = SummaryWriter(log_dir)
            self.use_wandb = False
        elif args.logger == "wandb":
            config_dict = vars(args)
            wandb_dir = Path(os.path.join(os.getcwd(), args.log_dir))
            if not wandb_dir.exists():
                os.makedirs(str(wandb_dir))
            wandb.init(config=config_dict,
                       project=args.project_name,
                       entity=args.wandb_user_name,
                       notes=socket.gethostname(),
                       dir=wandb_dir,
                       group=args.env_id,
                       job_type=args.agent,
                       name=time_string,
                       reinit=True)
            self.use_wandb = True
        else:
            raise RuntimeError(f"The logger named {args.logger} is implemented!")

        self.running_steps = args.running_steps
        self.train_per_step = args.train_per_step
        self.training_frequency = args.training_frequency
        self.current_step = 0
        self.env_step = 0
        self.current_episode = np.zeros((self.envs.num_envs,), np.int32)
        self.episode_length = self.envs.max_episode_steps
        self.num_agents = self.envs.num_agents
        args.n_agents = self.num_agents
        self.dim_obs = args.dim_obs = self.envs.obs_shape[-1]
        self.dim_state = self.envs.dim_state

        args.rew_shape = (self.num_agents, 1)
        args.done_shape = (self.num_agents, )
        args.observation_space = self.envs.observation_space
        args.action_space = self.envs.action_space
        args.state_space = self.envs.state_space

        args.obs_shape = (self.dim_obs,)
        if isinstance(args.action_space, Box):
            self.dim_act = args.dim_act = args.action_space.shape[-1]
            args.act_shape = (args.dim_act, )
        else:
            self.dim_act = args.dim_act = args.action_space.n
            args.act_shape = ()

        # Create MARL agents.
        self.agents = REGISTRY_Agents[args.agent](args, self.envs, args.device)
        self.on_policy = self.agents.on_policy

    def log_infos(self, info: dict, x_index: int):
        """
        info: (dict) information to be visualized
        n_steps: current step
        """
        if x_index <= self.running_steps:
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
        if x_index <= self.running_steps:
            if self.use_wandb:
                for k, v in info.items():
                    wandb.log({k: wandb.Video(v, fps=fps, format='gif')}, step=x_index)
            else:
                for k, v in info.items():
                    self.writer.add_video(k, v, fps=fps, global_step=x_index)

    def finish(self):
        self.envs.close()
        if self.use_wandb:
            wandb.finish()
        else:
            self.writer.close()

    def get_actions(self, obs_n, test_mode, act_mean_last, agent_mask, state):
        log_pi, a_onehot, values = None, None, None
        act_mean_current = act_mean_last
        if self.args.agent == "MFQ":
            _, a, a_mean = self.agents.act(obs_n, test_mode=test_mode, act_mean=act_mean_last, agent_mask=agent_mask)
            act_mean_current = a_mean
        elif self.args.agent == "MFAC":
            a, a_mean = self.agents.act(obs_n, test_mode, act_mean_last, agent_mask)
            act_mean_current = a_mean
            _, values = self.agents.values(obs_n, act_mean_current)
        elif self.args.agent == "VDAC":
            _, a, values = self.agents.act(obs_n, state=state, test_mode=test_mode)
        elif self.args.agent in ["MAPPO", "IPPO"]:
            _, a, log_pi = self.agents.act(obs_n, test_mode=test_mode, state=state)
            _, values = self.agents.values(obs_n, state=state)
        elif self.args.agent in ["COMA"]:
            _, a, a_onehot = self.agents.act(obs_n, test_mode)
            _, values = self.agents.values(obs_n, state=state, actions_n=a, actions_onehot=a_onehot)
        else:
            _, a = self.agents.act(obs_n, test_mode=test_mode)
        return {'actions_n': a, 'log_pi': log_pi, 'act_mean': act_mean_current,
                'act_n_onehot': a_onehot, 'values': values}

    def store_data(self, obs_n, next_obs_n, actions_dict, state, next_state, agent_mask, rew_n, done_n):
        data_step = {'obs': obs_n, 'obs_next': next_obs_n, 'actions': actions_dict['actions_n'],
                     'state': state, 'state_next': next_state, 'rewards': rew_n,
                     'agent_mask': agent_mask, 'terminals': done_n}
        if self.on_policy:
            data_step['values'] = actions_dict['values']
            if self.args.agent == "MAPPO":
                data_step['log_pi_old'] = actions_dict['log_pi']
            elif self.args.agent == "COMA":
                data_step['actions_onehot'] = actions_dict['act_n_onehot']
            self.agents.memory.store(data_step)
            if self.agents.memory.full:
                if self.args.agent == "COMA":
                    _, values_next = self.agents.values(next_obs_n,
                                                        state=next_state,
                                                        actions_n=actions_dict['actions_n'],
                                                        actions_onehot=actions_dict['act_n_onehot'])
                elif self.args.agent == "MFAC":
                    _, values_next = self.agents.values(next_obs_n, actions_dict['act_mean'])
                elif self.args.agent == "VDAC":
                    _, _, values_next = self.agents.act(next_obs_n)
                else:
                    _, values_next = self.agents.values(next_obs_n, state=next_state)
                for i_env in range(self.n_envs):
                    if done_n[i_env].all():
                        self.agents.memory.finish_path(0.0, i_env)
                    else:
                        self.agents.memory.finish_path(values_next[i_env], i_env)
        elif self.args.agent in ["MFQ", "MFAC"]:
            data_step['act_mean'] = actions_dict['act_mean']
        self.agents.memory.store(data_step)

    def train_steps(self, n_steps):
        act_mean_last = np.zeros([self.n_envs, self.args.dim_act])
        episode_score = np.zeros([self.n_envs, 1], dtype=np.float32)
        episode_info, train_info = {}, {}

        obs_n = self.envs.buf_obs
        state, agent_mask = self.envs.global_state(), self.envs.agent_mask()
        for _ in tqdm(range(n_steps)):
            step_info = {}
            actions_dict = self.get_actions(obs_n, False, act_mean_last, agent_mask, state)
            acts_execute = actions_dict['actions_n']
            next_obs_n, rew_n, terminated_n, truncated_n, infos = self.envs.step(acts_execute)
            next_state, agent_mask = self.envs.global_state(), self.envs.agent_mask()
            self.store_data(obs_n, next_obs_n, actions_dict, state, next_state, agent_mask, rew_n, terminated_n)

            # train the model for each step
            if self.train_per_step:
                if self.current_step % self.training_frequency == 0:
                    step_info.update(self.agents.train(self.current_step))

            obs_n, state, act_mean_last = deepcopy(next_obs_n), deepcopy(next_state), deepcopy(actions_dict['act_mean'])
            episode_score += np.mean(rew_n * agent_mask[:, :, np.newaxis], axis=1)

            for i_env in range(self.n_envs):
                if terminated_n.all(axis=-1)[i_env] or truncated_n.all(axis=-1)[i_env]:
                    if self.on_policy:
                        if self.args.agent == "COMA":
                            _, value_next_e = self.agents.values(next_obs_n,
                                                                 state=next_state,
                                                                 actions_n=actions_dict['actions_n'],
                                                                 actions_onehot=actions_dict['act_n_onehot'])
                        elif self.args.agent == "MFAC":
                            _, value_next_e = self.agents.values(next_obs_n, act_mean_last)
                        elif self.args.agent == "VDAC":
                            _, _, value_next_e = self.agents.act(next_obs_n)
                        else:
                            _, value_next_e = self.agents.values(next_obs_n, state=next_state)
                        self.agents.memory.finish_path(value_next_e[i_env], i_env)
                    obs_n[i_env] = infos[i_env]["reset_obs"]
                    agent_mask[i_env] = infos[i_env]["reset_agent_mask"]
                    act_mean_last[i_env] = np.zeros([self.args.dim_act])
                    episode_score[i_env] = np.mean(infos[i_env]["episode_score"])
                    state[i_env] = infos[i_env]["reset_state"]
                    self.current_episode[i_env] += 1
                    if self.use_wandb:
                        step_info["Episode-Steps/env-%d" % i_env] = infos[i_env]["episode_step"]
                        step_info["Train-Episode-Rewards/env-%d" % i_env] = np.mean(infos[i_env]["episode_score"])
                    else:
                        step_info["Episode-Steps"] = {"env-%d" % i_env: infos[i_env]["episode_step"]}
                        step_info["Train-Episode-Rewards"] = {"env-%d" % i_env: np.mean(infos[i_env]["episode_score"])}
                    self.log_infos(step_info, self.current_step)

            self.current_step += self.n_envs

            # train the model for each episode
            if not self.train_per_step:
                train_info = self.agents.train(self.current_step)
            self.log_infos(train_info, self.current_step)

    def test_episodes(self, env_fn, test_episodes):
        test_envs = env_fn()
        test_info = {}
        num_envs = test_envs.num_envs
        videos, episode_videos = [[] for _ in range(num_envs)], []
        current_episode, scores = 0, []
        obs_n, infos = test_envs.reset()
        state, agent_mask = test_envs.global_state(), test_envs.agent_mask()
        if self.args.render_mode == "rgb_array" and self.render:
            images = test_envs.render(self.args.render_mode)
            for idx, img in enumerate(images):
                videos[idx].append(img)
        act_mean_last = np.zeros([num_envs, self.args.dim_act])
        episode_score = np.zeros([num_envs, 1], dtype=np.float32)

        while current_episode < test_episodes:
            actions_dict = self.get_actions(obs_n, True, act_mean_last, agent_mask, state)
            next_obs_n, rew_n, terminated_n, truncated_n, infos = test_envs.step(actions_dict['actions_n'])
            if self.args.render_mode == "rgb_array" and self.render:
                images = test_envs.render(self.args.render_mode)
                for idx, img in enumerate(images):
                    videos[idx].append(img)

            next_state, agent_mask = test_envs.global_state(), test_envs.agent_mask()
            obs_n, state, act_mean_last = deepcopy(next_obs_n), deepcopy(next_state), deepcopy(actions_dict['act_mean'])

            for i_env in range(num_envs):
                if terminated_n.all(axis=-1)[i_env] or truncated_n.all(axis=-1)[i_env]:
                    obs_n[i_env] = infos[i_env]["reset_obs"]
                    agent_mask[i_env] = infos[i_env]["reset_agent_mask"]
                    act_mean_last[i_env] = np.zeros([self.args.dim_act])
                    episode_score[i_env] = np.mean(infos[i_env]["episode_score"])
                    state[i_env] = infos[i_env]["reset_state"]
                    current_episode += 1
                    if self.args.test_mode:
                        print("Episode: %d, Score: %.2f" % (current_episode, episode_score[i_env]))

        if self.args.render_mode == "rgb_array" and self.render:
            # time, height, width, channel -> time, channel, height, width
            videos_info = {"Videos_Test": np.array(videos, dtype=np.uint8).transpose((0, 1, 4, 2, 3))}
            self.log_videos(info=videos_info, fps=self.fps, x_index=self.current_step)

        test_info["Test-Episode-Rewards"] = np.mean(episode_score)
        self.log_infos(test_info, self.current_step)

        test_envs.close()

        return episode_score

    def run(self):
        if self.args.test_mode:
            def env_fn():
                args_test = deepcopy(self.args)
                args_test.parallels = 1  # args_test.test_episode
                args_test.render = True
                return make_envs(args_test)

            # self.render = True
            self.agents.load_model(self.agents.model_dir_load)
            self.test_episodes(env_fn, self.args.test_episode)
            print("Finish testing.")
        else:
            n_train_steps = self.args.running_steps // self.n_envs
            self.train_steps(n_train_steps)
            print("Finish training.")
            self.agents.save_model("final_train_model.pth")

        self.finish()

    def benchmark(self):
        def env_fn():
            args_test = deepcopy(self.args)
            args_test.parallels = args_test.test_episode
            return make_envs(args_test)

        train_steps = self.args.running_steps // self.n_envs
        eval_interval = self.args.eval_interval // self.n_envs
        test_episode = self.args.test_episode
        num_epoch = int(train_steps / eval_interval)

        test_scores = self.test_episodes(env_fn, test_episode)
        best_scores = {
            "mean": np.mean(test_scores),
            "std": np.std(test_scores),
            "step": self.current_step
        }
        self.agents.save_model("best_model.pth")

        for i_epoch in range(num_epoch):
            print(f"({self.args.agent}, {self.args.env_name}/{self.args.env_id}) Epoch: {i_epoch}/{num_epoch}:")
            self.train_steps(n_steps=eval_interval)
            test_scores = self.test_episodes(env_fn, test_episode)

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
        print("Best Score: ")
        print("Mean: ", best_scores["mean"], "Std: ", best_scores["std"])

        self.finish()
