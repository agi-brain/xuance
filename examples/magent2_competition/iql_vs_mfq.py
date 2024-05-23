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
from xuance.torch.agents import REGISTRY as REGISTRY_Agent
from gymnasium.spaces.box import Box

from xuance import get_arguments
from xuance.environment import make_envs
from xuance.torch.utils.operations import set_seed
from xuance.common import get_time_string


def parse_args():
    parser = argparse.ArgumentParser("Example of XuanCe: Magent2.")
    parser.add_argument("--method", type=str, default=["iql", "mfq"])
    parser.add_argument("--env", type=str, default="magent2")
    parser.add_argument("--env-id", type=str, default="adversarial_pursuit_v4")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--test", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--benchmark", type=int, default=1)
    parser.add_argument("--config", type=str, default=["./configs/iql_magent2.yaml",
                                                       "./configs/mfq_magent2.yaml"])

    return parser.parse_args()


class Runner():
    def __init__(self, args):
        # set random seeds
        set_seed(args[0].seed)

        # prepare directions
        self.args = args if type(args) == list else [args]
        self.fps = 20
        time_string = get_time_string()
        for i_method, arg in enumerate(self.args):
            arg.agent_name = arg.method[i_method]
            seed = f"seed_{arg.seed}_"
            arg.model_dir_load = arg.model_dir
            arg.model_dir_save = os.path.join(os.getcwd(), arg.model_dir, seed + time_string)

            if arg.logger == "tensorboard":
                log_dir = os.path.join(os.getcwd(), arg.log_dir, seed + time_string)
                if not os.path.exists(log_dir):
                    os.makedirs(log_dir)
                self.writer = SummaryWriter(log_dir)
                self.use_wandb = False
            else:
                self.use_wandb = True

        # build environments
        self.envs = make_envs(args[0])
        self.envs.reset()
        self.n_envs = self.envs.num_envs

        for arg in self.args:
            if arg.agent_name == "random":
                continue
            else:
                self.args_base = arg
                self.running_steps = arg.running_steps
                self.training_frequency = arg.training_frequency
                self.train_per_step = arg.train_per_step

                # build environments
                self.n_handles = len(self.envs.handles)
                self.agent_keys = self.envs.agent_keys
                self.agent_ids = self.envs.agent_ids
                self.agent_keys_all = self.envs.keys
                self.n_agents_all = len(self.agent_keys_all)
                self.render = arg.render

                self.n_steps = arg.running_steps
                self.test_mode = arg.test_mode
                self.marl_agents, self.marl_names = [], []
                self.current_step, self.current_episode = 0, np.zeros((self.envs.num_envs,), np.int32)

                if self.use_wandb:
                    config_dict = vars(arg)
                    wandb_dir = Path(os.path.join(os.getcwd(), arg.log_dir))
                    if not wandb_dir.exists():
                        os.makedirs(str(wandb_dir))
                    wandb.init(config=config_dict,
                               project=arg.project_name,
                               entity=arg.wandb_user_name,
                               notes=socket.gethostname(),
                               dir=wandb_dir,
                               group=arg.env_id,
                               job_type=arg.agent,
                               name=time_string,
                               reinit=True)
                break

        self.episode_length = self.envs.max_episode_length

        # environment details, representations, policies, optimizers, and agents.
        for h, arg in enumerate(self.args):
            arg.handle_name = self.envs.side_names[h]
            if self.n_handles > 1 and arg.agent != "RANDOM":
                arg.model_dir += "{}/".format(arg.handle_name)
            arg.handle, arg.n_agents = h, self.envs.n_agents[h]
            arg.agent_keys, arg.agent_ids = self.agent_keys[h], self.agent_ids[h]
            arg.state_space = self.envs.state_space
            arg.observation_space = self.envs.observation_space
            if isinstance(self.envs.action_space[self.agent_keys[h][0]], Box):
                arg.dim_act = self.envs.action_space[self.agent_keys[h][0]].shape[0]
                arg.act_shape = (arg.dim_act,)
            else:
                arg.dim_act = self.envs.action_space[self.agent_keys[h][0]].n
                arg.act_shape = ()
            arg.action_space = self.envs.action_space
            if arg.env_name == "MAgent2":
                arg.obs_shape = (np.prod(self.envs.observation_space[self.agent_keys[h][0]].shape),)
                arg.dim_obs = arg.obs_shape[0]
            else:
                arg.obs_shape = self.envs.observation_space[self.agent_keys[h][0]].shape
                arg.dim_obs = arg.obs_shape[0]
            arg.rew_shape, arg.done_shape, arg.act_prob_shape = (arg.n_agents, 1), (arg.n_agents,), (arg.dim_act,)
            self.marl_agents.append(REGISTRY_Agent[arg.agent](arg, self.envs, arg.device))
            self.marl_names.append(arg.agent)

        self.print_infos(self.args)

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

    def combine_env_actions(self, actions):
        actions_envs = []
        num_env = actions[0].shape[0]
        for e in range(num_env):
            act_handle = {}
            for h, keys in enumerate(self.agent_keys):
                act_handle.update({agent_name: actions[h][e][i] for i, agent_name in enumerate(keys)})
            actions_envs.append(act_handle)
        return actions_envs

    def get_actions(self, obs_n, test_mode, act_mean_last, agent_mask, state):
        actions_n, log_pi_n, values_n, actions_n_onehot = [], [], [], []
        act_mean_current = act_mean_last
        for h, mas_group in enumerate(self.marl_agents):
            if self.marl_names[h] == "MFQ":
                _, a, a_mean = mas_group.act(obs_n[h], test_mode=test_mode, act_mean=act_mean_last[h], agent_mask=agent_mask[h])
                act_mean_current[h] = a_mean
            elif self.marl_names[h] == "MFAC":
                a, a_mean = mas_group.act(obs_n[h], test_mode, act_mean_last[h], agent_mask[h])
                act_mean_current[h] = a_mean
                _, values = mas_group.values(obs_n[h], act_mean_current[h])
                values_n.append(values)
            elif self.marl_names[h] == "VDAC":
                _, a, values = mas_group.act(obs_n[h], state=state, test_mode=test_mode)
                values_n.append(values)
            elif self.marl_names[h] in ["MAPPO", "IPPO"]:
                _, a, log_pi = mas_group.act(obs_n[h], test_mode=test_mode, state=state)
                _, values = mas_group.values(obs_n[h], state=state)
                log_pi_n.append(log_pi)
                values_n.append(values)
            elif self.marl_names[h] in ["COMA"]:
                _, a, a_onehot = mas_group.act(obs_n[h], test_mode)
                _, values = mas_group.values(obs_n[h], state=state, actions_n=a, actions_onehot=a_onehot)
                actions_n_onehot.append(a_onehot)
                values_n.append(values)
            else:
                _, a = mas_group.act(obs_n[h], test_mode=test_mode)
            actions_n.append(a)
        return {'actions_n': actions_n, 'log_pi': log_pi_n, 'act_mean': act_mean_current,
                'act_n_onehot': actions_n_onehot, 'values': values_n}

    def store_data(self, obs_n, next_obs_n, actions_dict, state, next_state, agent_mask, rew_n, done_n):
        for h, mas_group in enumerate(self.marl_agents):
            if mas_group.args.agent_name == "random":
                continue
            data_step = {'obs': obs_n[h], 'obs_next': next_obs_n[h], 'actions': actions_dict['actions_n'][h],
                         'state': state, 'state_next': next_state, 'rewards': rew_n[h],
                         'agent_mask': agent_mask[h], 'terminals': done_n[h]}
            if mas_group.on_policy:
                data_step['values'] = actions_dict['values'][h]
                if self.marl_names[h] == "MAPPO":
                    data_step['log_pi_old'] = actions_dict['log_pi'][h]
                elif self.marl_names[h] == "COMA":
                    data_step['actions_onehot'] = actions_dict['act_n_onehot'][h]
                else:
                    pass
                mas_group.memory.store(data_step)
                if mas_group.memory.full:
                    if self.marl_names[h] == "COMA":
                        _, values_next = mas_group.values(next_obs_n[h],
                                                          state=next_state,
                                                          actions_n=actions_dict['actions_n'][h],
                                                          actions_onehot=actions_dict['act_n_onehot'][h])
                    elif self.marl_names[h] == "MFAC":
                        _, values_next = mas_group.values(next_obs_n[h], actions_dict['act_mean'][h])
                    elif self.marl_names[h] == "VDAC":
                        _, _, values_next = mas_group.act(next_obs_n[h])
                    else:
                        _, values_next = mas_group.values(next_obs_n[h], state=next_state)
                    for i_env in range(self.n_envs):
                        if done_n[h][i_env].all():
                            mas_group.memory.finish_path(0.0, i_env)
                        else:
                            mas_group.memory.finish_path(values_next[i_env], i_env)
                continue
            elif self.marl_names[h] in ["MFQ", "MFAC"]:
                data_step['act_mean'] = actions_dict['act_mean'][h]
            else:
                pass
            mas_group.memory.store(data_step)

    def train_episode(self, n_episodes):
        act_mean_last = [np.zeros([self.n_envs, arg.dim_act]) for arg in self.args]
        terminal_handle = np.zeros([self.n_handles, self.n_envs], dtype=np.bool_)
        truncate_handle = np.zeros([self.n_handles, self.n_envs], dtype=np.bool_)
        episode_score = np.zeros([self.n_handles, self.n_envs, 1], dtype=np.float32)
        episode_info, train_info = {}, {}
        for _ in tqdm(range(n_episodes)):
            obs_n = self.envs.buf_obs
            state, agent_mask = self.envs.global_state(), self.envs.agent_mask()
            for step in range(self.episode_length):
                actions_dict = self.get_actions(obs_n, False, act_mean_last, agent_mask, state)
                actions_execute = self.combine_env_actions(actions_dict['actions_n'])
                next_obs_n, rew_n, terminated_n, truncated_n, infos = self.envs.step(actions_execute)
                next_state, agent_mask = self.envs.global_state(), self.envs.agent_mask()

                self.store_data(obs_n, next_obs_n, actions_dict, state, next_state, agent_mask, rew_n, terminated_n)

                # train the model for each step
                if self.train_per_step:
                    if self.current_step % self.training_frequency == 0:
                        for h, mas_group in enumerate(self.marl_agents):
                            if mas_group.args.agent_name == "random":
                                continue
                            train_info = self.marl_agents[h].train(self.current_step)

                obs_n, state, act_mean_last = deepcopy(next_obs_n), deepcopy(next_state), deepcopy(
                    actions_dict['act_mean'])

                for h, mas_group in enumerate(self.marl_agents):
                    episode_score[h] += np.mean(rew_n[h] * agent_mask[h][:, :, np.newaxis], axis=1)
                    terminal_handle[h] = terminated_n[h].all(axis=-1)
                    truncate_handle[h] = truncated_n[h].all(axis=-1)

                for i_env in range(self.n_envs):
                    if terminal_handle.all(axis=0)[i_env] or truncate_handle.all(axis=0)[i_env]:
                        self.current_episode[i_env] += 1
                        for h, mas_group in enumerate(self.marl_agents):
                            if mas_group.args.agent_name == "random":
                                continue
                            if mas_group.on_policy:
                                if mas_group.args.agent == "COMA":
                                    _, value_next_e = mas_group.values(next_obs_n[h],
                                                                       state=next_state,
                                                                       actions_n=actions_dict['actions_n'][h],
                                                                       actions_onehot=actions_dict['act_n_onehot'][h])
                                elif mas_group.args.agent == "MFAC":
                                    _, value_next_e = mas_group.values(next_obs_n[h], act_mean_last[h])
                                elif mas_group.args.agent == "VDAC":
                                    _, _, value_next_e = mas_group.act(next_obs_n[h])
                                else:
                                    _, value_next_e = mas_group.values(next_obs_n[h], state=next_state)
                                mas_group.memory.finish_path(value_next_e[i_env], i_env)
                            obs_n[h][i_env] = infos[i_env]["reset_obs"][h]
                            agent_mask[h][i_env] = infos[i_env]["reset_agent_mask"][h]
                            act_mean_last[h][i_env] = np.zeros([self.args[h].dim_act])
                            episode_score[h, i_env] = np.mean(infos[i_env]["individual_episode_rewards"][h])
                        state[i_env] = infos[i_env]["reset_state"]
                self.current_step += self.n_envs

            if self.n_handles > 1:
                for h in range(self.n_handles):
                    episode_info["Train_Episode_Score/side_{}".format(self.args[h].handle_name)] = episode_score[h].mean()
            else:
                episode_info["Train_Episode_Score"] = episode_score[0].mean()

            # train the model for each episode
            if not self.train_per_step:
                for h, mas_group in enumerate(self.marl_agents):
                    if mas_group.args.agent_name == "random":
                        continue
                    train_info = self.marl_agents[h].train(self.current_step)
            self.log_infos(train_info, self.current_step)
            self.log_infos(episode_info, self.current_step)

    def test_episode(self, env_fn):
        test_envs = env_fn()
        test_info = {}
        num_envs = test_envs.num_envs
        videos, episode_videos = [[] for _ in range(num_envs)], []
        obs_n, infos = test_envs.reset()
        state, agent_mask = test_envs.global_state(), test_envs.agent_mask()
        if self.args_base.render_mode == "rgb_array" and self.render:
            images = test_envs.render(self.args_base.render_mode)
            for idx, img in enumerate(images):
                videos[idx].append(img)
        act_mean_last = [np.zeros([num_envs, arg.dim_act]) for arg in self.args]
        terminal_handle = np.zeros([self.n_handles, num_envs], dtype=np.bool_)
        truncate_handle = np.zeros([self.n_handles, num_envs], dtype=np.bool_)
        episode_score = np.zeros([self.n_handles, num_envs, 1], dtype=np.float32)

        for step in range(self.episode_length):
            actions_dict = self.get_actions(obs_n, True, act_mean_last, agent_mask, state)
            actions_execute = self.combine_env_actions(actions_dict['actions_n'])
            next_obs_n, rew_n, terminated_n, truncated_n, infos = test_envs.step(actions_execute)
            if self.args_base.render_mode == "rgb_array" and self.render:
                images = test_envs.render(self.args_base.render_mode)
                for idx, img in enumerate(images):
                    videos[idx].append(img)

            next_state, agent_mask = test_envs.global_state(), test_envs.agent_mask()

            obs_n, state, act_mean_last = deepcopy(next_obs_n), deepcopy(next_state), deepcopy(actions_dict['act_mean'])

            for h, mas_group in enumerate(self.marl_agents):
                episode_score[h] += np.mean(rew_n[h] * agent_mask[h][:, :, np.newaxis], axis=1)
                terminal_handle[h] = terminated_n[h].all(axis=-1)
                truncate_handle[h] = truncated_n[h].all(axis=-1)

            for i in range(num_envs):
                if terminal_handle.all(axis=0)[i] or truncate_handle.all(axis=0)[i]:
                    for h, mas_group in enumerate(self.marl_agents):
                        obs_n[h][i] = infos[i]["reset_obs"][h]
                        agent_mask[h][i] = infos[i]["reset_agent_mask"][h]
                        act_mean_last[h][i] = np.zeros([self.args[h].dim_act])
                    state = infos[i]["reset_state"]
        scores = episode_score.mean(axis=1).reshape([self.n_handles])
        if self.args_base.test_mode:
            print("Mean score: ", scores)

        if self.args_base.render_mode == "rgb_array" and self.render:
            # time, height, width, channel -> time, channel, height, width
            videos_info = {"Videos_Test": np.array(videos, dtype=np.uint8).transpose((0, 1, 4, 2, 3))}
            self.log_videos(info=videos_info, fps=self.fps, x_index=self.current_step)

        if self.n_handles > 1:
            for h in range(self.n_handles):
                test_info["Test-Episode-Rewards/Side_{}".format(self.args[h].handle_name)] = scores[h]
        else:
            test_info["Test-Episode-Rewards"] = scores[0]
        self.log_infos(test_info, self.current_step)

        test_envs.close()

        return episode_score

    def run(self):
        if self.args_base.test_mode:
            def env_fn():
                args_test = deepcopy(self.args_base)
                args_test.parallels = args_test.test_episode
                return make_envs(args_test)

            self.render = True
            for h, mas_group in enumerate(self.marl_agents):
                mas_group.load_model(mas_group.model_dir_load)
            self.test_episode(env_fn)
            print("Finish testing.")
        else:
            n_train_episodes = self.args_base.running_steps // self.episode_length // self.n_envs
            self.train_episode(n_train_episodes)
            print("Finish training.")
            for h, mas_group in enumerate(self.marl_agents):
                mas_group.save_model("final_train_model.pth")

        self.envs.close()
        if self.use_wandb:
            wandb.finish()
        else:
            self.writer.close()

    def benchmark(self):
        def env_fn():
            args_test = deepcopy(self.args_base)
            args_test.parallels = args_test.test_episode
            return make_envs(args_test)

        n_train_episodes = self.args_base.running_steps // self.episode_length // self.n_envs
        n_eval_interval = self.args_base.eval_interval // self.episode_length // self.n_envs
        num_epoch = int(n_train_episodes / n_eval_interval)

        test_scores = self.test_episode(env_fn)
        best_scores = [{
            "mean": np.mean(test_scores, axis=1).reshape([self.n_handles]),
            "std": np.std(test_scores, axis=1).reshape([self.n_handles]),
            "step": self.current_step
        } for _ in range(self.n_handles)]
        for h in range(self.n_handles):
            self.marl_agents[h].save_model("best_model.pth")

        for i_epoch in range(num_epoch):
            print("Epoch: %d/%d:" % (i_epoch, num_epoch))
            self.train_episode(n_episodes=n_eval_interval)
            test_scores = self.test_episode(env_fn)

            mean_test_scores = np.mean(test_scores, axis=1)
            for h in range(self.n_handles):
                if mean_test_scores[h] > best_scores[h]["mean"][h]:
                    best_scores[h] = {
                        "mean": mean_test_scores.reshape([self.n_handles]),
                        "std": np.std(test_scores, axis=1).reshape([self.n_handles]),
                        "step": self.current_step
                    }
                    # save best model
                    self.marl_agents[h].save_model("best_model.pth")

        # end benchmarking
        print("Finish benchmarking.")
        for h in range(self.n_handles):
            print("Best Score for {}: ".format(self.envs.envs[0].side_names[h]))
            print("Mean: ", best_scores[h]["mean"], "Std: ", best_scores[h]["std"])

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

    if args[0].benchmark:
        runner.benchmark()
    else:
        runner.run()

    runner.finish()
