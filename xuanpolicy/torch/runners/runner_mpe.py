import time

from .runner_basic import *
from xuanpolicy.torch.agents import REGISTRY as REGISTRY_Agent
from gymnasium.spaces.box import Box
from tqdm import tqdm
import numpy as np
from copy import deepcopy


class MPE_Runner(Runner_Base_MARL):
    def __init__(self, args):
        self.args = args if type(args) == list else [args]
        for arg in self.args:
            if arg.agent_name == "random":
                continue
            else:
                super(MPE_Runner, self).__init__(arg)
                self.training_steps = arg.training_steps
                self.training_frequency = arg.training_frequency
                self.off_policy = arg.off_policy
                self.on_policy = not self.off_policy
                break
        self.episode_length = self.envs.max_episode_length

        # environment details, representations, policies, optimizers, and agents.
        for h, arg in enumerate(self.args):
            if self.n_handles > 1 and arg.agent != "RANDOM":
                arg.modeldir += "side_{}/".format(h)
            arg.handle, arg.n_agents = h, self.envs.n_agents[h]
            arg.agent_keys, arg.agent_ids = self.agent_keys[h], self.agent_ids[h]
            arg.state_space = self.envs.state_space
            arg.observation_space = self.envs.observation_space
            if isinstance(self.envs.action_space[self.agent_keys[h][0]], Box):
                arg.dim_act = self.envs.action_space[self.agent_keys[h][0]].shape[0]
                arg.act_shape = (arg.n_agents,) + (arg.dim_act,)
            else:
                arg.dim_act = self.envs.action_space[self.agent_keys[h][0]].n
                arg.act_shape = (arg.n_agents,)
            arg.action_space = self.envs.action_space
            arg.dim_obs = self.envs.observation_space[self.agent_keys[h][0]].shape
            arg.obs_shape = (arg.n_agents,) + arg.dim_obs
            arg.rew_shape, arg.done_shape, arg.act_prob_shape = (arg.n_agents, 1), (arg.n_agents,), (arg.dim_act,)
            self.marl_agents.append(REGISTRY_Agent[arg.agent](arg, self.envs, arg.device))
            self.marl_names.append(arg.agent)
            if arg.test_mode:
                self.marl_agents[h].load_model(arg.modeldir)

        self.print_infos(self.args)

    def train_episode(self, n_episodes):
        obs_n = self.envs.buf_obs
        state, agent_mask = self.envs.global_state(), self.envs.agent_mask()

        act_mean_last = [np.zeros([self.n_envs, arg.dim_act]) for arg in self.args]
        terminal_handle = np.zeros([self.n_handles, self.n_envs], dtype=np.bool)
        truncate_handle = np.zeros([self.n_handles, self.n_envs], dtype=np.bool)
        episode_score = np.zeros([self.n_handles, self.n_envs, 1], dtype=np.float32)
        episode_info = {}
        for i_episode in tqdm(range(n_episodes)):
            for step in range(self.episode_length):
                actions_dict = self.get_actions(obs_n, False, act_mean_last, agent_mask, state)
                actions_execute = self.combine_env_actions(actions_dict['actions_n'])
                next_obs_n, rew_n, terminated_n, truncated_n, infos = self.envs.step(actions_execute)
                next_state, agent_mask = self.envs.global_state(), self.envs.agent_mask()

                self.store_data(obs_n, next_obs_n, actions_dict, state, next_state, agent_mask, rew_n, terminated_n, self.envs)

                # train the model for off-policy
                if self.off_policy and (self.current_step % self.training_frequency == 0):
                    for h, mas_group in enumerate(self.marl_agents):
                        if mas_group.args.agent_name == "random":
                            continue
                        train_info = self.marl_agents[h].train(self.current_episode)
                        mas_group.log_infos(train_info, self.current_step)
                        mas_group.log_infos(episode_info, self.current_step)

                obs_n, state, act_mean_last = deepcopy(next_obs_n), deepcopy(next_state), deepcopy(actions_dict['act_mean'])

                for h, mas_group in enumerate(self.marl_agents):
                    episode_score[h] += np.mean(rew_n[h] * agent_mask[h][:, :, np.newaxis], axis=1)
                    terminal_handle[h] = terminated_n[h].all(axis=-1)
                    truncate_handle[h] = truncated_n[h].all(axis=-1)

                for i in range(self.n_envs):
                    if terminal_handle.all(axis=0)[i] or truncate_handle.all(axis=0)[i]:
                        state[i] = self.envs.global_state_one_env(i)
                        self.current_episode[i] += 1
                        for h, mas_group in enumerate(self.marl_agents):
                            if mas_group.args.agent_name == "random":
                                continue
                            obs_n[h][i] = infos[i]["reset_obs"][h]
                            act_mean_last[h][i] = np.zeros([self.args[h].dim_act])
                            if (self.marl_names[h] in ["MAPPO", "CID_Simple", "VDAC"]) and (not self.args[h].consider_terminal_states):
                                value_next_e = mas_group.value(next_obs_n[h], next_state)[i]
                            else:
                                value_next_e = np.zeros([mas_group.n_agents, 1])
                            mas_group.memory.finish_ac_path(value_next_e, i)
                            episode_score[h, i] = np.mean(infos[i]["individual_episode_rewards"][h])
                self.current_step += self.n_envs

            for h in range(self.n_handles):
                episode_info["Train_Episode_Score/side_%d" % h] = episode_score.mean(axis=1)
                episode_info["Train_Episode_Score_std/side_%d" % h] = episode_score.std(axis=1)

            # train the model for on-policy
            if self.on_policy and (self.current_step % self.training_frequency == 0):
                for h, mas_group in enumerate(self.marl_agents):
                    if mas_group.args.agent_name == "random":
                        continue
                    train_info = self.marl_agents[h].train(self.current_episode)
                    mas_group.log_infos(train_info, self.current_step)
                    mas_group.log_infos(episode_info, self.current_step)

            self.current_episode += self.n_envs

    def test_episode(self, env_fn, n_episodes):
        test_envs = env_fn()
        num_envs = test_envs.num_envs
        videos, episode_videos = [[] for _ in range(num_envs)], []
        obs_n, infos = test_envs.reset()
        state, agent_mask = test_envs.global_state(), test_envs.agent_mask()
        if self.args_base.render_mode == "rgb_array" and self.render:
            images = test_envs.render(self.args_base.render_mode)
            for idx, img in enumerate(images):
                videos[idx].append(img)

        act_mean_last = [np.zeros([num_envs, arg.dim_act]) for arg in self.args]
        terminal_handle = np.zeros([self.n_handles, num_envs], dtype=np.bool)
        truncate_handle = np.zeros([self.n_handles, num_envs], dtype=np.bool)
        episode_score = np.zeros([self.n_handles, num_envs, 1], dtype=np.float32)
        for _ in range(n_episodes):
            for step in range(self.episode_length):
                actions_dict = self.get_actions(obs_n, True, act_mean_last, agent_mask, state)
                actions_execute = self.combine_env_actions(actions_dict['actions_n'])
                next_obs_n, rew_n, terminated_n, truncated_n, infos = test_envs.step(actions_execute)
                next_state, agent_mask = test_envs.global_state(), test_envs.agent_mask()

                obs_n, state, act_mean_last = deepcopy(next_obs_n), deepcopy(next_state), deepcopy(actions_dict['act_mean'])

                for h, mas_group in enumerate(self.marl_agents):
                    episode_score[h] += np.mean(rew_n[h] * agent_mask[h][:, :, np.newaxis], axis=1)
                    terminal_handle[h] = terminated_n[h].all(axis=-1)
                    truncate_handle[h] = truncated_n[h].all(axis=-1)

                for i in range(num_envs):
                    if terminal_handle.all(axis=0)[i] or truncate_handle.all(axis=0)[i]:
                        state[i] = test_envs.global_state_one_env(i)
                        for h, mas_group in enumerate(self.marl_agents):
                            obs_n[h][i] = infos[i]["reset_obs"][h]
                            act_mean_last[h][i] = np.zeros([self.args[h].dim_act])
                            episode_score[h][i] = 0.0

    def run(self):
        if self.args_base.test_mode:
            def env_fn():
                args_test = deepcopy(self.args_base)
                args_test.parallels = 1
                return make_envs(args_test)
            self.render = True
            for h, mas_group in enumerate(self.marl_agents):
                mas_group.load_model(mas_group.modeldir)
            self.test_episode(env_fn, self.args_base.test_episodes)
            print("Finish testing.")
        else:
            n_train_episodes = self.args_base.training_steps // self.episode_length // self.n_envs
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
            args_test.parallels = 1
            return make_envs(args_test)
        train_episodes = self.args_base.training_steps // self.episode_length // self.n_envs
        eval_interval = self.args_base.eval_interval // self.episode_length // self.n_envs
        test_episode = self.args_base.test_episode
        num_epoch = int(train_episodes / eval_interval)
        for i_epoch in range(num_epoch):
            print("Epoch: %d/%d:" % (i_epoch, num_epoch))
            self.train_episode(n_episodes=eval_interval)
            self.test_episode(env_fn, test_episode)

        self.envs.close()

    def store_data(self, obs_n, next_obs_n, actions_dict, state, next_state, agent_mask, rew_n, done_n, env):
        for h, mas_group in enumerate(self.marl_agents):
            if mas_group.args.agent_name == "random":
                continue
            data_step = {'obs': obs_n[h], 'obs_next': next_obs_n[h], 'actions': actions_dict['actions_n'][h],
                         'state': state, 'state_next': next_state, 'rewards': rew_n[h],
                         'agent_mask': agent_mask[h], 'terminals': done_n[h]}
            if self.marl_names[h] in ["CID_Simple", "CID_Rainbow"]:
                rew_n_assign = mas_group.reward_shaping(state, obs_n[h], actions_dict['actions_n'][h], rew_n[h], env)
                data_step.update({'rewards_assign': rew_n_assign})
            elif self.marl_names[h] in ["MAPPO_KL", "MAPPO_Clip", "VDAC"]:
                if self.marl_names[h] == "MAPPO_KL":
                    data_step.update({'values': actions_dict['values'][h], 'pi_dist_old': actions_dict['log_pi'][h]})
                else:
                    data_step.update({'values': actions_dict['values'][h], 'log_pi_old': actions_dict['log_pi'][h]})
            elif self.marl_names[h] in ["COMA"]:
                data_step.update({'actions_onehot': actions_dict['act_n_onehot'][h]})
            elif self.marl_names[h] in ["MFQ", "MFAC"]:
                data_step.update({'act_mean': actions_dict['act_mean'][h]})
            else:
                pass
            mas_group.memory.store(data_step)
