from .runner_basic import *
from xuance.tensorflow.agents import REGISTRY as REGISTRY_Agent
from gym.spaces import Box
from tqdm import tqdm
import numpy as np
from copy import deepcopy
import time


class Runner(Runner_Base_MARL):
    def __init__(self, args):
        self.args = args if type(args) == list else [args]
        super(Runner, self).__init__(self.args[0])

        # environment details, representations, policies, optimizers, and agents.
        for h, arg in enumerate(self.args):
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

    def run_episode(self, episode, test_mode=False):
        obs_n = self.envs.reset()
        state, agent_mask = self.envs.global_state(), self.envs.agent_mask()

        scores = np.zeros([self.n_handles, self.n_envs, 1], dtype=np.float32)
        done_envs = np.zeros([self.n_envs, 1], dtype=np.bool)
        act_mean_last = [np.zeros([self.n_envs, arg.dim_act]) for arg in self.args]

        while True:
            actions_dict = self.get_actions(obs_n, episode, test_mode, act_mean_last, agent_mask, state)
            actions_execute = self.combine_env_actions(actions_dict['actions_n'])
            next_obs_n, rew_n, done_n, dones, info = self.envs.step(actions_execute)
            next_state, agent_mask = self.envs.global_state(), self.envs.agent_mask()

            if self.render or self.test_mode:
                time.sleep(self.render_delay)
            if test_mode:
                for h in range(self.n_handles):
                    scores[h] += (1-done_envs) * np.mean(rew_n[h] * agent_mask[h][:, :, np.newaxis], axis=1)
            else:
                self.store_data(obs_n, next_obs_n, actions_dict, state, next_state, agent_mask, rew_n, done_n, self.envs)
                for h, mas_group in enumerate(self.marl_agents):
                    if self.args[h].train_at_step: self.marl_agents[h].train(episode)

            obs_n, state, act_mean_last = deepcopy(next_obs_n), deepcopy(next_state), deepcopy(actions_dict['act_mean'])

            for e, d in enumerate(dones):
                if d:  # if done, then reset this environment
                    done_envs[e], obs_reset = d, self.envs.reset_one_env(e)
                    state[e] = self.envs.global_state_one_env(e)
                    if not test_mode:
                        for h, mas_group in enumerate(self.marl_agents):
                            obs_n[h][e], act_mean_last[h][e] = obs_reset[h], np.zeros([self.args[h].dim_act])
                            if (self.marl_names[h] in ["MAPPO", "CID_Simple", "VDAC"]) and (not self.args[h].consider_terminal_states):
                                value_next_e = mas_group.value(next_obs_n[h], next_state)[e]
                            else:
                                value_next_e = np.zeros([mas_group.n_agents, 1])
                            mas_group.memory.finish_ac_path(value_next_e, e)

            if all(done_envs): break

        # train the model
        if not test_mode:
            for h, mas_group in enumerate(self.marl_agents):
                if not self.args[h].train_at_step:
                    mas_group.train(episode)

        return scores

    def run(self):
        for i_episode in tqdm(range(self.n_episodes)):
            self.run_episode(i_episode, test_mode=self.test_mode)

            # test and save models
            if (i_episode % self.test_period == 0) and (not self.test_mode):
                reward = np.zeros([self.n_handles, self.n_envs, 1])
                for i_test in range(self.n_tests):
                    r_episode = self.run_episode(i_episode, test_mode=True)
                    reward += r_episode
                reward = reward / self.n_tests
                for h, mas_group in enumerate(self.marl_agents):
                    for i in range(self.n_envs):
                        self.marl_agents[h].writer.add_scalars("reward_mean", {"env-%d" % i: reward[h, i]}, i_episode)
        for h, mas_group in enumerate(self.marl_agents):
            mas_group.save_model()

        self.envs.close()

    def store_data(self, obs_n, next_obs_n, actions_dict, state, next_state, agent_mask, rew_n, done_n, env):
        for h, mas_group in enumerate(self.marl_agents):
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
