import os
from argparse import Namespace
from copy import deepcopy
import numpy as np
from tqdm import tqdm
from operator import itemgetter
from xuance.torch.agents import REGISTRY_Agents
from xuance.environment import make_envs
from xuance.environment.vector_envs import combine_actions
from xuance.torch.utils.operations import set_seed


class RunnerCompetition(object):
    def __init__(self, configs):
        self.configs = configs
        # set random seeds
        set_seed(self.configs[0].seed)

        # build environments
        self.envs = make_envs(self.configs[0])
        self.n_envs = self.envs.num_envs
        self.current_step = 0
        self.envs.reset()
        self.groups_info = self.envs.groups_info
        self.groups = self.groups_info['agent_groups']
        self.num_groups = self.groups_info['num_groups']
        self.obs_space_groups = self.groups_info['observation_space_groups']
        self.act_space_groups = self.groups_info['action_space_groups']
        assert len(configs) == self.num_groups, "Number of groups must be equal to the number of methods."
        self.agents = []
        for group in range(self.num_groups):
            _env = Namespace(num_agents=len(self.groups[group]),
                             num_envs=self.envs.num_envs,
                             agents=self.groups[group],
                             state_space=self.envs.state_space,
                             observation_space=self.obs_space_groups[group],
                             action_space=self.act_space_groups[group],
                             max_episode_steps=self.envs.max_episode_steps)
            self.agents.append(REGISTRY_Agents[self.configs[group].agent](self.configs[group], _env))

        self.distributed_training = self.agents[0].distributed_training
        self.use_actions_mask = self.agents[0].use_actions_mask
        self.use_global_state = self.agents[0].use_global_state
        self.use_rnn = self.agents[0].use_rnn
        self.use_wandb = self.agents[0].use_wandb

        self.rank = 0
        if self.distributed_training:
            self.rank = int(os.environ['RANK'])

    def rprint(self, info: str):
        if self.rank == 0:
            print(info)

    def run(self):
        if self.configs[0].test_mode:
            def env_fn():
                config_test = deepcopy(self.configs[0])
                config_test.parallels = 1
                config_test.render = True
                return make_envs(config_test)

            for agent in self.agents:
                agent.render = True
                agent.load_model(agent.model_dir_load)

            scores = self.test(env_fn, self.configs[0].test_episode)

            print(f"Mean Score: {scores}, Std: {scores}")
            print("Finish testing.")
        else:
            n_train_steps = self.configs[0].running_steps // self.n_envs

            self.train(n_train_steps)

            print("Finish training.")
            for agent in self.agents:
                agent.save_model("final_train_model.pth")

        for agent in self.agents:
            agent.finish()
        self.envs.close()

    def benchmark(self):
        def env_fn():
            config_test = deepcopy(self.configs[0])
            config_test.parallels = 1  # config_test.test_episode
            return make_envs(config_test)

        train_steps = self.configs[0].running_steps // self.n_envs
        eval_interval = self.configs[0].eval_interval // self.n_envs
        test_episode = self.configs[0].test_episode
        num_epoch = int(train_steps / eval_interval)

        test_scores = self.test(env_fn, test_episode) if self.rank == 0 else 0.0

        best_scores_info = [{"mean": np.mean(test_scores[i]),
                             "std": np.std(test_scores[i]),
                             "step": self.agents[i].current_step} for i in range(self.num_groups)]

        for i_epoch in range(num_epoch):
            print("Epoch: %d/%d:" % (i_epoch, num_epoch))

            self.train(eval_interval)

            if self.rank == 0:

                test_scores = self.test(env_fn, test_episode)

                for i in range(self.num_groups):
                    if np.mean(test_scores[i]) > best_scores_info[i]["mean"]:
                        best_scores_info[i] = {"mean": np.mean(test_scores[i]),
                                               "std": np.std(test_scores[i]),
                                               "step": self.agents[i].current_step}
                        # save best model
                        self.agents[i].save_model(model_name="best_model.pth")

        # end benchmarking
        best_scores = [score["mean"] for score in best_scores_info]
        std_list = [score["std"] for score in best_scores_info]
        print(f"The training for {self.configs[0].env_name}/{self.configs[0].env_id} is finished.")
        print("Algorithms: ", [config.agent for config in self.configs])
        print("Best Model Score: ,", best_scores, "Std: ", std_list)
        for i in range(self.num_groups):
            self.agents[i].finish()
        self.envs.close()

    def train(self, n_steps):
        """
        Train the model for numerous steps.
        Args:
            n_steps (int): Number of steps to train the model:
        """
        if self.use_rnn:
            with tqdm(total=n_steps) as process_bar:
                step_start, step_last = deepcopy(self.current_step), deepcopy(self.current_step)
                n_steps_all = n_steps * self.n_envs
                while step_last - step_start < n_steps_all:
                    self.run_episodes(None, n_episodes=self.n_envs, test_mode=False)
                    for agent in self.agents:
                        if self.current_step >= agent.start_training:
                            train_info = agent.train_epochs(n_epochs=agent.n_epochs)
                            agent.log_infos(train_info, self.current_step)
                    process_bar.update((self.current_step - step_last) // self.n_envs)
                    step_last = deepcopy(self.current_step)
                process_bar.update(n_steps - process_bar.last_print_n)
            return

        obs_dict = self.envs.buf_obs
        avail_actions = self.envs.buf_avail_actions if self.use_actions_mask else None
        state = self.envs.buf_state.copy() if self.use_global_state else None
        for _ in tqdm(range(n_steps)):
            step_info = {}
            policy_out_list = [agent.action(obs_dict=obs_dict,
                                            state=state,
                                            avail_actions_dict=avail_actions,
                                            test_mode=False) for agent in self.agents]
            actions_execute = combine_actions(policy_out_list, self.n_envs)
            next_obs_dict, rewards_dict, terminated_dict, truncated, info = self.envs.step(actions_execute)
            next_state = self.envs.buf_state.copy() if self.use_global_state else None
            next_avail_actions = self.envs.buf_avail_actions.copy() if self.use_actions_mask else None
            for agent in self.agents:
                agent.store_experience(obs_dict, avail_actions, actions_execute, next_obs_dict, next_avail_actions,
                                       rewards_dict, terminated_dict, info,
                                       **{'state': state, 'next_state': next_state})
                if self.current_step >= agent.start_training and self.current_step % agent.training_frequency == 0:
                    train_info = agent.train_epochs(n_epochs=agent.n_epochs)
                    agent.log_infos(train_info, self.current_step)

            obs_dict = next_obs_dict

            if self.use_global_state:
                state = deepcopy(next_state)
            if self.use_actions_mask:
                avail_actions = deepcopy(next_avail_actions)

            for i in range(self.n_envs):
                if all(terminated_dict[i].values()) or truncated[i]:
                    obs_dict[i] = info[i]["reset_obs"]
                    self.envs.buf_obs[i] = info[i]["reset_obs"]
                    if self.use_global_state:
                        state = info[i]["reset_state"]
                        self.envs.buf_state[i] = info[i]["reset_state"]
                    if self.use_actions_mask:
                        avail_actions[i] = info[i]["reset_avail_actions"]
                        self.envs.buf_avail_actions[i] = info[i]["reset_avail_actions"]
                    for agent in self.agents:
                        episode_score = np.mean(itemgetter(*agent.agent_keys)(info[i]["episode_score"]))
                        if self.use_wandb:
                            step_info[f"Train-Results/Episode-Steps/rank_{self.rank}/env-%d" % i] = info[i][
                                "episode_step"]
                            step_info[f"Train-Results/Episode-Rewards/rank_{self.rank}/env-%d" % i] = episode_score
                        else:
                            step_info[f"Train-Results/Episode-Steps/rank_{self.rank}"] = {
                                "env-%d" % i: info[i]["episode_step"]}
                            step_info[f"Train-Results/Episode-Rewards/rank_{self.rank}"] = {"env-%d" % i: episode_score}
                        agent.log_infos(step_info, self.current_step)

            self.current_step += self.n_envs
            for agent in self.agents:
                agent.current_step += self.n_envs
                if not agent.on_policy:
                    agent._update_explore_factor()

    def run_episodes(self, env_fn=None, n_episodes: int = 1, test_mode: bool = False):
        """
        Run some episodes when use RNN.

        Parameters:
            env_fn: The function that can make some testing environments.
            n_episodes (int): Number of episodes.
            test_mode (bool): Whether to test the model.

        Returns:
            Scores: The episode scores.
        """
        envs = self.envs if env_fn is None else env_fn()
        num_envs = envs.num_envs
        videos = [[] for _ in range(num_envs)]
        episode_videos = [[] for _ in range(self.num_groups)]
        episode_count = 0
        scores = [[0.0 for _ in range(num_envs)] for _ in range(self.num_groups)]
        best_score = [-np.inf for _ in range(self.num_groups)]
        obs_dict, info = envs.reset()
        state = envs.buf_state.copy() if self.use_global_state else None
        avail_actions = envs.buf_avail_actions if self.use_actions_mask else None
        if test_mode:
            for config in self.configs:
                if config.render_mode == "rgb_array" and config.render:
                    images = envs.render(config.render_mode)
                    for idx, img in enumerate(images):
                        videos[idx].append(img)
        else:
            if self.use_rnn:
                for agent in self.agents:
                    agent.memory.clear_episodes()
        rnn_hidden = [agent.init_rnn_hidden(num_envs) for agent in self.agents]

        while episode_count < n_episodes:
            step_info = {}
            policy_out_list = [agent.action(obs_dict=obs_dict,
                                            state=state,
                                            avail_actions_dict=avail_actions,
                                            rnn_hidden=rnn_hidden[i_agt],
                                            test_mode=test_mode) for i_agt, agent in enumerate(self.agents)]
            actions_execute = combine_actions(policy_out_list, num_envs)
            rnn_hidden = [policy_out['hidden_state'] for policy_out in policy_out_list]
            next_obs_dict, rewards_dict, terminated_dict, truncated, info = envs.step(actions_execute)
            next_state = envs.buf_state.copy() if self.use_global_state else None
            next_avail_actions = envs.buf_avail_actions if self.use_actions_mask else None
            if test_mode:
                for config in self.configs:
                    if config.render_mode == "rgb_array" and config.render:
                        images = envs.render(config.render_mode)
                        for idx, img in enumerate(images):
                            videos[idx].append(img)
            else:
                for agent in self.agents:
                    agent.store_experience(obs_dict, avail_actions, actions_execute, next_obs_dict, next_avail_actions,
                                           rewards_dict, terminated_dict, info,
                                           **{'state': state, 'next_state': next_state})
            obs_dict = deepcopy(next_obs_dict)
            if self.use_global_state:
                state = deepcopy(next_state)
            if self.use_actions_mask:
                avail_actions = deepcopy(next_avail_actions)

            for i in range(num_envs):
                if all(terminated_dict[i].values()) or truncated[i]:
                    episode_count += 1
                    obs_dict[i] = info[i]["reset_obs"]
                    envs.buf_obs[i] = info[i]["reset_obs"]
                    if self.use_global_state:
                        state = info[i]["reset_state"]
                        self.envs.buf_state[i] = info[i]["reset_state"]
                    if self.use_actions_mask:
                        avail_actions[i] = info[i]["reset_avail_actions"]
                        envs.buf_avail_actions[i] = info[i]["reset_avail_actions"]
                    if self.use_rnn:
                        rnn_hidden = [agent.init_hidden_item(i_env=i, rnn_hidden=rnn_hidden) for agent in self.agents]
                        if not test_mode:
                            terminal_data = {'obs': next_obs_dict[i],
                                             'episode_step': info[i]['episode_step']}
                            if self.use_global_state:
                                terminal_data['state'] = next_state[i]
                            if self.use_actions_mask:
                                terminal_data['avail_actions'] = next_avail_actions[i]
                            for agent in self.agents:
                                agent.memory.finish_path(i, **terminal_data)
                    if not test_mode:
                        self.current_step += info[i]["episode_step"]
                        for agent in self.agents:
                            agent.current_step += info[i]["episode_step"]
                    for i_group in range(self.num_groups):
                        episode_score = float(np.mean(itemgetter(*self.groups[i_group])(info[i]["episode_score"])))
                        scores[i_group].append(episode_score)
                        if test_mode:
                            if best_score[i_group] < episode_score:
                                best_score[i_group] = episode_score
                                episode_videos[i_group] = videos[i].copy()
                            if self.configs[i_group].test_mode:
                                print("Episode: %d, Score: %.2f" % (episode_count, episode_score))
                        else:
                            for agent in self.agents:
                                episode_score = np.mean(itemgetter(*agent.agent_keys)(info[i]["episode_score"]))
                                if self.use_wandb:
                                    step_info["Train-Results/Episode-Steps/env-%d" % i] = info[i]["episode_step"]
                                    step_info["Train-Results/Episode-Rewards/env-%d" % i] = episode_score
                                else:
                                    step_info["Train-Results/Episode-Steps"] = {"env-%d" % i: info[i]["episode_step"]}
                                    step_info["Train-Results/Episode-Rewards"] = {"env-%d" % i: episode_score}
                                agent.log_infos(step_info, self.current_step)
                                if not agent.on_policy:
                                    agent._update_explore_factor()

        if test_mode:
            for i_group in range(self.num_groups):
                config = self.configs[i_group]
                if config.render_mode == "rgb_array" and config.render:
                    # time, height, width, channel -> time, channel, height, width
                    videos_info = {"Videos_Test": np.array([episode_videos[i_group]],
                                                           dtype=np.uint8).transpose((0, 1, 4, 2, 3))}
                    self.agents[i_group].log_videos(info=videos_info, fps=config.fps, x_index=self.current_step)

            if self.configs[0].test_mode:
                print("Best Score: ", best_score)

            for i_group in range(self.num_groups):
                test_info = {
                    "Test-Results/Episode-Rewards": np.mean(scores[i_group]),
                    "Test-Results/Episode-Rewards-Std": np.std(scores[i_group]),
                }

                self.agents[i_group].log_infos(test_info, self.current_step)
            if env_fn is not None:
                envs.close()
        return scores

    def test(self, env_fn, n_episodes):
        """
        Test the model for some episodes.

        Parameters:
            env_fn: The function that can make some testing environments.
            n_episodes (int): Number of episodes to test.

        Returns:
            scores (List(float)): A list of cumulative rewards for each episode.
        """
        scores = self.run_episodes(env_fn=env_fn, n_episodes=n_episodes, test_mode=True)
        return scores
