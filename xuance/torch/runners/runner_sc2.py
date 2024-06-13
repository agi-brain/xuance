from .runner_basic import Runner_Base
from xuance.torch.agents import REGISTRY_Agents
import time
import numpy as np
from copy import deepcopy


class SC2_Runner(Runner_Base):
    def __init__(self, config):
        super(SC2_Runner, self).__init__(config)
        config.n_agents = self.envs.num_agents
        self.agents = REGISTRY_Agents[config.agent](config, self.envs)
        self.config = config
        self.running_steps = config.running_steps
        self.num_agents, self.num_enemies = self.get_agent_num()

    def get_agent_num(self):
        return self.envs.num_agents, self.envs.num_enemies

    def get_actions(self, obs_n, avail_actions, *rnn_hidden, state=None, test_mode=False):
        log_pi_n, values_n, actions_n_onehot = None, None, None
        rnn_hidden_policy, rnn_hidden_critic = rnn_hidden[0], rnn_hidden[1]
        if self.on_policy:
            if self.args.agent == "COMA":
                rnn_hidden_next, actions_n, actions_n_onehot = self.agents.act(obs_n, *rnn_hidden_policy,
                                                                               avail_actions=avail_actions,
                                                                               test_mode=test_mode)
            elif self.args.agent == "VDAC":
                rnn_hidden_next, actions_n, values_n = self.agents.act(obs_n, *rnn_hidden_policy,
                                                                       avail_actions=avail_actions,
                                                                       state=state,
                                                                       test_mode=test_mode)
            else:
                rnn_hidden_next, actions_n, log_pi_n = self.agents.act(obs_n, *rnn_hidden_policy,
                                                                       avail_actions=avail_actions,
                                                                       test_mode=test_mode)
            if test_mode:
                rnn_hidden_critic_next, values_n = None, 0
            else:
                if self.args.agent == "VDAC":
                    rnn_hidden_critic_next = [None, None]
                else:
                    kwargs = {"state": state}
                    if self.args.agent == "COMA":
                        kwargs.update({"actions_n": actions_n, "actions_onehot": actions_n_onehot})
                    rnn_hidden_critic_next, values_n = self.agents.values(obs_n, *rnn_hidden_critic, **kwargs)
        else:
            rnn_hidden_next, actions_n = self.agents.act(obs_n, *rnn_hidden_policy,
                                                         avail_actions=avail_actions, test_mode=test_mode)
            rnn_hidden_critic_next = None
        return {'actions_n': actions_n, 'log_pi': log_pi_n,
                'rnn_hidden': rnn_hidden_next, 'rnn_hidden_critic': rnn_hidden_critic_next,
                'act_n_onehot': actions_n_onehot, 'values': values_n}

    def get_battles_info(self):
        battles_game, battles_won = self.envs.battles_game.sum(), self.envs.battles_won.sum()
        dead_allies, dead_enemies = self.envs.dead_allies_count.sum(), self.envs.dead_enemies_count.sum()
        return battles_game, battles_won, dead_allies, dead_enemies

    def get_battles_result(self, last_battles_info):
        battles_game, battles_won, dead_allies, dead_enemies = list(last_battles_info)
        incre_battles_game = float(self.envs.battles_game.sum() - battles_game)
        incre_battles_won = float(self.envs.battles_won.sum() - battles_won)
        win_rate = incre_battles_won / incre_battles_game if incre_battles_game > 0 else 0.0
        allies_count, enemies_count = incre_battles_game * self.num_agents, incre_battles_game * self.num_enemies
        incre_allies = float(self.envs.dead_allies_count.sum() - dead_allies)
        incre_enemies = float(self.envs.dead_enemies_count.sum() - dead_enemies)
        allies_dead_ratio = incre_allies / allies_count if allies_count > 0 else 0.0
        enemies_dead_ratio = incre_enemies / enemies_count if enemies_count > 0 else 0.0
        return win_rate, allies_dead_ratio, enemies_dead_ratio

    def run_episodes(self, test_mode=False):
        step_info, train_info = {}, {}
        videos, best_videos = [[] for _ in range(self.n_envs)], []
        episode_score, best_score = [], -np.inf
        # reset the envs and settings
        obs_n, state, info = self.envs.reset()
        envs_done = self.envs.buf_done
        self.env_step = 0
        filled = np.zeros([self.n_envs, self.episode_length, 1], np.int32)
        rnn_hidden, rnn_hidden_critic = self.init_rnn_hidden()

        if test_mode and self.render:
            images = self.envs.render(self.args.render_mode)
            if self.args.render_mode == "rgb_array":
                for idx, img in enumerate(images):
                    videos[idx].append(img)

        while not envs_done.all():  # start episodes
            available_actions = self.envs.get_avail_actions()
            actions_dict = self.get_actions(obs_n, available_actions, rnn_hidden, rnn_hidden_critic,
                                            state=state, test_mode=test_mode)
            next_obs_n, next_state, rewards, terminated, truncated, info = self.envs.step(actions_dict['actions_n'])
            envs_done = self.envs.buf_done
            rnn_hidden, rnn_hidden_critic = actions_dict['rnn_hidden'], actions_dict['rnn_hidden_critic']

            if test_mode:
                if self.render:
                    images = self.envs.render(self.args.render_mode)
                    if self.args.render_mode == "rgb_array":
                        for idx, img in enumerate(images):
                            videos[idx].append(img)
                for i_env in range(self.n_envs):
                    if terminated[i_env] or truncated[i_env]:  # one env is terminal
                        episode_score.append(info[i_env]["episode_score"])
                        if best_score < episode_score[-1]:
                            best_score = episode_score[-1]
                            best_videos = videos[i_env].copy()
            else:
                filled[:, self.env_step] = np.ones([self.n_envs, 1])
                # store transition data
                transition = (obs_n, actions_dict, state, rewards, terminated, available_actions)
                self.agents.memory.store_transitions(self.env_step, *transition)
                for i_env in range(self.n_envs):
                    if envs_done[i_env]:
                        filled[i_env, self.env_step, 0] = 0
                    else:
                        self.current_step += 1
                    if terminated[i_env] or truncated[i_env]:  # one env is terminal
                        episode_score.append(info[i_env]["episode_score"])
                        available_actions = self.envs.get_avail_actions()
                        # log
                        if self.use_wandb:
                            step_info["Episode-Steps/env-%d" % i_env] = info[i_env]["episode_step"]
                            step_info["Train-Episode-Rewards/env-%d" % i_env] = info[i_env]["episode_score"]
                        else:
                            step_info["Train-Results/Episode-Steps"] = {"env-%d" % i_env: info[i_env]["episode_step"]}
                            step_info["Train-Results/Episode-Rewards"] = {"env-%d" % i_env: info[i_env]["episode_score"]}
                        self.log_infos(step_info, self.current_step)

                        terminal_data = (next_obs_n, next_state, available_actions, filled)
                        if self.on_policy:
                            if terminated[i_env]:
                                values_next = np.array([0.0 for _ in range(self.num_agents)])
                            else:
                                batch_select = np.arange(i_env * self.num_agents, (i_env + 1) * self.num_agents)
                                kwargs = {"state": [next_state[i_env]]}
                                if self.args.agent == "VDAC":
                                    rnn_h_ac_i = self.agents.policy.representation.get_hidden_item(batch_select,
                                                                                                   *rnn_hidden)
                                    kwargs.update({"avail_actions": available_actions[i_env:i_env+1],
                                                   "test_mode": test_mode})
                                    _, _, values_next = self.agents.act(next_obs_n[i_env:i_env+1],
                                                                        *rnn_h_ac_i, **kwargs)
                                else:
                                    rnn_h_critic_i = self.agents.policy.representation_critic.get_hidden_item(batch_select,
                                                                                                              *rnn_hidden_critic)
                                    if self.args.agent == "COMA":
                                        kwargs.update({"actions_n": actions_dict["actions_n"],
                                                       "actions_onehot": actions_dict["act_n_onehot"]})
                                    _, values_next = self.agents.values(next_obs_n[i_env:i_env + 1],
                                                                        *rnn_h_critic_i, **kwargs)
                            self.agents.memory.finish_path(i_env, self.env_step+1, *terminal_data,
                                                           value_next=values_next,
                                                           value_normalizer=self.agents.learner.value_normalizer)
                        else:
                            self.agents.memory.finish_path(i_env, self.env_step + 1, *terminal_data)
                        self.current_step += 1
                self.env_step += 1
            obs_n, state = deepcopy(next_obs_n), deepcopy(next_state)

        if test_mode:
            if self.render and self.args.render_mode == "rgb_array":
                # time, height, width, channel -> time, channel, height, width
                videos_info = {"Videos_Test": np.array([best_videos], dtype=np.uint8).transpose((0, 1, 4, 2, 3))}
                self.log_videos(info=videos_info, fps=self.fps, x_index=self.current_step)
        else:
            self.agents.memory.store_episodes()  # store episode data
            n_epoch = self.agents.n_epoch if self.on_policy else self.n_envs
            train_info = self.agents.train(self.current_step, n_epoch=n_epoch)  # train
            self.log_infos(train_info, self.current_step)

        mean_episode_score = np.mean(episode_score)
        return mean_episode_score

    def test_episodes(self, test_T, n_test_runs):
        test_scores = np.zeros(n_test_runs, np.float)
        last_battles_info = self.get_battles_info()
        for i_test in range(n_test_runs):
            test_scores[i_test] = self.run_episodes(test_mode=True)
        win_rate, allies_dead_ratio, enemies_dead_ratio = self.get_battles_result(last_battles_info)
        mean_test_score = test_scores.mean()
        results_info = {"Test-Results/Mean-Episode-Rewards": mean_test_score,
                        "Test-Results/Win-Rate": win_rate,
                        "Test-Results/Allies-Dead-Ratio": allies_dead_ratio,
                        "Test-Results/Enemies-Dead-Ratio": enemies_dead_ratio}
        self.log_infos(results_info, test_T)
        return mean_test_score, test_scores.std(), win_rate

    def run(self):
        if self.args.test_mode:
            self.render = True
            n_test_episodes = self.args.test_episode
            self.agents.load_model(self.args.model_dir_load)
            test_score_mean, test_score_std, test_win_rate = self.test_episodes(0, n_test_episodes)
            agent_info = f"Algo: {self.args.agent}, Map: {self.args.env_id}, seed: {self.args.seed}, "
            print(agent_info, "Win rate: %.3f, Mean score: %.2f. " % (test_win_rate, test_score_mean))
            print("Finish testing.")
        else:
            test_interval = self.args.eval_interval
            last_test_T = 0
            episode_scores = []
            agent_info = f"Algo: {self.args.agent}, Map: {self.args.env_id}, seed: {self.args.seed}, "
            print(f"Steps: {self.current_step} / {self.running_steps}: ")
            print(agent_info, "Win rate: %-, Mean score: -.")
            last_battles_info = self.get_battles_info()
            time_start = time.time()
            while self.current_step <= self.running_steps:
                score = self.run_episodes(test_mode=False)
                episode_scores.append(score)
                if (self.current_step - last_test_T) / test_interval >= 1.0:
                    last_test_T += test_interval
                    # log train results before testing.
                    train_win_rate, allies_dead_ratio, enemies_dead_ratio = self.get_battles_result(last_battles_info)
                    results_info = {"Train-Results/Win-Rate": train_win_rate,
                                    "Train-Results/Allies-Dead-Ratio": allies_dead_ratio,
                                    "Train-Results/Enemies-Dead-Ratio": enemies_dead_ratio}
                    self.log_infos(results_info, last_test_T)
                    last_battles_info = self.get_battles_info()
                    time_pass, time_left = self.time_estimate(time_start)
                    print(f"Steps: {self.current_step} / {self.running_steps}: ")
                    print(agent_info, "Win rate: %.3f, Mean score: %.2f. " % (train_win_rate, np.mean(episode_scores)),
                          time_pass, time_left)
                    episode_scores = []

            print("Finish training.")
            self.agents.save_model("final_train_model.pth")

        self.agents.finish()

    def benchmark(self):
        test_interval = self.args.eval_interval
        n_test_runs = self.args.test_episode // self.n_envs
        last_test_T = 0

        # test the mode at step 0
        test_score_mean, test_score_std, test_win_rate = self.test_episodes(last_test_T, n_test_runs)
        best_score = {"mean": test_score_mean,
                      "std": test_score_std,
                      "step": self.current_step}
        best_win_rate = test_win_rate

        agent_info = f"Algo: {self.args.agent}, Map: {self.args.env_id}, seed: {self.args.seed}, "
        print(f"Steps: {self.current_step} / {self.running_steps}: ")
        print(agent_info, "Win rate: %.3f, Mean score: %.2f. " % (test_win_rate, test_score_mean))
        last_battles_info = self.get_battles_info()
        time_start = time.time()
        while self.current_step <= self.running_steps:
            # train
            self.run_episodes(test_mode=False)
            # test
            if (self.current_step - last_test_T) / test_interval >= 1.0:
                last_test_T += test_interval
                # log train results before testing.
                train_win_rate, allies_dead_ratio, enemies_dead_ratio = self.get_battles_result(last_battles_info)
                results_info = {"Train-Results/Win-Rate": train_win_rate,
                                "Train-Results/Allies-Dead-Ratio": allies_dead_ratio,
                                "Train-Results/Enemies-Dead-Ratio": enemies_dead_ratio}
                self.log_infos(results_info, last_test_T)

                # test the model
                test_score_mean, test_score_std, test_win_rate = self.test_episodes(last_test_T, n_test_runs)

                if best_score["mean"] < test_score_mean:
                    best_score = {"mean": test_score_mean,
                                  "std": test_score_std,
                                  "step": self.current_step}
                if best_win_rate < test_win_rate:
                    best_win_rate = test_win_rate
                    self.agents.save_model("best_model.pth")  # save best model

                last_battles_info = self.get_battles_info()

                # Estimate the physic running time
                time_pass, time_left = self.time_estimate(time_start)
                print(f"Steps: {self.current_step} / {self.running_steps}: ")
                print(agent_info, "Win rate: %.3f, Mean score: %.2f. " % (test_win_rate, test_score_mean), time_pass, time_left)

        # end benchmarking
        print("Finish benchmarking.")
        print("Best Score: %.4f, Std: %.4f" % (best_score["mean"], best_score["std"]))
        print("Best Win Rate: {}%".format(best_win_rate * 100))

        self.agents.finish()

    def time_estimate(self, start):
        current_step = self.agents.current_step
        time_pass = int(time.time() - start)
        time_left = int((self.running_steps - current_step) / current_step * time_pass)
        if time_left < 0:
            time_left = 0
        hours_pass, hours_left = time_pass // 3600, time_left // 3600
        min_pass, min_left = np.mod(time_pass, 3600) // 60, np.mod(time_left, 3600) // 60
        sec_pass, sec_left = np.mod(np.mod(time_pass, 3600), 60), np.mod(np.mod(time_left, 3600), 60)
        INFO_time_pass = f"Time pass: {hours_pass}h{min_pass}m{sec_pass}s,"
        INFO_time_left = f"Time left: {hours_left}h{min_left}m{sec_left}s"
        return INFO_time_pass, INFO_time_left
