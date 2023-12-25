Runner_Football
==============================================

A generic framework for training and testing reinforcement learning in the Football environment.

.. raw:: html

    <br><hr>

PyTorch
------------------------------------------

.. py:class::
    xuance.torch.runners.runner_football.Football_Runner(args)

    :param args: the arguments.
    :type args: Namespace

.. py:function::
    xuance.torch.runners.runner_football.Football_Runner.get_agent_num()

    Return the number of agents and adversaries in the environment.

    :return: the number of agents and adversaries.
    :rtype: tuple

.. py:function::
    xuance.torch.runners.runner_football.Football_Runner.get_battles_info()

    Calculate and return information about the total number of battles in the environment and the number of battles won.

    :return: the formation about the total number of battles in the environment and the number of battles won.
    :rtype: tuple

.. py:function::
    xuance.torch.runners.runner_football.Football_Runner.get_battles_result(last_battles_info)

    Provide the win rate based on changes in the total number of battles and battles won between the last recorded point in time and the current state of the environment.

    :param last_battles_info: contain information about the battles at the last recorded point in time.
    :type last_battles_info: tuple
    :return: the win rate in the environment.
    :rtype: float

.. py:function::
    xuance.torch.runners.runner_football.Football_Runner.run_episodes(test_mode)

    Execute a series of episodes in the environment, either for training or testing purposes.

    :param test_mode: control the mode in which episodes are executed.
    :type test_mode: bool
    :return: the average episode score achieved during the training or testing process.
    :rtype: float

.. py:function::
    xuance.torch.runners.runner_football.Football_Runner.test_episodes(test_T, n_test_runs)

    Run multiple testing cycles in a testing mode and calculate statistics such as test scores and win rates.

    :param test_T: the time step for recording test results.
    :type test_T: int
    :param n_test_runs: the number of testing cycles to execute.
    :type n_test_runs: int
    :return: the average test score, the standard deviation of test scores and the win rate.
    :rtype: tuple

.. py:function::
    xuance.torch.runners.runner_football.Football_Runner.benchmark()

    This method conducts a thorough performance evaluation throughout the training and testing processes,
    providing comprehensive assessment results and statistics..

.. raw:: html

    <br><hr>


Source Code
-----------------

.. tabs::

  .. group-tab:: PyTorch

    .. code-block:: python

        from .runner_sc2 import SC2_Runner
        import numpy as np
        from copy import deepcopy
        import time
        import wandb


        class Football_Runner(SC2_Runner):
            def __init__(self, args):
                self.num_agents, self.num_adversaries = 0, 0
                if args.test:
                    args.parallels = 1
                    args.render = True
                else:
                    args.render = False
                super(Football_Runner, self).__init__(args)

            def get_agent_num(self):
                return self.envs.num_agents, self.envs.num_adversaries

            def get_battles_info(self):
                battles_game, battles_won = self.envs.battles_game.sum(), self.envs.battles_won.sum()
                return battles_game, battles_won

            def get_battles_result(self, last_battles_info):
                battles_game, battles_won = list(last_battles_info)
                incre_battles_game = float(self.envs.battles_game.sum() - battles_game)
                incre_battles_won = float(self.envs.battles_won.sum() - battles_won)
                win_rate = incre_battles_won / incre_battles_game if incre_battles_game > 0 else 0.0
                return win_rate

            def run_episodes(self, test_mode=False):
                episode_score, episode_step, best_score = [], [], -np.inf

                # reset the envs
                obs_n, state, infos = self.envs.reset()
                envs_done = self.envs.buf_done
                self.env_step = 0
                filled = np.zeros([self.n_envs, self.episode_length, 1], np.int32)
                rnn_hidden, rnn_hidden_critic = self.init_rnn_hidden()

                while not envs_done.all():
                    available_actions = self.envs.get_avail_actions()
                    actions_dict = self.get_actions(obs_n, available_actions, rnn_hidden, rnn_hidden_critic,
                                                    state=state, test_mode=test_mode)
                    next_obs_n, next_state, rewards, terminated, truncated, info = self.envs.step(actions_dict['actions_n'])
                    envs_done = self.envs.buf_done
                    rnn_hidden, rnn_hidden_critic = actions_dict['rnn_hidden'], actions_dict['rnn_hidden_critic']

                    if test_mode:
                        for i_env in range(self.n_envs):
                            if terminated[i_env] or truncated[i_env]:
                                episode_score.append(info[i_env]["episode_score"])
                                if best_score < episode_score[-1]:
                                    best_score = episode_score[-1]
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
                                episode_step.append(info[i_env]["episode_step"])
                                available_actions = self.envs.get_avail_actions()
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
                                            kwargs.update({"avail_actions": available_actions[i_env:i_env + 1],
                                                        "test_mode": test_mode})
                                            _, _, values_next = self.agents.act(next_obs_n[i_env:i_env + 1],
                                                                                *rnn_h_ac_i, **kwargs)
                                        else:
                                            rnn_h_critic_i = self.agents.policy.representation_critic.get_hidden_item(
                                                batch_select,
                                                *rnn_hidden_critic)
                                            if self.args.agent == "COMA":
                                                kwargs.update({"actions_n": actions_dict["actions_n"],
                                                            "actions_onehot": actions_dict["act_n_onehot"]})
                                            _, values_next = self.agents.values(next_obs_n[i_env:i_env + 1],
                                                                                *rnn_h_critic_i, **kwargs)
                                    self.agents.memory.finish_path(i_env, self.env_step + 1, *terminal_data,
                                                                value_next=values_next,
                                                                value_normalizer=self.agents.learner.value_normalizer)
                                else:
                                    self.agents.memory.finish_path(i_env, self.env_step + 1, *terminal_data)
                                self.current_step += 1
                        self.env_step += 1
                    obs_n, state = deepcopy(next_obs_n), deepcopy(next_state)

                if not test_mode:
                    self.agents.memory.store_episodes()  # store episode data
                    n_epoch = self.agents.n_epoch if self.on_policy else self.n_envs
                    train_info = self.agents.train(self.current_step, n_epoch=n_epoch)  # train
                    train_info["Train-Results/Train-Episode-Rewards"] = np.mean(episode_score)
                    train_info["Train-Results/Episode-Steps"] = np.mean(episode_step)
                    self.log_infos(train_info, self.current_step)

                mean_episode_score = np.mean(episode_score)
                return mean_episode_score

            def test_episodes(self, test_T, n_test_runs):
                test_scores = np.zeros(n_test_runs, np.float)
                last_battles_info = self.get_battles_info()
                for i_test in range(n_test_runs):
                    test_scores[i_test] = self.run_episodes(test_mode=True)
                win_rate = self.get_battles_result(last_battles_info)
                mean_test_score = test_scores.mean()
                results_info = {"Test-Results/Mean-Episode-Rewards": mean_test_score,
                                "Test-Results/Win-Rate": win_rate}
                self.log_infos(results_info, test_T)
                return mean_test_score, test_scores.std(), win_rate

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
                        train_win_rate = self.get_battles_result(last_battles_info)
                        results_info = {"Train-Results/Win-Rate": train_win_rate}
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

                self.envs.close()
                if self.use_wandb:
                    wandb.finish()
                else:
                    self.writer.close()


