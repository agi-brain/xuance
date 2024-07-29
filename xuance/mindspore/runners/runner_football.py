import time
import numpy as np
from xuance.mindspore.runners.runner_sc2 import SC2_Runner
from xuance.mindspore.agents import REGISTRY_Agents


class Football_Runner(SC2_Runner):
    def __init__(self, config):
        super(Football_Runner, self).__init__(config)
        config.n_agents = self.envs.num_agents
        self.agents = REGISTRY_Agents[config.agent](config, self.envs)
        self.config = config
        self.running_steps = config.running_steps
        self.num_agents, self.num_adversaries = self.get_agent_num()

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

    def test_episodes(self, test_T, n_test_runs):
        test_scores = np.zeros(n_test_runs, np.float32)
        last_battles_info = self.get_battles_info()
        for i_test in range(n_test_runs):
            running_scores = self.agents.run_episodes(None, n_episodes=self.n_envs, test_mode=True)
            test_scores[i_test] = np.mean(running_scores)
        win_rate = self.get_battles_result(last_battles_info)
        mean_test_score = test_scores.mean()
        results_info = {"Test-Results/Mean-Episode-Rewards": mean_test_score,
                        "Test-Results/Win-Rate": win_rate}
        self.agents.log_infos(results_info, test_T)
        return mean_test_score, test_scores.std(), win_rate

    def run(self):
        if self.config.test_mode:
            n_test_episodes = self.config.test_episode
            self.agents.load_model(self.config.model_dir_load)
            test_score_mean, test_score_std, test_win_rate = self.test_episodes(0, n_test_episodes)
            agent_info = f"Algo: {self.config.agent}, Map: {self.config.env_id}, seed: {self.config.seed}, "
            print(agent_info, "Win rate: %.3f, Mean score: %.2f. " % (test_win_rate, test_score_mean))
            print("Finish testing.")
        else:
            test_interval = self.config.eval_interval
            last_test_T = 0
            episode_scores = []
            agent_info = f"Algo: {self.config.agent}, Map: {self.config.env_id}, seed: {self.config.seed}, "
            print(f"Steps: {self.agents.current_step} / {self.running_steps}: ")
            print(agent_info, "Win rate: %-, Mean score: -.")
            last_battles_info = self.get_battles_info()
            time_start = time.time()
            while self.agents.current_step <= self.running_steps:
                score = self.agents.run_episodes(None, n_episodes=self.n_envs, test_mode=False)
                if self.agents.current_step >= self.agents.start_training:
                    train_info = self.agents.train_epochs(n_epochs=1)
                    self.agents.log_infos(train_info, self.agents.current_step)
                episode_scores.append(np.mean(score))
                if (self.agents.current_step - last_test_T) / test_interval >= 1.0:
                    last_test_T += test_interval
                    # log train results before testing.
                    train_win_rate, allies_dead_ratio, enemies_dead_ratio = self.get_battles_result(last_battles_info)
                    results_info = {"Train-Results/Win-Rate": train_win_rate}
                    self.agents.log_infos(results_info, last_test_T)
                    last_battles_info = self.get_battles_info()
                    time_pass, time_left = self.time_estimate(time_start)
                    print(f"Steps: {self.agents.current_step} / {self.running_steps}: ")
                    print(agent_info, "Win rate: %.3f, Mean score: %.2f. " % (train_win_rate, np.mean(episode_scores)),
                          time_pass, time_left)
                    episode_scores = []

            print("Finish training.")
            self.agents.save_model("final_train_model.pth")

        self.agents.finish()

    def benchmark(self):
        test_interval = self.config.eval_interval
        n_test_runs = self.config.test_episode // self.n_envs
        last_test_T = 0

        # test the model at step 0
        test_score_mean, test_score_std, test_win_rate = self.test_episodes(last_test_T, n_test_runs)
        best_score = {"mean": test_score_mean,
                      "std": test_score_std,
                      "step": self.agents.current_step}
        best_win_rate = test_win_rate

        agent_info = f"Algo: {self.config.agent}, Map: {self.config.env_id}, seed: {self.config.seed}, "
        print(f"Steps: {self.agents.current_step} / {self.running_steps}: ")
        print(agent_info, "Win rate: %.3f, Mean score: %.2f. " % (test_win_rate, test_score_mean))
        last_battles_info = self.get_battles_info()
        time_start = time.time()
        while self.agents.current_step <= self.running_steps:
            # train
            self.agents.run_episodes(test_mode=False)
            if self.agents.current_step >= self.agents.start_training:
                train_info = self.agents.train_epochs(n_epochs=self.n_envs)
                self.agents.log_infos(train_info, self.agents.current_step)
            # test
            if (self.agents.current_step - last_test_T) / test_interval >= 1.0:
                last_test_T += test_interval
                # log train results before testing.
                train_win_rate = self.get_battles_result(last_battles_info)
                results_info = {"Train-Results/Win-Rate": train_win_rate}
                self.agents.log_infos(results_info, last_test_T)

                # test the model
                test_score_mean, test_score_std, test_win_rate = self.test_episodes(last_test_T, n_test_runs)

                if best_score["mean"] < test_score_mean:
                    best_score = {"mean": test_score_mean,
                                  "std": test_score_std,
                                  "step": self.agents.current_step}
                if best_win_rate < test_win_rate:
                    best_win_rate = test_win_rate
                    self.agents.save_model("best_model.pth")  # save best model

                last_battles_info = self.get_battles_info()

                # Estimate the physic running time
                time_pass, time_left = self.time_estimate(time_start)
                print(f"Steps: {self.agents.current_step} / {self.running_steps}: ")
                print(agent_info, "Win rate: %.3f, Mean score: %.2f. " % (test_win_rate, test_score_mean), time_pass, time_left)

        # end benchmarking
        print("Finish benchmarking.")
        print("Best Score: %.4f, Std: %.4f" % (best_score["mean"], best_score["std"]))
        print("Best Win Rate: {}%".format(best_win_rate * 100))

        self.agents.finish()


