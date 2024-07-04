import time
import argparse
import numpy as np
from xuance.common import get_configs, recursive_dict_update
from xuance.environment import make_envs
from xuance.torch.utils.operations import set_seed
from xuance.torch.agents import QMIX_Agents


def parse_args():
    parser = argparse.ArgumentParser("Example: QMIX of XuanCe for SMAC environments.")
    parser.add_argument("--env-id", type=str, default="3m")
    parser.add_argument("--test", type=int, default=0)
    parser.add_argument("--benchmark", type=int, default=1)

    return parser.parse_args()


class SC2_Runner:
    def __init__(self, config):
        set_seed(config.seed)  # Set the random seed.
        self.envs = make_envs(config)  # Make the environment.
        self.n_envs = self.envs.num_envs  # Get the number of parallel envs.
        self.agents = QMIX_Agents(config=config, envs=self.envs)  # Create the Independent PPO agents.
        self.config = config
        self.running_steps = config.running_steps
        self.num_agents, self.num_enemies = self.get_agent_num()

    def get_agent_num(self):
        return self.envs.num_agents, self.envs.num_enemies

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

    def test_episodes(self, test_T, n_test_runs):
        test_scores = np.zeros(n_test_runs, np.float32)
        last_battles_info = self.get_battles_info()
        for i_test in range(n_test_runs):
            running_scores = self.agents.run_episodes(None, n_episodes=self.n_envs, test_mode=True)
            test_scores[i_test] = np.mean(running_scores)
        win_rate, allies_dead_ratio, enemies_dead_ratio = self.get_battles_result(last_battles_info)
        mean_test_score = test_scores.mean()
        results_info = {"Test-Results/Win-Rate": win_rate,
                        "Test-Results/Allies-Dead-Ratio": allies_dead_ratio,
                        "Test-Results/Enemies-Dead-Ratio": enemies_dead_ratio}
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
                    results_info = {"Train-Results/Win-Rate": train_win_rate,
                                    "Train-Results/Allies-Dead-Ratio": allies_dead_ratio,
                                    "Train-Results/Enemies-Dead-Ratio": enemies_dead_ratio}
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

        # test the mode at step 0
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
            self.agents.run_episodes(None, n_episodes=self.n_envs, test_mode=False)
            if self.agents.current_step >= self.agents.start_training:
                train_info = self.agents.train_epochs(n_epochs=self.n_envs)
                self.agents.log_infos(train_info, self.agents.current_step)
            # test
            if (self.agents.current_step - last_test_T) / test_interval >= 1.0:
                last_test_T += test_interval
                # log train results before testing.
                train_win_rate, allies_dead_ratio, enemies_dead_ratio = self.get_battles_result(last_battles_info)
                results_info = {"Train-Results/Win-Rate": train_win_rate,
                                "Train-Results/Allies-Dead-Ratio": allies_dead_ratio,
                                "Train-Results/Enemies-Dead-Ratio": enemies_dead_ratio}
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


if __name__ == "__main__":
    parser = parse_args()
    configs_dict = get_configs(file_dir=f"qmix_sc2_configs/{parser.env_id}.yaml")
    configs_dict = recursive_dict_update(configs_dict, parser.__dict__)
    configs = argparse.Namespace(**configs_dict)

    runner = SC2_Runner(configs)
    if parser.benchmark:
        runner.benchmark()
    else:
        runner.run()



