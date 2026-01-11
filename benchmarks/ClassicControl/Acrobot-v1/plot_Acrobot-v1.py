import os
import glob
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from typing import Optional

sns.set(style="darkgrid")

algorithm_legend = {
    'a2c': 'A2C',
    'c51': 'C51',
    'ddqn': 'DoubleDQN',
    'dqn': 'DQN',
    'drqn': 'DRQN',
    'dueldqn': 'DuelingDQN',
    'noisydqn': 'NoisyDQN',
    'perdqn': 'PerDQN',
    'pg': 'PG',
    'ppg': 'PPG',
    'ppo': 'PPO',
    'qrdqn': 'QRDQN',
    'sac': 'SAC',
}
fig_title = "Acrobot-v1"


def load_algo_curves(
        algo_dir: Optional[str] = None,
        algo_name: Optional[str] = None
) -> pd.DataFrame:
    all_dfs = []
    csv_files = glob.glob(os.path.join(algo_dir, "results", "seed_*", "learning_curve.csv"))

    for seed_id, csv_path in enumerate(csv_files):
        df = pd.read_csv(csv_path)

        df = df.iloc[:, :2]
        df.columns = ["step", "avg_return"]

        df["algorithm"] = algo_name
        df["seed"] = seed_id
        all_dfs.append(df)

    return pd.concat(all_dfs, ignore_index=True)


def main():
    root = os.getcwd()

    dfs = []
    for algo in algorithm_legend.keys():
        dfs.append(load_algo_curves(os.path.join(root, algo), algorithm_legend[algo]))

    data = pd.concat(dfs, ignore_index=True)

    plt.figure(figsize=(8, 5))

    sns.lineplot(
        data=data,
        x="step",
        y="avg_return",
        hue="algorithm",
        estimator="mean",
        errorbar="sd",   # mean ± std
        linewidth=2,
    )

    plt.title(f"{fig_title}")
    plt.xlabel("Step")
    plt.ylabel("Average Return")
    plt.tight_layout()
    plt.savefig(f"learning_curves_{fig_title}.pdf", dpi=200)
    # plt.show()


if __name__ == "__main__":
    main()
