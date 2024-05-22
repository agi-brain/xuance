## Baselines for the papers pull from original implementations modified for the navigation environment

### CADRL:
```python baselines/cadrl/cadrl_navigation/main.py```

### GPG:
Test run:
```
python -u -W ignore baselines/gpg/rl_navigation/main.py \
--project_name "compare_3" \
--env_name "MPE" \
--algorithm_name "gpg" \
--seed 0 \
--graph_type "static" \
--experiment_name "test" \
--scenario_name "navigation_gpg" \
--num_agents=3 \
--num_env_steps 2000000 \
--user_name "marl" --use_wandb
```

### MPNN
```
python baselines/mpnn/nav/main.py --n_rollout_threads=128 --scenario_name='navigation' --use_wandb --verbose --obs_type 'global'
```

### MVDN/VDN/QMIX/MQMIX
```
algo="mvdn" # or "vdn", "mqmix", "qmix"

python baselines/offpolicy/scripts/train/train_mpe.py \
--env_name "MPE" \
--algorithm_name ${algo} --n_rollout_threads=1 \
--experiment_name "test" \
--scenario_name "navigation" \
--num_agents 3 \
--num_landmarks 3 \
--seed 0 \
--episode_length 25 \
--use_soft_update \
--lr 7e-4 --use_reward_normalization \
--hard_update_interval_episode 200 \
--num_env_steps 30000 --use_wandb \
--obs_type='nbd'
```

### MADDPG/RMADDPG/MATD3/RMATD3
```
algo="maddpg" # or "rmaddpg", "matd3", "rmatd3"

python baselines/offpolicy/scripts/train/train_mpe.py \
--env_name "MPE" \
--algorithm_name ${algo} \
--experiment_name "test" \
--scenario_name "navigation" \
--num_agents 3 \
--num_landmarks 3 \
--seed 0 \
--actor_train_interval_step 1 \
--episode_length 25 \
--use_soft_update \
--lr 7e-4 \
--hard_update_interval_episode 200 \
--num_env_steps 10000000 --use_wandb
```