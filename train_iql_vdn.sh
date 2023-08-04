#!/bin/sh

seed_max=5

echo "max seed is ${seed_max}"
for seed in `seq ${seed_max}`;
do
  echo "seed is ${seed}:"
  CUDA_VISIBLE_DEVICES=0 python benchmark_marl.py --method iql --env sc2 --env-id 2m_vs_1z --seed ${seed}
done

seed_max=5

echo "max seed is ${seed_max}"
for seed in `seq ${seed_max}`;
do
  echo "seed is ${seed}:"
  CUDA_VISIBLE_DEVICES=0 python benchmark_marl.py --method vdn --env sc2 --env-id 2m_vs_1z --seed ${seed}
done