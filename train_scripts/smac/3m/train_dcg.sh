#!/bin/sh

seed_max=5

echo "max seed is ${seed_max}"
for seed in `seq ${seed_max}`;
do
  echo "seed is ${seed}:"
  CUDA_VISIBLE_DEVICES=0 python benchmark_marl.py --method dcg --env sc2 --env-id 3m --seed ${seed}
done