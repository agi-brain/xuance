#!/bin/sh

seed_max=5

echo "max seed is ${seed_max}"
for seed in `seq ${seed_max}`;
do
  echo "seed is ${seed}:"
  CUDA_VISIBLE_DEVICES=0 python benchmark_marl.py --method iql --env sc2 --env-id corridor --seed ${seed}
done


seed_max=5

echo "max seed is ${seed_max}"
for seed in `seq ${seed_max}`;
do
  echo "seed is ${seed}:"
  CUDA_VISIBLE_DEVICES=0 python benchmark_marl.py --method vdn --env sc2 --env-id corridor --seed ${seed}
done


seed_max=5

echo "max seed is ${seed_max}"
for seed in `seq ${seed_max}`;
do
  echo "seed is ${seed}:"
  CUDA_VISIBLE_DEVICES=0 python benchmark_marl.py --method qmix --env sc2 --env-id corridor --seed ${seed}
done


seed_max=5

echo "max seed is ${seed_max}"
for seed in `seq ${seed_max}`;
do
  echo "seed is ${seed}:"
  CUDA_VISIBLE_DEVICES=0 python benchmark_marl.py --method wqmix --env sc2 --env-id corridor --seed ${seed}
done


seed_max=5

echo "max seed is ${seed_max}"
for seed in `seq ${seed_max}`;
do
  echo "seed is ${seed}:"
  CUDA_VISIBLE_DEVICES=0 python benchmark_marl.py --method dcg --env sc2 --env-id corridor --seed ${seed}
done


seed_max=5

echo "max seed is ${seed_max}"
for seed in `seq ${seed_max}`;
do
  echo "seed is ${seed}:"
  CUDA_VISIBLE_DEVICES=0 python benchmark_marl.py --method mappo --env sc2 --env-id corridor --seed ${seed}
done


seed_max=5

echo "max seed is ${seed_max}"
for seed in `seq ${seed_max}`;
do
  echo "seed is ${seed}:"
  CUDA_VISIBLE_DEVICES=0 python benchmark_marl.py --method iql --env sc2 --env-id MMM2 --seed ${seed}
done


seed_max=5

echo "max seed is ${seed_max}"
for seed in `seq ${seed_max}`;
do
  echo "seed is ${seed}:"
  CUDA_VISIBLE_DEVICES=0 python benchmark_marl.py --method vdn --env sc2 --env-id MMM2 --seed ${seed}
done


seed_max=5

echo "max seed is ${seed_max}"
for seed in `seq ${seed_max}`;
do
  echo "seed is ${seed}:"
  CUDA_VISIBLE_DEVICES=0 python benchmark_marl.py --method qmix --env sc2 --env-id MMM2 --seed ${seed}
done


seed_max=5

echo "max seed is ${seed_max}"
for seed in `seq ${seed_max}`;
do
  echo "seed is ${seed}:"
  CUDA_VISIBLE_DEVICES=0 python benchmark_marl.py --method wqmix --env sc2 --env-id MMM2 --seed ${seed}
done


seed_max=5

echo "max seed is ${seed_max}"
for seed in `seq ${seed_max}`;
do
  echo "seed is ${seed}:"
  CUDA_VISIBLE_DEVICES=0 python benchmark_marl.py --method dcg --env sc2 --env-id MMM2 --seed ${seed}
done


seed_max=5

echo "max seed is ${seed_max}"
for seed in `seq ${seed_max}`;
do
  echo "seed is ${seed}:"
  CUDA_VISIBLE_DEVICES=0 python benchmark_marl.py --method mappo --env sc2 --env-id MMM2 --seed ${seed}
done
