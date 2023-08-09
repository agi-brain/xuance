#!/bin/sh

CUDA_VISIBLE_DEVICES=0 python benchmark.py --method c51 --env atari --env-id ALE/Assault-v5

CUDA_VISIBLE_DEVICES=0 python benchmark.py --method c51 --env atari --env-id ALE/Asterix-v5

CUDA_VISIBLE_DEVICES=0 python benchmark.py --method c51 --env atari --env-id ALE/Asteroids-v5
