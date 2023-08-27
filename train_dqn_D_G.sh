#!/bin/sh

CUDA_VISIBLE_DEVICES=0 python benchmark.py --method dqn --env atari --env-id ALE/DoubleDunk-v5

CUDA_VISIBLE_DEVICES=0 python benchmark.py --method dqn --env atari --env-id ALE/ElevatorAction-v5

CUDA_VISIBLE_DEVICES=0 python benchmark.py --method dqn --env atari --env-id ALE/FishingDerby-v5

CUDA_VISIBLE_DEVICES=0 python benchmark.py --method dqn --env atari --env-id ALE/Frostbite-v5

CUDA_VISIBLE_DEVICES=0 python benchmark.py --method dqn --env atari --env-id ALE/Gopher-v5

CUDA_VISIBLE_DEVICES=0 python benchmark.py --method dqn --env atari --env-id ALE/Gravitar-v5