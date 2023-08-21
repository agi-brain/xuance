#!/bin/sh

CUDA_VISIBLE_DEVICES=0 python benchmark.py --method dqn --env atari --env-id ALE/Defender-v5

CUDA_VISIBLE_DEVICES=0 python benchmark.py --method dqn --env atari --env-id ALE/DemonAttack-v5

CUDA_VISIBLE_DEVICES=0 python benchmark.py --method dqn --env atari --env-id ALE/DoubleDunk-v5

CUDA_VISIBLE_DEVICES=0 python benchmark.py --method dqn --env atari --env-id ALE/ElevatorAction-v5

CUDA_VISIBLE_DEVICES=0 python benchmark.py --method dqn --env atari --env-id ALE/Enduro-v5