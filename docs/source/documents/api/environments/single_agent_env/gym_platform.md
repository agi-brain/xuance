# Gym Platform

The Platform environment uses a parameterised action space and continuous state space. 
The task involves an agent learning to avoid enemies and traverse across platforms to reach a goal.

**GitHub repository:** [**https://github.com/cycraig/gym-platform.git**](https://github.com/cycraig/gym-platform.git)

## Overview

There are three actions are available to the agent:

- run(dx)
- hop(dx)
- leap(dx)

A dense reward is given to the agent based on the distance it travels. 
The cumulative return is normalised to 1, achieved by reaching the goal. 
An episode terminates if the agent touches an enemy or falls into a gap between platforms.

## Installation

You can install this environment by the following commands:

```{code-block} bash
git clone https://github.com/cycraig/gym-platform
cd gym-platform
pip install -e '.[gym-platform]'
```

or you can install it via

```{code-block} bash
pip install -e git+https://github.com/cycraig/gym-platform#egg=gym_platform
```

## Usage With XuanCe

In XuanCe, this environment can be run with three algorithms, they are P-DQN, MP-DQN, and SP-DQN.

## Citations

```{code-block} bash
@inproceedings{Masson2016ParamActions,
    author = {Masson, Warwick and Ranchod, Pravesh and Konidaris, George},
    title = {Reinforcement Learning with Parameterized Actions},
    booktitle = {Proceedings of the Thirtieth AAAI Conference on Artificial Intelligence},
    year = {2016},
    location = {Phoenix, Arizona},
    pages = {1934--1940},
    numpages = {7},
    publisher = {AAAI Press},
}
```

```{code-block} bash
@article{bester2019mpdqn,
	author    = {Bester, Craig J. and James, Steven D. and Konidaris, George D.},
	title     = {Multi-Pass {Q}-Networks for Deep Reinforcement Learning with Parameterised Action Spaces},
	journal   = {arXiv preprint arXiv:1905.04388},
	year      = {2019},
	archivePrefix = {arXiv},
	eprinttype    = {arxiv},
	eprint    = {1905.04388},
	url       = {http://arxiv.org/abs/1905.04388},
}
```

## APIs

```{eval-rst}
.. automodule:: xuance.environment.single_agent_env.platform
    :members:
    :undoc-members:
    :show-inheritance:
```


