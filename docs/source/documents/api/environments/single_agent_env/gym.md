# Gymnasium

Gymnasium is a community-driven toolkit for DRL, 
developed as an enhanced and actively maintained fork of 
[OpenAI's Gym](https://github.com/openai/gym) by the [Farama Foundation](https://farama.org/). 
It provides a standardized interface for building and benchmarking DRL algorithms while addressing the limitations of the original Gym. 
Gymnasium retains backward compatibility with Gym while introducing significant improvements to modernize the toolkit.

**Official documentation:** [**https://gymnasium.farama.org/**](https://gymnasium.farama.org/).

```{eval-rst}
.. note::

    Gymnasium is a community-driven fork of OpenAI's Gym, actively maintained by the Farama Foundation. 
    It offers enhanced APIs, richer info outputs, clear termination criteria, and modern Python support. 
    Unlike Gym, whose updates have slowed, Gymnasium ensures compatibility with new RL libraries, 
    improved documentation, and ongoing support for future advancements. 
    You can visit original Gym's documentation from this link: `https://www.gymlibrary.dev/ <https://www.gymlibrary.dev/>`_

```

## Classic Control

### Overview

```{raw} html
    :file: lists/classic_control_list.html
```

### Features

The Classic Control environment contains five scenarios: CartPole, Mountain Car Continuous, Mountain Car, Acrobot, Pendulum.
These five tasks are usually used as preliminary verification for a DRL algorithm.
The key features of each scenario are summarized in the table below:

<font size=2>

| Env-id                   | Observation Space                                                                                    | Action Space                      |
|--------------------------|------------------------------------------------------------------------------------------------------|-----------------------------------|
| CartPole-v1              | ``Box([-4.8 -inf -0.41887903 -inf], [4.8 inf 0.41887903 inf], (4,), float32)``                       | ``Discrete(2)``                   |
| MountainCarContinuous-v0 | ``Box([-1.2 -0.07], [0.6 0.07], (2,), float32)``                                                     | ``Box(-1.0, 1.0, (1,), float32)`` |
| MountainCar-v0           | ``Box([-1.2 -0.07], [0.6 0.07], (2,), float32)``                                                     | ``Discrete(3)``                   |
| Acrobot-v1               | ``Box([ -1. -1. -1. -1. -12.566371 -28.274334], [ 1. 1. 1. 1. 12.566371 28.274334], (6,), float32)`` | ``Discrete(3)``                   |
| Pendulum-v1              | ``Box([-1. -1. -8.], [1. 1. 8.], (3,), float32)``                                                    | ``Box(-2.0, 2.0, (1,), float32)`` |

</font>

### Arguments

In XuanCe, the arguments for running Classic Control environment are listed below.

| Arguments       | Value/Description                                                                                |
|-----------------|--------------------------------------------------------------------------------------------------|
| ``env_name``    | "Classic Control"                                                                                |
| ``env_seed``    | The env-id.                                                                                      |
| ``vectorize``   | Choose the method to vectorize the environment.<br/> Choices: "DummyVecEnv", "SubprocVecEnv".    |
| ``parallels``   | The number of environments that run in parallel.                                                 |
| ``env_seed``    | The env-seed for the first environment of vectorized environments.                               |
| ``render_mode`` | The render mode to visualize the environment, default is "human". Choices: "human", "rgb_array". |

### Run in XuanCe

In XuanCe, if you want to run the Classic Control environment, 
you can specify the attributes in your config file. For example:

```{code-block} yaml
env_name: "Classic Control"  # The name of classic control environment.
env_id: "CartPole-v1"  # The env-id of the tasks in classic control environment.
env_seed: 1  # The random seed of the task.
vectorize: "SubprocVecEnv"  # Choose the method to vectorize the environment.
parallels: 10  # The number of environments that run in parallel.
render_mode: "rgb_array"  # The render mode.
```

You can also make the environments in the Python console to have a test:

```{code-block} python
from xuance import make_envs
from argparse import Namespace
envs = make_envs(Namespace(env_name="Classic Control", 
                           env_id="CartPole-v1", 
                           vectorize="DummyVecEnv", 
                           parallels=1, 
                           env_seed=1))
envs.reset()                        
```

To run a DRL demo with Classic Control environment, you can see the [Quick Start](../../../usage/basic_usage.rst#run-a-drl-example).

## Box2D

### Overview

```{raw} html
    :file: lists/box2d_list.html
```

### Features

The Box2D environment is built using [box2d](https://box2d.org/) for physics control.
It contains three different scenarios: Bipedal Walker, Car Racing, Lunar Lander.
The key features of each scenario are summarized in the table below:

<font size=2>

| Env-id           | Observation Space                                                                                                                                                                                                                                                             | Action Space                                                |
|------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------|
| BipedalWalker-v3 | ``Box([-3.1415927 -5. -5. -5. -3.1415927 -5. -3.1415927 -5. -0. -3.1415927 -5. -3.1415927 -5. -0. -1. -1. -1. -1. -1. -1. -1. -1. -1. -1. ], [3.1415927 5. 5. 5. 3.1415927 5. 3.1415927 5. 5. 3.1415927 5. 3.1415927 5. 5. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. ], (24,), float32)`` | ``Box(-1.0, 1.0, (4,), float32)``                           |
| CarRacing-v2     | ``Box(0, 255, (96, 96, 3), uint8)``                                                                                                                                                                                                                                           | ``Discrete(5)`` or ``Box([-1. 0. 0.], 1.0, (3,), float32)`` | 
| LunarLander-v2   | ``Box([ -2.5 -2.5 -10. -10. -6.2831855 -10. -0. -0. ], [ 2.5 2.5 10. 10. 6.2831855 10. 1. 1. ], (8,), float32)``                                                                                                                                                              | ``Discrete(4)`` or ``Box(-1, +1, (2,), dtype=np.float32)``  |

</font>

### Installation

The Box2D environment is not included with the installation of XuanCe. 
As an external package, it needs to be installed separately.

```{code-block} bash
pip install swig
pip install gymnasium[box2d]
```

```{eval-rst}

.. note::

    If you're using macOS and encounter the following error:
    
    .. code-block:: bash
        
        zsh: no matches found: gymnasium[box2d]
        
    You can resolve it by typing the command as follows:
    
    .. code-block:: bash
        
        pip install 'gymnasium[box2d]'  
```

### Arguments

In XuanCe, the arguments for running Box2D environment are listed below.

| Arguments       | Value/Description                                                                                                     |
|-----------------|-----------------------------------------------------------------------------------------------------------------------|
| ``env_name``    | "Box2D"                                                                                                               |
| ``env_seed``    | The env-id.                                                                                                           |
| ``vectorize``   | Choose the method to vectorize the environment.<br/> Choices: "DummyVecEnv", "SubprocVecEnv".                         |
| ``parallels``   | The number of environments that run in parallel.                                                                      |
| ``env_seed``    | The env-seed for the first environment of vectorized environments.                                                    |
| ``render_mode`` | The render mode to visualize the environment, default is "human". Choices: "human", "rgb_array".                      |
| ``continuous``  | Determines if discrete or continuous actions will be used. (Only CarRacing-v2 and LunarLander-v2 have this argument.) |

### Run in XuanCe

In XuanCe, if you want to run the Box2D environment, 
you can specify the attributes in your config file. For example:

```{code-block} yaml
env_name: "Box2D"  # The name of classic control environment.
env_id: "BipedalWalker-v3"  # The env-id of the tasks in classic control environment.
env_seed: 1  # The random seed of the task.
vectorize: "SubprocVecEnv"  # Choose the method to vectorize the environment.
parallels: 10  # The number of environments that run in parallel.
render_mode: "rgb_array"  # The render mode.
```

You can also make the environments in the Python console to have a test:

```{code-block} python
from xuance import make_envs
from argparse import Namespace
envs = make_envs(Namespace(env_name="Box2D", 
                           env_id="BipedalWalker-v3", 
                           vectorize="DummyVecEnv", 
                           parallels=1, 
                           env_seed=1))
envs.reset()                        
```

To run a DRL demo with Classic Control environment, you can see the [Quick Start](../../../usage/basic_usage.rst#run-a-drl-example).

## MuJoCo

### Overview

```{raw} html
    :file: lists/mujoco_list.html
```

MuJoCo stands for Multi-Joint dynamics with Contact. 
It is a physics engine for facilitating research and development in robotics, biomechanics, graphics and animation,
and other areas where fast and accurate simulation is needed. 
There is physical contact between the robots and their environment - 
and MuJoCo attempts at getting realistic physics simulations for the possible physical contact dynamics 
by aiming for physical accuracy and computational efficiency.

Learn more about this environment [**here**](https://gymnasium.farama.org/environments/mujoco/).

### Features

The key features of each scenario are summarized in the table below:

<font size=2>

| Env-id                    | Observation Space                    | Action Space                       |
|---------------------------|--------------------------------------|------------------------------------|
| Ant-v4                    | ``Box(-inf, inf, (105,), float64)``  | ``Box(-1.0, 1.0, (8,), float32)``  |
| HalfCheetah-v4            | ``Box(-inf, inf, (17,), float64)``   | ``Box(-1.0, 1.0, (6,), float32)``  |
| Hopper-v4                 | ``Box(-inf, inf, (11,), float64)``   | ``Box(-1.0, 1.0, (3,), float32)``  |
| HumanoidStandIp-v4        | ``Box(-inf, inf, (348,), float64)``  | ``Box(-0.4, 0.4, (17,), float32)`` |
| Humanoid-v4               | ``Box(-inf, inf, (348,), float64)``  | ``Box(-0.4, 0.4, (17,), float32)`` |
| InvertedDoublePendulum-v4 | ``Box(-inf, inf, (9,), float64)``    | ``Box(-1.0, 1.0, (1,), float32)``  |
| InvertedPendulum-v4       | ``Box(-inf, inf, (4,), float64)``    | ``Box(-3.0, 3.0, (1,), float32)``  |
| Pusher-v4                 | ``Box(-inf, inf, (23,), float64)``   | ``Box(-2.0, 2.0, (7,), float32)``  |
| Reacher-v4                | ``Box(-inf, inf, (10,), float64)``   | ``Box(-1.0, 1.0, (2,), float32)``  |
| Swimmer-v4                | ``Box(-inf, inf, (8,), float64)``    | ``Box(-1.0, 1.0, (2,), float32)``  |
| Walker2d-v4               | ``Box(-inf, inf, (17,), float64)``   | ``Box(-1.0, 1.0, (6,), float32)``  |

</font>

### Installation

The MuJoCo environment is not included with the installation of XuanCe. 
As an external package, it needs to be installed separately.

```{code-block} bash
pip install gymnasium[mujoco]
```

```{eval-rst}

.. note::

    If you're using macOS and encounter the following error:
    
    .. code-block:: bash
        
        zsh: no matches found: gymnasium[mujoco]
        
    You can resolve it by typing the command as follows:
    
    .. code-block:: bash
        
        pip install 'gymnasium[mujoco]'  
```

### Arguments

In XuanCe, the arguments for running MuJoCo environment are listed below.

| Arguments       | Value/Description                                                                                                     |
|-----------------|-----------------------------------------------------------------------------------------------------------------------|
| ``env_name``    | "MuJoCo"                                                                                                              |
| ``env_seed``    | The env-id.                                                                                                           |
| ``vectorize``   | Choose the method to vectorize the environment.<br/> Choices: "DummyVecEnv", "SubprocVecEnv".                         |
| ``parallels``   | The number of environments that run in parallel.                                                                      |
| ``env_seed``    | The env-seed for the first environment of vectorized environments.                                                    |
| ``render_mode`` | The render mode to visualize the environment, default is "human". Choices: "human", "rgb_array".                      |

### Run in XuanCe

In XuanCe, if you want to run the MuJoCo environment, 
you can specify the attributes in your config file. For example:

```{code-block} yaml
env_name: "MuJoCo"  # The name of classic control environment.
env_id: "Ant-v4"  # The env-id of the tasks in classic control environment.
env_seed: 1  # The random seed of the task.
vectorize: "SubprocVecEnv"  # Choose the method to vectorize the environment.
parallels: 10  # The number of environments that run in parallel.
render_mode: "rgb_array"  # The render mode.
```

You can also make the environments in the Python console to have a test:

```{code-block} python
from xuance import make_envs
from argparse import Namespace
envs = make_envs(Namespace(env_name="MuJoCo", 
                           env_id="Ant-v4", 
                           vectorize="DummyVecEnv", 
                           parallels=1, 
                           env_seed=1))
envs.reset()                        
```

To run a DRL demo with Classic Control environment, you can see the [Quick Start](../../../usage/basic_usage.rst#run-a-drl-example).


## Atari

### Overview

```{raw} html
    :file: lists/atari_list.html
```

### Feaures

The Atari environment contains 62 different tasks, which are simulated via the 
[Arcade Learning Environment (ALE)](https://www.jair.org/index.php/jair/article/view/10819).



### Installation

The Atari environment is not included with the installation of XuanCe. 
As an external package, it needs to be installed separately.

After installing XuanCe, you need to install Atari dependencies via the following command:

```{code-block} bash
pip install gymnasium[accept-rom-license], gymnasium[atari], atari-py==0.2.9, ale-py==0.7.5
```

```{eval-rst}
.. note::

    If you're using macOS and encounter the following error:
    
    .. code-block:: bash
        
        zsh: no matches found: gymnasium[accept-rom-license]
        
    You can resolve it by typing the command as follows:
    
    .. code-block:: bash
        
        pip install 'gymnasium[accept-rom-license]', 'gymnasium[atari]', atari-py==0.2.9, ale-py==0.7.5
```

### Arguments

### Run in XuanCe

## APIs

```{eval-rst}
.. automodule:: xuance.environment.single_agent_env.gym
    :members:
    :undoc-members:
    :show-inheritance:
```
