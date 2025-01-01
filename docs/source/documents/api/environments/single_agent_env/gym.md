# Gymnasium

Gymnasium is a community-driven toolkit for DRL, 
developed as an enhanced and actively maintained fork of 
[OpenAI's Gym](https://github.com/openai/gym) by the [Farama Foundation](https://farama.org/). 
It provides a standardized interface for building and benchmarking DRL algorithms while addressing the limitations of the original Gym. 
Gymnasium retains backward compatibility with Gym while introducing significant improvements to modernize the toolkit.

**Documentation**: [**https://gymnasium.farama.org/**](https://gymnasium.farama.org/)

```{eval-rst}
.. note::

    Gymnasium is a community-driven fork of OpenAI’s Gym, actively maintained by the Farama Foundation. 
    It offers enhanced APIs, richer info outputs, clear termination criteria, and modern Python support. 
    Unlike Gym, whose updates have slowed, Gymnasium ensures compatibility with new RL libraries, 
    improved documentation, and ongoing support for future advancements.

```

## Classic Control

```{eval-rst}
.. image:: ../../../../_static/figures/classic_control/cart_pole.gif
    :height: 120px
.. image:: ../../../../_static/figures/classic_control/pendulum.gif
    :height: 120px
.. image:: ../../../../_static/figures/classic_control/acrobot.gif
    :height: 120px
```

## Box2D

```{eval-rst}
.. image:: ../../../../_static/figures/box2d/car_racing.gif
    :height: 120px
.. image:: ../../../../_static/figures/box2d/lunar_lander.gif
    :height: 120px
.. image:: ../../../../_static/figures/box2d/bipedal_walker.gif
    :height: 120px
```

## Atari

```{eval-rst}
.. image:: ../../../../_static/figures/atari/adventure.gif
    :height: 150px
.. image:: ../../../../_static/figures/atari/air_raid.gif
    :height: 150px
.. image:: ../../../../_static/figures/atari/alien.gif
    :height: 150px
.. image:: ../../../../_static/figures/atari/boxing.gif
    :height: 150px
.. image:: ../../../../_static/figures/atari/breakout.gif
    :height: 150px
```

## MuJoCo

```{eval-rst}
.. image:: ../../../../_static/figures/mujoco/ant.gif
    :height: 150px
.. image:: ../../../../_static/figures/mujoco/half_cheetah.gif
    :height: 150px
.. image:: ../../../../_static/figures/mujoco/hopper.gif
    :height: 150px
.. image:: ../../../../_static/figures/mujoco/humanoid.gif
    :height: 150px
```

## APIs

```{eval-rst}
.. automodule:: xuance.environment.single_agent_env.gym
    :members:
    :undoc-members:
    :show-inheritance:
```
