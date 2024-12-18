Configuration Examples
------------------------------------------------------

As an example, taking the parameter configuration for the DQN algorithm in the Atari environment,
in addition to the basic parameter configuration, the algorithm-specific parameters are stored in the "xuance/configs/dqn/atari.yaml" file.

Due to the presence of over 60 different scenarios in the Atari environment,
where the scenarios are relatively consistent with variations only in tasks,
a single default parameter configuration file is sufficient.

For environments with significant scene variations, such as the "CarRacing-v2" and "LunarLander" scenarios in the "Box2D" environment,
the former has a state input of a 96x96x3 RGB image, while the latter consists of an 8-dimensional vector.
Therefore, the DQN algorithm parameter configurations for these two scenarios are stored in the following two files:

    * xuance/configs/dqn/box2d/CarRacing-v2.yaml
    * xuance/configs/dqn/box2d/LunarLander-v2.yaml

Within the following content, we provid the preset arguments for each implementation that can be run by following the steps in :doc:`Quick Start </documents/usage/basic_usage>`.

.. include:: example_value_based.rst
.. include:: example_policy_based.rst
.. include:: example_marl.rst
