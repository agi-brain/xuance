Version: 1.2.4
==============================================

Date: 12 May, 2024
----------------------------------------------

- Synchronized updated to the MindSpore based on version 1.2.3.
- Added support for multi-GPU distributed training.
- Updated VDAC, COMA, MFQ, MFAC, DCG, QTRAN, and other algorithms for version of 1.2.x.
- Added IAC multi-agent reinforcement learning baseline algorithm.
- Improved the training mechanism for multi-agent adversarial tasks.
- Further standardized and optimized the underlying APIs.

`https://pypi.org/project/xuance/1.2.4/ <https://pypi.org/project/xuance/1.2.4/>`_.

Version: 1.2.3
==============================================

Date: 12 May, 2024
----------------------------------------------

- Synchronized updates to the MindSpore version based on version 1.2.2.
- Further standardized and optimized the underlying APIs.

`https://pypi.org/project/xuance/1.2.3/ <https://pypi.org/project/xuance/1.2.3/>`_.

Version: 1.2.2
==============================================

Date: 12 May, 2024
----------------------------------------------

- Synchronized updates to the TensorFlow 2 version based on version 1.2.1.
- Refactored the Agent module, distinguishing between on-policy and off-policy algorithms.
- Further standardized and optimized the underlying APIs.

`https://pypi.org/project/xuance/1.2.2/ <https://pypi.org/project/xuance/1.2.2/>`_.

Version: 1.2.1
==============================================

Date: 12 May, 2024
----------------------------------------------

- Update the MARL algorithms based on version 1.2.0.
- Add support for non-parameter-sharing in multi-agent systems, with individual agent models indexed by name.
- Improve the algorithms like MADDPG and MASAC with support for RNNs.

`https://pypi.org/project/xuance/1.2.1/ <https://pypi.org/project/xuance/1.2.1/>`_.

Version: 1.2.0
==============================================

Date: 12 May, 2024
----------------------------------------------

- Modify the environment wrapper program to facilitate adding new environments.
- Standardize the wrapping of existing single-agent and multi-agent environments.
- Organize commonly used APIs into a unified structure.
- Integrate the creation of representation, policy, and optimizer into each agent's initialization method.
- Add support for non-parameter-sharing in MARL algorithms.
- Standardize the naming of parameters in config files.

`https://pypi.org/project/xuance/1.2.0/ <https://pypi.org/project/xuance/1.2.0/>`_.

Version: 1.1.1
==============================================

Date: 12 May, 2024
----------------------------------------------

- Improve some algorithms, such as SAC, MASAC, DDPG, and TD3, etc.
- Add multi-robot warehouse (RWARE) environment.
- n_size -> buffer_size;
- The saving and loading of observation status;
- Unify the names of attributes for both single-agent and multi-agent DRL.

Version: 1.1.0
==============================================

Date: 01 May, 2024
----------------------------------------------

- Support MetaDrive environment, and provide examples;
- Refine the configuration settings for the gym-pybullet-drones environment;
- Fix some issues about models saving;
- Revise the model loading method;
- Implement a configuration option for selecting different activation function of the output layer of actors for continuous control;
- Update the corresponding examples to adapt the above changes;
- Fix some other bugs.

Download: `https://pypi.org/project/xuance/1.1.0/ <https://pypi.org/project/xuance/1.1.1/>`_.

Version: 1.0.11
==============================================

Date: 11 April, 2024
-----------------------------------------------

- Support and finish test for gym-pybullet-drones environments;
- Fix some issues for installation of xuance. Now it is more easy to install and use;
- Improve the compatibility for MacOS with Apple's M chips;
- Fix some other bugs.

Download: `https://pypi.org/project/xuance/1.0.11/ <https://pypi.org/project/xuance/1.0.11/>`_.

Version: 1.0.8
==============================================

Date: 3 Jan, 2024
-----------------------------------------------

- Added MiniGrid environment with PPO implementations;
- Added gym-pybullet-drones UAV environment with PPO implementation;
- Fixed the issue of failed parameter reading for multi-agent reinforcement learning with adversarial tasks;
- Fixed several bugs under the TensorFlow and MindSpore frameworks.

Download: `https://pypi.org/project/xuance/1.0.8/ <https://pypi.org/project/xuance/1.0.8/>`_.