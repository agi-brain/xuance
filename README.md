# XuanPolicy —— "玄策" #
Version: 2.0

**XuanPolicy** is an open-source ensemble of Deep Reinforcement Learning (DRL) algorithm implementations.

We call it as XuanCe. 
"Xuan" means magic box and "Ce" means the policy in Chinese.

DRL algorithms are sensitive to hyper-parameters tuning, varying in performance with different tricks, 
and suffering from unstable training processes, therefore, sometimes DRL algorithms seems elusive and "Xuan". 
This project gives a thorough, high-quality and easy-to-understand implementation of RL algorithms, 
and hope this implementation can give a hint on the magics of reinforcement learning.

We expect it to be compatible with multiple deep learning toolboxes (torch, mindspore, and tensorflow),
and hope it can really become a zoo full of DRL algorithms. 

This project is supported by Peng Cheng Laboratory.

## Installation and Setup ##
Step 1: Create and activate a new conda environment (python=3.7 is suggested):
```
$ conda create -n xuanpolicy python=3.7
$ conda activate xuanpolicy
```

Step 2: Install the python modules with:  

```
$ pip install -r requirement.txt
$ pip install -e git+https://github.com/cycraig/gym-platform#egg=gym_platform
```

step 3: Install the library:

```
$ pip install xuanpolicy
```
Note: Some modules should be installed manually according to the difference devices. 

## Currently Supported Agents ##

### DRL ###
- Vanilla Policy Gradient - PG (pytorch, tensorflow, mindspore)
- Phasic Policy Gradient - PPG (pytorch, tensorflow, mindspore)
- Advantage Actor Critic - A2C (pytorch, tensorflow, mindspore)
- Soft actor-critic based on maximum entropy - SAC(pytorch, tensorflow, mindspore)
- Soft actor-critic for discrete actions - SAC-Discrete(pytorch, tensorflow, mindspore)
- Proximal Policy Optimization with clipped objective - PPO-Clip (pytorch, tensorflow, mindspore)
- Phasic Policy Gradient with KL divergence - PPO-KL (pytorch, tensorflow, mindspore)
- Phasic Policy Gradient with specialized clipping - PPO-Clip (pytorch, tensorflow, mindspore)
- Deep Q Network - DQN(pytorch, tensorflow, mindspore)
- DQN with Double Q-learning - Double DQN (pytorch, tensorflow, mindspore)
- DQN with Dueling network - Dueling DQN (pytorch, tensorflow, mindspore)
- DQN with Prioritized Experience Replay - PER (pytorch, tensorflow, mindspore)
- DQN with Parameter Space Noise for Exploration - NoisyNet (pytorch, tensorflow, mindspore)
- DQN with Convolutional Neural Network - C-DQN (pytorch, tensorflow, mindspore)
- DQN with Long Short-term Memory - L-DQN (pytorch, tensorflow, mindspore)
- DQN with CNN and Long Short-term Memory - CL-DQN (pytorch, tensorflow, mindspore)
- DQN with Quantile Regression - QRDQN (pytorch, tensorflow, mindspore)
- Distributional Reinforcement Learning - C51 (pytorch, tensorflow, mindspore)
- Deep Deterministic Policy Gradient - DDPG (pytorch, tensorflow, mindspore)
- Twin Delayed Deep Deterministic Policy Gradient - TD3 (pytorch, tensorflow, mindspore)
- Soft actor-critic based on maximum entropy with discrete action - SAC-Dis(pytorch, tensorflow, mindspore)
- Parameterised deep Q network - P-DQN (pytorch, tensorflow, mindspore)
- Multi-pass parameterised deep Q network - MP-DQN (pytorch, tensorflow, mindspore)
- Split parameterised deep Q network - SP-DQN (pytorch, tensorflow, mindspore)

### MARL ###
- Independent Q-learning - IQL (pytorch, tensorflow, mindspore)
- Value Decomposition Networks - VDN (pytorch, tensorflow, mindspore)
- Q-mixing networks - QMIX (pytorch, tensorflow, mindspore)
- Weighted Q-mixing networks - WQMIX (pytorch, tensorflow, mindspore)
- Q-transformation - QTRAN (pytorch, tensorflow, mindspore)
- Deep Coordination Graphs - DCG (pytorch, tensorflow, mindspore)
- Independent Deep Deterministic Policy Gradient - IDDPG (pytorch, tensorflow, mindspore)
- Multi-agent Deep Deterministic Policy Gradient - MADDPG (pytorch, tensorflow, mindspore)
- Counterfactual Multi-agent Policy Gradient - COMA (pytorch, tensorflow, mindspore)
- Multi-agent Proximal Policy Optimization - MAPPO (pytorch, tensorflow, mindspore)
- Multi-agent Proximal Policy Optimization with KL divergence - MAPPO-KL (pytorch, tensorflow, mindspore)
- Multi-agent Proximal Policy Optimization with specialized clipping - MAPPO-clip (pytorch, tensorflow, mindspore)
- Mean-Field Q-learning - MFQ (pytorch, tensorflow, mindspore)
- Mean-Field Actor-Critic - MFAC (pytorch, tensorflow, mindspore)
- Independent Soft Actor-Critic - ISAC (pytorch, tensorflow, mindspore)
- Multi-agent Twin Delayed Deep Deterministic Policy Gradient - MATD3 (pytorch, tensorflow, mindspore)

## Used Tricks ## 
- Vectorized Environment
- Multi-processing Training
- Generalized Advantage Estimation
- Observation Normalization
- Reward Normalization
- Advantage Normalization
- Gradient Clipping

You can block any last five tricks as you like by changing the default parameters in functions.

## Basic Usage ##

### Quickly Start ###
```python
import xuanpolicy as xp

runner = xp.get_runner(agent_name='dqn', env_name='toy/CartPole-v0', is_test=False)
runner.run()
```

[//]: # (### Run a Demo ###)

[//]: # (The following four lines of code are enough to start training an RL agent.)

[//]: # (```)

[//]: # ($ python main.py --method dqn --env toy)

[//]: # (```)

[//]: # (As our project support multiprocess communication by mpi4py, so you can run with the following command to start training with K sub-process.)

[//]: # (```)

[//]: # ($ mpiexec -n K python main.py --method dqn --env toy)

[//]: # (```)

[//]: # ()
[//]: # (## Customize Usage ##)

[//]: # (- If you want to train an RL agent in your own environments, you can write an environment wrapper and implement the core function reset&#40;&#41; and step&#40;action&#41; and add it in make_env_funcs.py file. The environment template is shown in "./envs/wrappers/xxx_wrappers.py".)

[//]: # (- If you want to train an agent with some novel network architecture, you can modify content in the function define_network in the xxx_agent.py file in "agents/xxx/xxx_xx_agent". &#40;Hints: Better not playing with the content in define_optimization&#40;&#41; function.&#41;)

## Logger ##
You can use tensorboard to visualize what happened in the training process. After training, the log file will be automatically generated in the directory ".results/" and you should be able to see some training data after running the command.
``` 
$ tensorboard --logdir ./logs/
```
If everything going well, you should get a similar display like below. 

![Tensorboard](./common/debug.png)

To visualize the training scores, training times and the performance, you need to initialize the environment as 
```
env = MonitorVecEnv(DummyVecEnv(...))
```  
then, after training terminated, two extra files "xxx.npy" and "xxx.gif" will be generated in the "./results/" directory. The "xxx.npy" record the scores and clock time for each episode in training. But we haven't provided a plotter.py to draw the curves for this.  


[//]: # (## Experiments ##)

[//]: # (### MuJoCo ###)

[//]: # (We train our agents in MuJoCo benchmark &#40;HalfCheetah,...&#41; for 1M experience and compare with some other implementations &#40;stable-baselines, stable-baselines3, ...&#41;. The performance is shown below. We noticed that the scale of reward in our experiment is different, and we reckon it is mainly because the version of mujoco and the timesteps for each episode. For fair comparsion, we use the same )

[//]: # (hyperparameters for all the implementations.)

[//]: # (#### A2C ####)

[//]: # (| Environments&#40;1M,4 parallels&#41; | Ours | Stable-baselines&#40;tf&#41; |Stable-baselines3&#40;torch&#41;  |)

[//]: # (|  :----:  | :----:  |:--------------------:| :----: |)

[//]: # (| HalfCheetah-v3              |      |                      |                          |)

[//]: # (| Hopper-v3                   |      |                      |                          |)

[//]: # (| Walker2d-v3                 |      |                      |                          |)

[//]: # (| Ant-v3                      |      |                      |                          |)

[//]: # (| Swimmer-v3                  |      |                      |                          |)

[//]: # (| Humanoid-v3                 |      |                      |                          |)

[//]: # ()
[//]: # (#### ACER ####)

[//]: # (| Environments&#40;1M,4 parallels&#41; | Ours |  Stable-baselines&#40;tf&#41;  |Stable-baselines3&#40;torch&#41;  |)

[//]: # (|  :----:  | :----:  | :----: | :----: |)

[//]: # (| HalfCheetah-v3              |      |                      |                          |)

[//]: # (| Hopper-v3                   |      |                      |                          |)

[//]: # (| Walker2d-v3                 |      |                      |                          |)

[//]: # (| Ant-v3                      |      |                      |                          |)

[//]: # (| Swimmer-v3                  |      |                      |                          |)

[//]: # (| Humanoid-v3                 |      |                      |                          |)

[//]: # ()
[//]: # (#### ACKTR ####)

[//]: # (| Environments&#40;1M,4 parallels&#41; | Ours |  Stable-baselines&#40;tf&#41;  |Stable-baselines3&#40;torch&#41;  |)

[//]: # (|  :----:  | :----:  | :----: | :----: |)

[//]: # (| HalfCheetah-v3              |      |                      |                          |)

[//]: # (| Hopper-v3                   |      |                      |                          |)

[//]: # (| Walker2d-v3                 |      |                      |                          |)

[//]: # (| Ant-v3                      |      |                      |                          |)

[//]: # (| Swimmer-v3                  |      |                      |                          |)

[//]: # (| Humanoid-v3                 |      |                      |                          |)

[//]: # ()
[//]: # (#### TRPO ####)

[//]: # (| Environments&#40;1M,4 parallels&#41; | Ours |  Stable-baselines&#40;tf&#41;  |Stable-baselines3&#40;torch&#41;  |)

[//]: # (|  :----:  | :----:  | :----: | :----: |)

[//]: # (| HalfCheetah-v3              |      |                      |                          |)

[//]: # (| Hopper-v3                   |      |                      |                          |)

[//]: # (| Walker2d-v3                 |      |                      |                          |)

[//]: # (| Ant-v3                      |      |                      |                          |)

[//]: # (| Swimmer-v3                  |      |                      |                          |)

[//]: # (| Humanoid-v3                 |      |                      |                          |)

[//]: # ()
[//]: # (#### PPO ####)

[//]: # (| Environments&#40;1M,4 parallels&#41; | Ours |  Stable-baselines&#40;tf&#41;  |Stable-baselines3&#40;torch&#41;  |)

[//]: # (|  :----:  | :----:  | :----: | :----: |)

[//]: # (| HalfCheetah-v3              | ~3283 | ~1336.76&#40;std~133.12&#41;            |                          |)

[//]: # (| Hopper-v3                   |       | ~2764.86&#40;std~1090.03&#41;           |                          |)

[//]: # (| Walker2d-v3                 |       |  ~3094.35&#40;std~83.41&#41;            |                          |)

[//]: # (| Ant-v3                      |       | ~2508.44&#40;std~106.25&#41;            |                          |)

[//]: # (| Swimmer-v3                  |       |  ~43.13&#40;std~1.58&#41;                |                          |)

[//]: # (| Humanoid-v3                 |       |  ~549.35&#40;std~92.78&#41;              |                          |)

[//]: # (| Reacher-v3                  |       |  ~360.45&#40;std~43.95&#41;              |                          |)

[//]: # (| InvertedPendulum-v3                 |      |                      |                          |)

[//]: # (| InvertedDoublePendulum-v3                 |      |                      |                          |)

[//]: # (#### DDPG ####)

[//]: # (| Environments&#40;1M,4 parallels&#41; | Ours |  Stable-baselines&#40;tf&#41;  |Stable-baselines3&#40;torch&#41;  |)

[//]: # (|  :----:  | :----:  | :----: | :----: |)

[//]: # (| HalfCheetah-v3              |      |                 |                          |)

[//]: # (| Hopper-v3                   |      |                      |                          |)

[//]: # (| Walker2d-v3                 |      |                      |                          |)

[//]: # (| Ant-v3                      |      |                      |                          |)

[//]: # (| Swimmer-v3                  |      |                      |                          |)

[//]: # (| Humanoid-v3                 |      |                      |                          |)

[//]: # ()
[//]: # ()
[//]: # (#### TD3 ####)

[//]: # (| Environments&#40;1M,4 parallels&#41; | Ours |  Stable-baselines&#40;tf&#41;  |Stable-baselines3&#40;torch&#41;  |)

[//]: # (|  :----:  | :----:  | :----: | :----: |)

[//]: # (| HalfCheetah-v3              |   |                  |                          |)

[//]: # (| Hopper-v3                   |       |                 |                          |)

[//]: # (| Walker2d-v3                 |       |                      |                          |)

[//]: # (| Ant-v3                      |       |                      |                          |)

[//]: # (| Swimmer-v3                  |      |                      |                          |)

[//]: # (| Humanoid-v3                 |      |                      |                          |)

[//]: # ()
[//]: # (#### SAC ####)

[//]: # (| Environments&#40;1M,4 parallels&#41; | Ours |  Stable-baselines&#40;tf&#41;  |Stable-baselines3&#40;torch&#41;  |)

[//]: # (|  :----:  | :----:  | :----: | :----: |)

[//]: # (| HalfCheetah-v3              |   |                  |                          |)

[//]: # (| Hopper-v3                   |      |                      |                          |)

[//]: # (| Walker2d-v3                 |      |                      |                          |)

[//]: # (| Ant-v3                      |      |                      |                          |)

[//]: # (| Swimmer-v3                  |      |                      |                          |)

[//]: # (| Humanoid-v3                 |      |                      |                          |)

[//]: # ()

XuanPolicy

OpenRelearnware

May. 20, 2023


