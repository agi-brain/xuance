# Natural Policy Gradient (NPG)

**Paper Link:** [**https://proceedings.neurips.cc/paper_files/paper/2001/file/4b86abe48d358ecf194c56c69108433e-Paper.pdf**](https://proceedings.neurips.cc/paper_files/paper/2001/file/4b86abe48d358ecf194c56c69108433e-Paper.pdf)

Natural Policy Gradient (NPG) is an algorithm in DRL that aims to optimize the policy 
by using Fisher Information Matrix (FIM) and directly maximizing the expected return. 
It was developed by Sham Kakade in 2001. 
NPG has been widely used in various RL problems, including robotics, finance, and game theory.

This table lists some general features about NPG algorithm:

| Features of NPG   | Values | Description                                              |
|-------------------|--------|----------------------------------------------------------|
| On-policy         | ✅      | The evaluate policy is the same as the target policy.    |
| Off-policy        | ❌      | The evaluate policy is different from the target policy. | 
| Model-free        | ✅      | No need to prepare an environment dynamics model.        | 
| Model-based       | ❌      | Need an environment model to train the policy.           | 
| Discrete Action   | ✅      | Deal with discrete action space.                         |   
| Continuous Action | ✅      | Deal with continuous action space.                       |    

## Policy Optimization
The key idea of NPG is to optimize the policy by computing the gradient of the expected return with respect to the policy parameters. Let's denote the policy as $\pi(\theta)$ where $\theta$  is the parameter vector. The expected return $J(\theta)$ is given by:

$$
J(\theta) = \mathbb{E}_{\pi(\theta)}[R] = \sum_s \rho^\pi(s) \sum_a \pi(a; s, \theta) R(s, a)
$$

where $\rho^\pi(s)$ is the stationary distribution of the policy $\pi$. The gradient of the expected return with respect to $\theta$ is: 

$$
\nabla_{\theta} J(\theta) = \sum_s \rho^\pi(s) \sum_a \nabla_{\theta} \pi(a; s, \theta) R(s, a)
$$

The update rule for the policy parameters $\theta$ is given by:

$$
\theta_{t+1} = \theta_t + \alpha \nabla_{\theta} J(\theta_t)
$$

where $\alpha$ is the learning rate.

## Fisher Information Matrix
In the context of NPG, the Fisher Information Matrix (FIM) plays a crucial role. The FIM is a matrix that measures the amount of information that a random variable contains about an unknown parameter. In the case of policy optimization, the FIM is defined based the probability distribution of the actions under the policy.

For a stochastic policy $\pi(a;s,\theta)$, the FIM is given by:

$$
F_s(\theta) \equiv E_{\pi(a; s, \theta)} \left[ \frac{\partial \log \pi(a; s, \theta)}{\partial \theta_i} \frac{\partial \log \pi(a; s, \theta)}{\partial \theta_j} \right]
$$

Here are some important properties of the FIM:
- **Positive Definiteness**
  - The FIM is typically positive definite. This property ensures that the natural gradient, which is based on the inverse of the FIM, is well-defined and points in the direction of the steepest descent of the expected return.

- **Information Content**
  - The elements of FIM measure the mutual information between the parameters $\theta$ and the actions $a$. A large value of $F_s(\theta)$ indicates that the action distribution contains a lot of information about the parameters, and vice versa.

- **Invariance**
  - The FIM is invariant under reparameterization of the policy. This means that the choice of the parameterization of the policy does not affect the value of FIM, as long as the underlying probability distribution of the actions remains the same. 

The FIM is used to define the metric for the natural gradient. The steepest descent direction of the expected return is given by:

$$
\tilde{\nabla}_\eta(\theta) \equiv F(\theta)^{-1} \nabla_\eta(\theta)
$$

where $F(\theta) = E_{\rho^{\pi}(s)} \left[ F_s(\theta) \right]$ is the average Fisher Information Matrix over the stationary distribution of the policy.

## Actor-Critic Framework
NPG can be implemented in an actor-critic framework. In this framework, the actor network is responsible for generating actions based on the current state, and the critic network is responsible for estimating the value of the state-action pairs. The actor and critic networks are trained jointly to optimize the policy.



Strengths of NPG:
- Simple and intuitive: NPG has a simple and intuitive update rule that directly maximizes the expected return.
- Can handle discrete and continuous actions: NPG can be applied to both discrete and continuous action spaces, making it suitable for a wide range of RL problems.
- Efficient use of data: NPG only requires sampling trajectories from the environment, which can be done efficiently, especially in high-dimensional spaces.

## Algorithm

The full algorithm for training NPG is presented in Algorithm 1:

```{eval-rst}
.. image:: ./../../../_static/figures/pseucodes/pseucode-NPG.png
    :width: 80%
    :align: center
```

## Run NPG in XuanCe

Before running NPG in XuanCe, you need to prepare a conda environment and install ``xuance`` following 
the [**installation steps**](./../../usage/installation.rst#install-xuance).

### Run Build-in Demos

After completing the installation, you can open a Python console and run NPG directly using the following commands:

```python3
import xuance
runner = xuance.get_runner(method='npg',
                           env='classic_control',  # Choices: classic_control, box2d, atari.
                           env_id='CartPole-v1',  # Choices: CartPole-v1, LunarLander-v2, ALE/Breakout-v5, etc.
                           is_test=False)
runner.run()  # Or runner.benchmark()
```

### Run With Self-defined Configs

If you want to run NPG with different configurations, you can build a new ``.yaml`` file, e.g., ``my_config.yaml``.
Then, run the NPG by the following code block:

```python3
import xuance as xp
runner = xp.get_runner(method='npg',
                       env='classic_control',  # Choices: classic_control, box2d, atari.
                       env_id='CartPole-v1',  # Choices: CartPole-v1, LunarLander-v2, ALE/Breakout-v5, etc.
                       config_path="my_config.yaml",  # The path of my_config.yaml file should be correct.
                       is_test=False)
runner.run()  # Or runner.benchmark()
```

To learn more about the configurations, please visit the 
[**tutorial of configs**](./../../api/configs/configuration_examples.rst).

### Run With Custom Environment

If you would like to run XuanCe's NPG in your own environment that was not included in XuanCe, 
you need to define the new environment following the steps in 
[**New Environment Tutorial**](./../../usage/custom_env/custom_drl_env.rst).
Then, [**prepare the configuration file**](./../../usage/custom_env/custom_drl_env.rst#step-2-create-the-config-file-and-read-the-configurations) 
 ``npg_myenv.yaml``.

After that, you can run NPG in your own environment with the following code:

```python3
import argparse
from xuance.common import get_configs
from xuance.environment import REGISTRY_ENV
from xuance.environment import make_envs
from xuance.torch.agents import NPG_Agent

configs_dict = get_configs(file_dir="npg_myenv.yaml")
configs = argparse.Namespace(**configs_dict)
REGISTRY_ENV[configs.env_name] = MyNewEnv

envs = make_envs(configs)  # Make parallel environments.
Agent = NPG_Agent(config=configs, envs=envs)  # Create a NPG agent from XuanCe.
Agent.train(configs.running_steps // configs.parallels)  # Train the model for numerous steps.
Agent.save_model("final_train_model.pth")  # Save the model to model_dir.
Agent.finish()  # Finish the training.
```

## Citation

```{code-block} bash
@article{kakade2001natural,
  title={A natural policy gradient},
  author={Kakade, Sham M},
  journal={Advances in neural information processing systems},
  volume={14},
  year={2001}
}
```
