# Policy Gradient (PG)

**Paper Link:** [**Download PDF**](https://proceedings.neurips.cc/paper_files/paper/1999/file/464d828b85b0bed98e80ade0a5c43b0f-Paper.pdf)

The Policy Gradient (PG) algorithm, introduced by 
[Richard Sutton](http://www.incompleteideas.net/) et al. 
in their seminal 1999 paper 
"Policy Gradient Methods for Reinforcement Learning with Function Approximation", 
is a foundational approach in reinforcement learning for optimizing policies directly. 
It is particularly effective in scenarios where value-based methods like Q-learning struggle, 
such as high-dimensional or continuous action spaces.

| Features of PG    | Values | Description                                              |
|-------------------|--------|----------------------------------------------------------|
| On-policy         | ✅      | The evaluate policy is the same as the target policy.    |
| Off-policy        | ❌      | The evaluate policy is different from the target policy. | 
| Model-free        | ✅      | No need to prepare an environment dynamics model.        | 
| Model-based       | ❌      | Need an environment model to train the policy.           | 
| Discrete Action   | ✅      | Deal with discrete action space.                         |   
| Continuous Action | ✅      | Deal with continuous action space.                       |

## Method

## Citation

```{code-block} bash
@article{sutton1999policy,
  title={Policy gradient methods for reinforcement learning with function approximation},
  author={Sutton, Richard S and McAllester, David and Singh, Satinder and Mansour, Yishay},
  journal={Advances in neural information processing systems},
  volume={12},
  year={1999}
}
```

## APIs

### PyTorch

```{eval-rst}
.. automodule:: xuance.torch.agents.policy_gradient.pg_agent
    :members:
    :undoc-members:
    :show-inheritance:
```

### TensorFlow2

```{eval-rst}
.. automodule:: xuance.tensorflow.agents.policy_gradient.pg_agent
    :members:
    :undoc-members:
    :show-inheritance:
```

### MindSpore

```{eval-rst}
.. automodule:: xuance.mindspore.agents.policy_gradient.pg_agent
    :members:
    :undoc-members:
    :show-inheritance:
```
