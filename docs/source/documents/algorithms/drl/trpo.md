# Trust Region Policy Optimization (TRPO)

**Paper Link:** [**https://proceedings.mlr.press/v37/schulman15.pdf**](https://proceedings.mlr.press/v37/schulman15.pdf)

Trust Region Policy Optimization (TRPO) is an on-policy, model-free policy gradient algorithm
introduced by Schulman et al. (ICML 2015). TRPO optimizes a surrogate policy objective subject
to a KL-divergence trust-region constraint between consecutive policies, preventing destructive
policy updates and ensuring stable, near-monotonic improvement.

This table lists some general features about the TRPO algorithm in XuanCe (PyTorch backend):

| Features of TRPO  | Values | Description                                              |
|-------------------|--------|----------------------------------------------------------|
| On-policy         | ✅      | The behavior policy is the same as the target policy.    |
| Off-policy        | ❌      | The behavior policy is different from the target policy. |
| Model-free        | ✅      | No need to prepare an environment dynamics model.        |
| Model-based       | ❌      | Need an environment model to train the policy.           |
| Discrete Action   | ✅      | Supports discrete action spaces (`CategoricalActor`).    |
| Continuous Action | ✅      | Supports continuous action spaces (`GaussianActor`).     |

## Surrogate Objective and Trust-Region Constraint

Given trajectories collected under the current policy $\pi_{\theta_{\text{old}}}$, TRPO solves
the constrained optimization problem:

$$
\max_{\theta} \; \mathcal{L}_{\theta_{\text{old}}}(\theta)
= \mathbb{E}_{s, a \sim \pi_{\theta_{\text{old}}}} \left[
    \frac{\pi_{\theta}(a \mid s)}{\pi_{\theta_{\text{old}}}(a \mid s)} \hat{A}_{\theta_{\text{old}}}(s, a)
\right]
$$

subject to the average KL-divergence constraint:

$$
\bar{D}_{\text{KL}}\left(\pi_{\theta_{\text{old}}}(\cdot \mid s) \,\|\, \pi_{\theta}(\cdot \mid s)\right) \le \delta
$$

where $\delta$ (`target_kl`) is the trust-region radius.

## Natural Gradient via Conjugate Gradient and Line Search

Using a first-order approximation to the surrogate objective and a second-order approximation
to the KL constraint, the search direction $x \approx F^{-1} g$ is obtained without explicitly
forming or inverting the Fisher Information Matrix $F$:

1. **Fisher-Vector Product (FVP):** Compute $(F + \lambda_{\text{damp}} I) v$ directly via
   double automatic differentiation of the mean KL divergence.
2. **Conjugate Gradient (CG):** Iteratively solve $(F + \lambda_{\text{damp}} I) x = g$ in
   `cg_max_steps` iterations.
3. **Step Scaling & Backtracking Line Search:** Scale the direction by
   $\beta = \sqrt{2 \delta / (x^\top F x)}$ and backtrack with factor $\alpha^k$ until both
   $\bar{D}_{\text{KL}}(\pi_{\theta_{\text{old}}} \| \pi_{\theta_{\text{new}}}) \le \delta$ and
   $\mathcal{L}(\theta_{\text{new}}) > \mathcal{L}(\theta_{\text{old}})$ hold. If no candidate
   satisfies both conditions, the actor parameters are kept unchanged.
4. **Value Function Update:** Optimize the separate critic network $V_\phi(s)$ using Adam on
   mini-batches of the rollout buffer for `n_epochs` epochs.

## Run TRPO in XuanCe

Before running TRPO in XuanCe, make sure `xuance` is installed with PyTorch following
the [**installation steps**](./../../usage/installation.rst#install-xuance).

### Run Built-in Demos

```python3
import xuance

runner = xuance.get_runner(
    algo='trpo',
    env='classic_control',  # Choices: classic_control, box2d, mujoco.
    env_id='CartPole-v1',   # Choices: CartPole-v1, Pendulum-v1, Acrobot-v1, LunarLander-v3, Ant-v4, etc.
)
runner.run(mode='train')  # Or mode='benchmark'
```

### Run With Self-defined Configs

```python3
import xuance

runner = xuance.get_runner(
    algo='trpo',
    env='classic_control',
    env_id='Pendulum-v1',
    config_path='my_trpo_config.yaml',
)
runner.run(mode='train')
```

## Citation

```{code-block} bibtex
@inproceedings{schulman2015trust,
  title={Trust Region Policy Optimization},
  author={Schulman, John and Levine, Sergey and Abbeel, Pieter and Jordan, Michael and Moritz, Philipp},
  booktitle={Proceedings of the 32nd International Conference on Machine Learning (ICML)},
  pages={1889--1897},
  year={2015}
}
```
