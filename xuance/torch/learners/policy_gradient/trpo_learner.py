"""
Trust Region Policy Optimization (TRPO)
Paper link: https://proceedings.mlr.press/v37/schulman15.pdf
Implementation: PyTorch
"""
from argparse import Namespace
from typing import Callable, List, Sequence

import torch
from torch.distributions import Categorical, Normal, kl_divergence
from xuance.torch import Module, Tensor, nn
from xuance.torch.learners import Learner
from xuance.torch.rl_models.modules.distributions import (
    CategoricalDistribution,
    DiagGaussianDistribution,
)


def _get_flat_params(params: Sequence[nn.Parameter]) -> Tensor:
    """Flatten a sequence of parameters into a single 1-D tensor."""
    return torch.cat([param.data.reshape(-1) for param in params])


def _set_flat_params(params: Sequence[nn.Parameter], flat_params: Tensor) -> None:
    """Copy a 1-D flattened parameter tensor back into a sequence of parameters."""
    offset = 0
    for param in params:
        numel = param.numel()
        param.data.copy_(flat_params[offset:offset + numel].view_as(param))
        offset += numel


def _flat_grad(grads: Sequence[Tensor | None], params: Sequence[nn.Parameter]) -> Tensor:
    """Flatten autograd outputs into a 1-D tensor, substituting zeros for unused parameters."""
    views = []
    for grad, param in zip(grads, params):
        if grad is None:
            views.append(torch.zeros_like(param).reshape(-1))
        else:
            views.append(grad.contiguous().reshape(-1))
    return torch.cat(views)


def conjugate_gradient(
        fvp_fn: Callable[[Tensor], Tensor],
        b: Tensor,
        max_steps: int = 10,
        residual_tol: float = 1e-10
) -> Tensor:
    """Solve Ax = b for a symmetric positive-definite linear operator A via conjugate gradient.

    Args:
        fvp_fn: Callable computing the matrix-vector product A @ v.
        b: Right-hand side 1-D tensor.
        max_steps: Maximum number of conjugate gradient iterations.
        residual_tol: Early-stopping threshold on squared residual norm ||r||^2.

    Returns:
        Approximate solution tensor x of the same shape as b.
    """
    x = torch.zeros_like(b)
    r = b.clone()
    p = b.clone()
    r_dot_r = torch.dot(r, r)

    for _ in range(max_steps):
        if r_dot_r < residual_tol or not torch.isfinite(r_dot_r):
            break
        Ap = fvp_fn(p)
        p_dot_Ap = torch.dot(p, Ap)
        if p_dot_Ap <= 0.0 or not torch.isfinite(p_dot_Ap):
            break
        alpha = r_dot_r / (p_dot_Ap + 1e-8)
        x = x + alpha * p
        r = r - alpha * Ap
        new_r_dot_r = torch.dot(r, r)
        if new_r_dot_r < residual_tol or not torch.isfinite(new_r_dot_r):
            break
        beta = new_r_dot_r / (r_dot_r + 1e-8)
        p = r + beta * p
        r_dot_r = new_r_dot_r

    return x


class TRPO_Learner(Learner):
    """Learner for Trust Region Policy Optimization (TRPO).

    Updates actor parameters via natural policy gradient (Fisher-vector product +
    conjugate gradient) with KL-constrained backtracking line search, and updates
    critic parameters via Adam on MSE value loss.
    """

    def __init__(self,
                 config: Namespace,
                 model: Module,
                 callback):
        super(TRPO_Learner, self).__init__(config, model, callback)
        self.optimizer = torch.optim.Adam(self.model.critic.parameters(), self.config.learning_rate, eps=1e-5)
        self.scheduler = torch.optim.lr_scheduler.LinearLR(self.optimizer,
                                                           start_factor=1.0,
                                                           end_factor=self.end_factor_lr_decay,
                                                           total_iters=self.total_iters)
        self.mse_loss = nn.MSELoss()

        self.target_kl = getattr(config, "target_kl", 0.01)
        self.cg_max_steps = getattr(config, "cg_max_steps", 10)
        self.cg_damping = getattr(config, "cg_damping", 0.1)
        self.cg_residual_tol = getattr(config, "cg_residual_tol", 1e-10)
        self.line_search_max_iter = getattr(config, "line_search_max_iter", 10)
        self.line_search_shrinking_factor = getattr(config, "line_search_shrinking_factor", 0.8)
        self.sub_sampling_factor = max(1, int(getattr(config, "sub_sampling_factor", 1)))
        self.ent_coef = getattr(config, "ent_coef", 0.0)

        self.actor_params: List[nn.Parameter] = [p for p in self.model.actor.parameters() if p.requires_grad]
        self.critic_iterations = 0

    def estimate_total_iterations(self):
        """Estimate total number of critic optimization steps for LR scheduling."""
        buffer_size = max(1, self.config.horizon_size * self.config.parallels)
        update_times = max(1, self.config.running_steps // buffer_size)
        total_iters = max(1, update_times * self.config.n_epochs * self.config.n_minibatch)
        return total_iters

    @staticmethod
    def _snapshot_distribution(dist, step: int = 1):
        """Create an independent detached torch Distribution snapshot.

        XuanCe actor heads reuse a single `self.policy_distribution` instance across
        forward passes, so snapshotting detached parameters prevents subsequent
        forward calls from overwriting the old policy distribution in place.
        """
        if isinstance(dist, CategoricalDistribution):
            logits = dist.logits.detach().clone()
            if step > 1:
                logits = logits[::step]
            return Categorical(logits=logits)
        elif isinstance(dist, DiagGaussianDistribution):
            loc = dist.distribution.loc.detach().clone()
            scale = dist.distribution.scale.detach().clone()
            if step > 1:
                loc = loc[::step]
                scale = scale[::step]
            return Normal(loc=loc, scale=scale)
        else:
            raise NotImplementedError(f"TRPO does not support distribution type {type(dist)}.")

    @staticmethod
    def _compute_kl(old_torch_dist, new_dist) -> Tensor:
        """Compute mean KL(old || new), summing over action dimensions for diagonal Gaussians."""
        kl = kl_divergence(old_torch_dist, new_dist.distribution)
        if kl.ndim > 1:
            kl = kl.sum(dim=-1)
        return kl.mean()

    def _compute_surrogate(self,
                           obs_batch: Tensor,
                           act_batch: Tensor,
                           adv_batch: Tensor,
                           old_log_prob: Tensor):
        """Evaluate the TRPO surrogate objective (to be maximized) and policy entropy."""
        policy_dist = self.model.actor(obs_batch).distributions
        log_prob = policy_dist.log_prob(act_batch)
        ratio = torch.exp(log_prob - old_log_prob)
        surrogate = (ratio * adv_batch).mean()
        entropy = policy_dist.entropy().mean()
        if self.ent_coef != 0.0:
            surrogate = surrogate + self.ent_coef * entropy
        return surrogate, entropy, policy_dist

    def _fisher_vector_product(self, obs_fvp: Tensor, old_dist_fvp, vector: Tensor) -> Tensor:
        """Compute damped Fisher-vector product (F + cg_damping * I) @ vector."""
        new_dist = self.model.actor(obs_fvp).distributions
        kl = self._compute_kl(old_dist_fvp, new_dist)
        kl_grads = torch.autograd.grad(kl, self.actor_params, create_graph=True, allow_unused=True)
        flat_kl_grad = _flat_grad(kl_grads, self.actor_params)
        kl_v = torch.dot(flat_kl_grad, vector)
        fvp_grads = torch.autograd.grad(kl_v, self.actor_params, retain_graph=False, allow_unused=True)
        flat_fvp = _flat_grad(fvp_grads, self.actor_params).detach()
        return flat_fvp + self.cg_damping * vector

    def update_actor(self, **samples) -> dict:
        """Perform one trust-region policy update via CG and backtracking line search."""
        self.iterations += 1
        obs_batch = torch.as_tensor(samples['obs'], device=self.device)
        act_batch = torch.as_tensor(samples['actions'], device=self.device)
        adv_batch = torch.as_tensor(samples['advantages'], device=self.device)
        info = self.callback.on_update_start(self.iterations, method="update_actor",
                                             model=self.model, obs=obs_batch, act=act_batch,
                                             advantages=adv_batch)

        # Snapshot old policy distribution and log-probabilities before any parameter mutation.
        with torch.no_grad():
            old_policy_dist = self.model.actor(obs_batch).distributions
            old_log_prob = old_policy_dist.log_prob(act_batch).detach()
            old_torch_dist = self._snapshot_distribution(old_policy_dist, step=1)
            if self.sub_sampling_factor > 1:
                obs_fvp = obs_batch[::self.sub_sampling_factor]
                old_dist_fvp = self._snapshot_distribution(old_policy_dist, step=self.sub_sampling_factor)
            else:
                obs_fvp = obs_batch
                old_dist_fvp = old_torch_dist

        # Compute surrogate objective and its policy gradient at theta_old.
        old_surrogate, entropy, _ = self._compute_surrogate(obs_batch, act_batch, adv_batch, old_log_prob)
        policy_grads = torch.autograd.grad(old_surrogate, self.actor_params, allow_unused=True)
        flat_policy_grad = _flat_grad(policy_grads, self.actor_params).detach()

        # Solve F x = g via conjugate gradient to obtain the natural gradient step direction.
        fvp_fn = lambda v: self._fisher_vector_product(obs_fvp, old_dist_fvp, v)
        step_dir = conjugate_gradient(fvp_fn, flat_policy_grad,
                                      max_steps=self.cg_max_steps,
                                      residual_tol=self.cg_residual_tol)

        # Compute natural step length satisfying 0.5 * step^T F step = target_kl.
        shs = 0.5 * torch.dot(step_dir, fvp_fn(step_dir))
        grad_dot_step = torch.dot(flat_policy_grad, step_dir)

        old_params = _get_flat_params(self.actor_params)
        accepted = False
        actual_kl = 0.0
        final_surrogate = old_surrogate.detach()
        step_size_val = 0.0

        if torch.isfinite(shs) and shs > 0.0 and torch.isfinite(grad_dot_step) and grad_dot_step > 0.0:
            max_step_coef = torch.sqrt(self.target_kl / (shs + 1e-8))
            full_step = max_step_coef * step_dir

            # Backtracking line search enforcing both surrogate improvement and KL <= target_kl.
            for step_idx in range(self.line_search_max_iter):
                step_frac = self.line_search_shrinking_factor ** step_idx
                candidate_params = old_params + step_frac * full_step
                _set_flat_params(self.actor_params, candidate_params)
                with torch.no_grad():
                    new_surrogate, new_entropy, new_dist = self._compute_surrogate(
                        obs_batch, act_batch, adv_batch, old_log_prob
                    )
                    kl = self._compute_kl(old_torch_dist, new_dist)
                    actual_improve = new_surrogate - old_surrogate.detach()

                if (torch.isfinite(actual_improve)
                        and torch.isfinite(kl)
                        and kl.item() <= self.target_kl
                        and actual_improve.item() > 0.0):
                    accepted = True
                    actual_kl = kl.item()
                    final_surrogate = new_surrogate
                    entropy = new_entropy
                    step_size_val = (step_frac * max_step_coef).item()
                    break

            if not accepted:
                _set_flat_params(self.actor_params, old_params)

        a_loss = -final_surrogate.item()
        e_loss = entropy.item()
        ls_success = 1.0 if accepted else 0.0

        if self.distributed_training:
            info.update({
                f"actor-loss/rank_{self.rank}": a_loss,
                f"entropy/rank_{self.rank}": e_loss,
                f"kl/rank_{self.rank}": actual_kl,
                f"line_search_success/rank_{self.rank}": ls_success,
                f"step_size/rank_{self.rank}": step_size_val,
            })
        else:
            info.update({
                "actor-loss": a_loss,
                "entropy": e_loss,
                "kl": actual_kl,
                "line_search_success": ls_success,
                "step_size": step_size_val,
            })

        info.update(self.callback.on_update_end(self.iterations, method="update_actor",
                                                model=self.model, info=info,
                                                a_loss=a_loss, e_loss=e_loss,
                                                kl=actual_kl, line_search_success=ls_success,
                                                step_size=step_size_val))
        return info

    def update_critic(self, **samples) -> dict:
        """Perform one mini-batch value function update via Adam."""
        self.critic_iterations += 1
        obs_batch = torch.as_tensor(samples['obs'], device=self.device)
        ret_batch = torch.as_tensor(samples['returns'], device=self.device)
        info = self.callback.on_update_start(self.iterations, method="update_critic",
                                             model=self.model, obs=obs_batch, returns=ret_batch)

        v_pred = self.model.critic(obs_batch).values
        c_loss = self.mse_loss(v_pred, ret_batch.detach())

        self.optimizer.zero_grad()
        c_loss.backward()
        if self.use_grad_clip:
            torch.nn.utils.clip_grad_norm_(self.model.critic.parameters(), self.grad_clip_norm)
        self.optimizer.step()
        if self.scheduler is not None:
            self.scheduler.step()

        lr = self.optimizer.state_dict()['param_groups'][0]['lr']

        if self.distributed_training:
            info.update({
                f"critic-loss/rank_{self.rank}": c_loss.item(),
                f"learning_rate/rank_{self.rank}": lr,
                f"predict_value/rank_{self.rank}": v_pred.mean().item(),
            })
        else:
            info.update({
                "critic-loss": c_loss.item(),
                "learning_rate": lr,
                "predict_value": v_pred.mean().item(),
            })

        info.update(self.callback.on_update_end(self.iterations, method="update_critic",
                                                model=self.model, info=info,
                                                v_pred=v_pred, c_loss=c_loss))
        return info

    def update(self, **samples) -> dict:
        """Run both actor trust-region update and critic update on the provided batch."""
        info = {}
        info.update(self.update_actor(**samples))
        info.update(self.update_critic(**samples))
        return info
