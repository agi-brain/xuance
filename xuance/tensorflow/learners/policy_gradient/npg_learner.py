import torch
from torch import nn
from xuance.torch.learners import Learner
from argparse import Namespace


class NPG_Learner(Learner):
    def __init__(self,
                 config: Namespace,
                 policy: nn.Module,
                 callback):
        super(NPG_Learner, self).__init__(config, policy, callback)
        self.actor_optimizer = torch.optim.Adam(self.policy.actor.parameters(), config.learning_rate, eps=1e-5)
        self.critic_optimizer = torch.optim.Adam(self.policy.critic.parameters(), config.learning_rate, eps=1e-5)
        self.actor_scheduler = torch.optim.lr_scheduler.LinearLR(self.actor_optimizer,
                                                                 start_factor=1.0,
                                                                 end_factor=self.end_factor_lr_decay,
                                                                 total_iters=config.running_steps)
        self.critic_scheduler = torch.optim.lr_scheduler.LinearLR(self.critic_optimizer,
                                                                  start_factor=1.0,
                                                                  end_factor=self.end_factor_lr_decay,
                                                                  total_iters=config.running_steps)

        self.gamma = config.gamma
        self.mse_loss = nn.MSELoss()

    def update(self, **samples):
        self.iterations += 1
        obs_batch = torch.as_tensor(samples['obs'], device=self.device)
        act_batch = torch.as_tensor(samples['actions'], device=self.device)
        ret_batch = torch.as_tensor(samples['returns'], device=self.device)
        adv_batch = torch.as_tensor(samples['advantages'], device=self.device)
        info = self.callback.on_update_start(self.iterations,
                                             policy=self.policy, obs=obs_batch, act=act_batch,
                                             returns=ret_batch, advantages=adv_batch)

        outputs, a_dist, v_pred = self.policy(obs_batch)
        log_prob = a_dist.log_prob(act_batch)

        a_loss = -(adv_batch * log_prob).mean()  # actor_loss
        c_loss = self.mse_loss(v_pred, ret_batch)  # critic_loss

        # train critic
        self.critic_optimizer.zero_grad()
        c_loss.backward(retain_graph=True)
        if self.use_grad_clip:
            torch.nn.utils.clip_grad_norm_(self.policy.critic.parameters(), self.grad_clip_norm)
        self.critic_optimizer.step()

        #train actor
        self.actor_optimizer.zero_grad()
        a_loss.backward()
        for param in self.policy.actor.parameters():
            if param.requires_grad:
                fisher_info = self.compute_fisher_information(param, obs_batch, act_batch)
                grads = param.grad.view(-1)
                natural_grads = torch.matmul(fisher_info, grads)
                natural_grads = natural_grads.view(param.size())
                param.grad = natural_grads.clone()

        if self.use_grad_clip:
            torch.nn.utils.clip_grad_norm_(self.policy.actor.parameters(), self.grad_clip_norm)

        self.actor_optimizer.step()
        if self.critic_scheduler is not None:
            self.critic_scheduler.step()

        if self.actor_scheduler is not None:
            self.actor_scheduler.step()

            # Logger
        lr_actor = self.actor_optimizer.state_dict()['param_groups'][0]['lr']
        lr_critic = self.actor_optimizer.state_dict()['param_groups'][0]['lr']

        if self.distributed_training:
            info.update({
                f"actor-loss/rank_{self.rank}": a_loss.item(),
                f"critic-loss/rank_{self.rank}": c_loss.item(),
                f"learning_rate_actor/rank_{self.rank}": lr_actor,
                f"learning_rate_critic/rank_{self.rank}": lr_critic,
                f"predict_value/rank_{self.rank}": v_pred.mean().item()
            })
        else:
            info.update({
                "actor-loss": a_loss.item(),
                "critic-loss": c_loss.item(),
                "learning_rate_actor": lr_actor,
                "learning_rate_critic": lr_critic,
                "predict_value": v_pred.mean().item()
            })
        info.update(self.callback.on_update_end(self.iterations,
                                                policy=self.policy, info=info, rep_output=outputs,
                                                a_dist=a_dist, v_pred=v_pred, log_prob=log_prob,
                                                a_loss=a_loss, c_loss=c_loss))
        return info

    def compute_fisher_information(self, params, obs, act):
        param_num = 0
        for param in params:
            param_num += param.numel()
        fisher_information = torch.zeros((param_num, param_num)).to(self.device)
        _, prob, _ = self.policy(obs)
        log_probs = prob.log_prob(act)
        score = torch.autograd.grad(log_probs.sum(), params, retain_graph=True)[0]
        score = score.view(-1).to(self.device)
        fisher_information += torch.outer(score, score) * log_probs.sum().item()
        fisher_information /= self.config.horizon_size
        fisher_information = fisher_information + 1e-3 * torch.eye(fisher_information.shape[0]).to(self.device)
        fisher_inv = torch.linalg.inv(fisher_information)
        return fisher_inv

