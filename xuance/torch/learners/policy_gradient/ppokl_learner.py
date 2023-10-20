from xuance.torch.learners import *
from xuance.torch.utils.operations import merge_distributions


class PPOKL_Learner(Learner):
    def __init__(self,
                 policy: nn.Module,
                 optimizer: torch.optim.Optimizer,
                 scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
                 device: Optional[Union[int, str, torch.device]] = None,
                 model_dir: str = "./",
                 vf_coef: float = 0.25,
                 ent_coef: float = 0.005,
                 target_kl: float = 0.25):
        super(PPOKL_Learner, self).__init__(policy, optimizer, scheduler, device, model_dir)
        self.vf_coef = vf_coef
        self.ent_coef = ent_coef
        self.target_kl = target_kl
        self.kl_coef = 1.0

    def update(self, obs_batch, act_batch, ret_batch, adv_batch, old_dists):
        self.iterations += 1
        act_batch = torch.as_tensor(act_batch, device=self.device)
        ret_batch = torch.as_tensor(ret_batch, device=self.device)
        adv_batch = torch.as_tensor(adv_batch, device=self.device)

        _, a_dist, v_pred = self.policy(obs_batch)
        log_prob = a_dist.log_prob(act_batch)
        old_dist = merge_distributions(old_dists)
        kl = a_dist.kl_divergence(old_dist).mean()
        old_logp_batch = old_dist.log_prob(act_batch)

        # ppo-clip core implementations 
        ratio = (log_prob - old_logp_batch).exp().float()
        a_loss = -(ratio * adv_batch).mean() + self.kl_coef * kl
        c_loss = F.mse_loss(v_pred, ret_batch)
        e_loss = a_dist.entropy().mean()
        loss = a_loss - self.ent_coef * e_loss + self.vf_coef * c_loss
        if kl > self.target_kl * 1.5:
            self.kl_coef = self.kl_coef * 2.
        elif kl < self.target_kl * 0.5:
            self.kl_coef = self.kl_coef / 2.
        self.kl_coef = np.clip(self.kl_coef, 0.1, 20)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        if self.scheduler is not None:
            self.scheduler.step()
        # Logger
        lr = self.optimizer.state_dict()['param_groups'][0]['lr']

        info = {
            "actor-loss": a_loss.item(),
            "critic-loss": c_loss.item(),
            "entropy": e_loss.item(),
            "learning_rate": lr,
            "kl": kl.item(),
            "predict_value": v_pred.mean().item()
        }

        return info

