from xuanpolicy.torch.learners import *


class PPOCLIP_Learner(Learner):
    def __init__(self,
                 policy: nn.Module,
                 optimizer: torch.optim.Optimizer,
                 scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
                 device: Optional[Union[int, str, torch.device]] = None,
                 modeldir: str = "./",
                 vf_coef: float = 0.25,
                 ent_coef: float = 0.005,
                 clip_range: float = 0.25):
        super(PPOCLIP_Learner, self).__init__(policy, optimizer, scheduler, device, modeldir)
        self.vf_coef = vf_coef
        self.ent_coef = ent_coef
        self.clip_range = clip_range

    def update(self, obs_batch, act_batch, ret_batch, adv_batch, old_logp):
        self.iterations += 1
        act_batch = torch.as_tensor(act_batch, device=self.device)
        ret_batch = torch.as_tensor(ret_batch, device=self.device)
        adv_batch = torch.as_tensor(adv_batch, device=self.device)
        old_logp_batch = torch.as_tensor(old_logp, device=self.device)
        outputs, a_dist, v_pred = self.policy(obs_batch)
        log_prob = a_dist.log_prob(act_batch)

        # ppo-clip core implementations 
        ratio = (log_prob - old_logp_batch).exp().float()
        surrogate1 = ratio.clamp(1.0 - self.clip_range, 1.0 + self.clip_range) * adv_batch
        surrogate2 = adv_batch * ratio
        a_loss = -torch.minimum(surrogate1, surrogate2).mean()
        c_loss = F.mse_loss(v_pred, ret_batch)
        e_loss = a_dist.entropy().mean()
        loss = a_loss - self.ent_coef * e_loss + self.vf_coef * c_loss
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.clip_range)
        self.optimizer.step()
        if self.scheduler is not None:
            self.scheduler.step()
        # Logger
        lr = self.optimizer.state_dict()['param_groups'][0]['lr']
        cr = ((ratio < 1 - self.clip_range).sum() + (ratio > 1 + self.clip_range).sum()) / ratio.shape[0]
        
        info = {
            "actor-loss": a_loss.item(),
            "critic-loss": c_loss.item(),
            "entropy": e_loss.item(),
            "learning_rate": lr,
            "predict_value": v_pred.mean().item(),
            "clip_ratio": cr
        }

        return info
