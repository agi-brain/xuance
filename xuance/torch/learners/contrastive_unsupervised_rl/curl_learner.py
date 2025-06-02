import torch
import torch.nn as nn
from xuance.common import Optional
from xuance.torch.learners import Learner
from argparse import Namespace
try:
    from torchvision import transforms
except:
    pass


class FrameStackTransform:
    def __init__(self):
        self.transform = transforms.Compose([
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4)], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomHorizontalFlip(),
            transforms.RandomAdjustSharpness(2),
            transforms.RandomPerspective(0.2),
            transforms.RandomRotation(15),
        ])

    def __call__(self, x: torch.Tensor):
        # x: (B, H, W, C), C=4
        x = x.permute(0,3,1,2)
        B, C, H, W = x.shape
        x_merged = x.contiguous().reshape(B*C, 1, H, W)  # (B*C,1,H,W)
        x_transformed = self.transform(x_merged)
        _, _, new_H, new_W = x_transformed.shape
        x_transformed = x_transformed.view(B, C, new_H, new_W)
        return x_transformed.permute(0,2,3,1)


class CURL_Learner(Learner):
    def __init__(self,
                 config: Namespace,
                 policy: nn.Module,
                 callback):
        super(CURL_Learner, self).__init__(config, policy, callback)

        self.temperature = config.temperature  #  temperature of InfoNCE Loss
        self.tau = config.tau  # moment update coefficient
        self.sync_frequency = config.sync_frequency

        self.encoder_optim = torch.optim.Adam(
            self.policy.representation.parameters(),
            lr=config.repr_lr,
            eps=1e-5
        )
        self.q_optim = torch.optim.Adam(
            self.policy.parameters(),
            lr=self.learning_rate,
            eps=1e-5
        )

        self.scheduler = torch.optim.lr_scheduler.LinearLR(
            self.q_optim,
            start_factor=1.0,
            end_factor=self.end_factor_lr_decay,
            total_iters=self.running_steps
        )

        self.mse_loss = nn.MSELoss()
        self.ce_loss = nn.CrossEntropyLoss()
        self.one_hot = nn.functional.one_hot
        self.n_actions = self.policy.action_dim

        self.transform =  FrameStackTransform()

    def _update_target_encoder(self):
        with torch.no_grad():
            for target_param, param in zip(
                    self.policy.target_representation.parameters(),
                    self.policy.representation.parameters()
            ):
                target_param.data.copy_(
                    self.tau * target_param.data + (1 - self.tau) * param.data
                )

    def _compute_contrastive_loss(self, obs: torch.Tensor) -> torch.Tensor:
        """InfoNCE Loss"""
        aug_obs_q = self.transform(obs)
        aug_obs_k = self.transform(obs)


        q = self.policy.representation(aug_obs_q)
        with torch.no_grad():
            k = self.policy.target_representation(aug_obs_k)


        q = nn.functional.normalize(q['state'], dim=1)
        k = nn.functional.normalize(k['state'], dim=1)


        logits = torch.mm(q, k.T) / self.temperature  # (batch_size, batch_size)
        labels = torch.arange(obs.size(0), device=obs.device)


        return self.ce_loss(logits, labels)

    def update(self, **samples):
        self.iterations += 1
        obs = torch.as_tensor(samples['obs'], device=self.device)
        act = torch.as_tensor(samples['actions'], device=self.device)
        next_obs = torch.as_tensor(samples['obs_next'], device=self.device)
        rew = torch.as_tensor(samples['rewards'], device=self.device)
        done = torch.as_tensor(samples['terminals'], dtype=torch.float, device=self.device)
        info = self.callback.on_update_start(self.iterations,
                                             policy=self.policy, obs=obs, act=act,
                                             next_obs=next_obs, rew=rew, termination=done)

        # --------------------- update CURL---------------------
        curl_loss = self._compute_contrastive_loss(obs)
        self.encoder_optim.zero_grad()
        curl_loss.backward()
        if self.use_grad_clip:
            torch.nn.utils.clip_grad_norm_(self.policy.representation.parameters(), self.grad_clip_norm)
        self.encoder_optim.step()
        self._update_target_encoder()


        _, _, evalQ = self.policy(obs)
        _, _, targetQ = self.policy.target(next_obs)
        targetQ = targetQ.max(dim=-1).values
        targetQ = rew + self.gamma * (1 - done) * targetQ
        predictQ = (evalQ * self.one_hot(act.long(), evalQ.shape[1])).sum(dim=-1)

        q_loss = self.mse_loss(predictQ, targetQ.detach())
        self.q_optim.zero_grad()
        q_loss.backward()
        if self.use_grad_clip:
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.grad_clip_norm)
        self.q_optim.step()
        if self.scheduler is not None:
            self.scheduler.step()

        if self.iterations % self.sync_frequency == 0:
            self.policy.copy_target()

        lr = self.q_optim.state_dict()['param_groups'][0]['lr']


        info.update({
            "curl_loss": curl_loss.item(),
            "q_loss": q_loss.item(),
            "predictQ": predictQ.mean().item(),
            "learning_rate": lr,
        })
        # print(info)
        if self.distributed_training:
            info.update({f"{k}/rank_{self.rank}": v for k, v in info.items()})

        info.update(self.callback.on_update_end(self.iterations,
                                                policy=self.policy, info=info,
                                                curl_loss=curl_loss, q_loss=q_loss,
                                                evalQ=evalQ, predictQ=predictQ, targetQ=targetQ))

        return info