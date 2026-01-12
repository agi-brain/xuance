from torch import optim

import torch
import torch.nn as nn
from xuance.torch.learners import Learner
import torch.nn.functional as F
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

class SPR_Learner(Learner):

    def __init__(self, config, policy, callback, temperature=0.1, tau=0.99, repr_lr=1e-4, prediction_steps=3):
        super().__init__(config, policy, callback)
        self.temperature = temperature
        self.tau = tau
        self.prediction_steps = prediction_steps

        self.encoder_optim = optim.Adam(policy.representation.parameters(), lr=repr_lr)
        self.q_optim = optim.Adam(policy.parameters(), lr=config.learning_rate)
        self.transition_model = nn.GRUCell(64 + self.policy.action_dim, 64).to(config.device)
        self.mse_loss = nn.MSELoss()
        self.transform = FrameStackTransform()

    def _update_target_encoder(self):
        for target_param, param in zip(self.policy.target_representation.parameters(),
                                       self.policy.representation.parameters()):
            target_param.data.copy_(self.tau * target_param.data + (1 - self.tau) * param.data)

    def _compute_contrastive_loss(self, obs: torch.Tensor, actions: torch.Tensor):
        aug_obs = self.transform(obs)
        q = self.policy.representation(aug_obs)['state']
        with torch.no_grad():
            k = self.policy.target_representation(obs)['state']
            k = nn.functional.normalize(k, dim=1)

        predicted_latents = [q]
        hx = q.clone()
        actions_one_hot = F.one_hot(actions.long(), num_classes=self.policy.action_dim).float().to(q.device)
        for _ in range(self.prediction_steps):
            hx = self.transition_model(torch.cat([hx, actions_one_hot], dim=1), hx)
            predicted_latents.append(hx)

        total_loss = 0
        for t in range(1, self.prediction_steps + 1):
            logits = torch.mm(predicted_latents[t], k.T) / self.temperature
            labels = torch.arange(obs.size(0), device=obs.device)
            total_loss += nn.CrossEntropyLoss()(logits, labels)
        return total_loss / self.prediction_steps

    def update(self, **samples):
        obs = torch.as_tensor(samples['obs'], device=self.device)
        actions = torch.as_tensor(samples['actions'], device=self.device)
        next_obs = torch.as_tensor(samples['obs_next'], device=self.device)
        rew = torch.as_tensor(samples['rewards'], device=self.device)
        done = torch.as_tensor(samples['terminals'], dtype=torch.float, device=self.device)
        info = self.callback.on_update_start(self.iterations,
                                             policy=self.policy, obs=obs, act=actions,
                                             next_obs=next_obs, rew=rew, termination=done)

        spr_loss = self._compute_contrastive_loss(obs, actions)
        self.encoder_optim.zero_grad()
        spr_loss.backward()
        if self.use_grad_clip:
            torch.nn.utils.clip_grad_norm_(self.policy.representation.parameters(), self.grad_clip_norm)
        self.encoder_optim.step()
        self._update_target_encoder()

        _, _, evalQ = self.policy(obs)
        _, _, targetQ = self.policy.target(next_obs)
        predictQ = rew + self.gamma * (1 - done) * targetQ.max(dim=1).values
        q_loss = self.mse_loss(evalQ.gather(1, actions.long().unsqueeze(1)).squeeze(), predictQ)

        self.q_optim.zero_grad()
        q_loss.backward()
        if self.use_grad_clip:
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.grad_clip_norm)
        self.q_optim.step()

        # print("spr_loss:", spr_loss.item(), "q_loss:", q_loss.item())
        info.update({
            "spr_loss": spr_loss.item(),
            "q_loss": q_loss.item(),
            "learning_rate": self.q_optim.param_groups[0]['lr']
        })

        info.update(self.callback.on_update_end(self.iterations,
                                                policy=self.policy, info=info,
                                                spr_loss=spr_loss, q_loss=q_loss,
                                                evalQ=evalQ, predictQ=predictQ, targetQ=targetQ))

        return info
