import torch
import torch.nn as nn

class AlphaPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(2, 8, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(8, 1)
        )

    def forward(self, x):
        mu = self.net(x)                 # [B,1]
        log_std = torch.zeros_like(mu)   # можно сделать обучаемым, но так ок
        std = torch.exp(log_std)         # [B,1] = 1
        return mu, std

def apply_action(sr_out, alpha):
    # sr_out: [B,2,H,W] или [B,1,H,W], alpha: [B,1]
    a = alpha.view(-1, 1, 1, 1)
    return sr_out * torch.sigmoid(a)