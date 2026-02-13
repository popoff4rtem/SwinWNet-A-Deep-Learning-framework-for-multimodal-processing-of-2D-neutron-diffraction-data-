import torch
import torch.nn.functional as F
from torch import nn


"""
Supervised loss functions for SwinWNet framework.

This module provides a small, consistent set of loss functions for:
- segmentation (binary / regression-like masks)
- upscaling / super‑resolution (regression on images or feature maps)

Design principles:
- Simple, explicit API: loss(pred, target, *[, mask, reduction])*.
- Works with tensors of shape (N, C, H, W) or any broadcast‑compatible shape.
- Uses torch built‑ins under the hood, similar to the Dice / CombinedLoss
  implementations used in the segmentation experiments notebook.
"""


# ---------------------------------------------------------------------------
# Segmentation losses
# ---------------------------------------------------------------------------

# ---- Dice Loss ----
class DiceLoss(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps
    def forward(self, pred_logits, target):
        # pred_logits: (N,1,H,W) logits
        pred = torch.sigmoid(pred_logits)
        target = target.float()
        intersection = (pred * target).sum(dim=[1,2,3])
        union = pred.sum(dim=[1,2,3]) + target.sum(dim=[1,2,3])
        dice = (2. * intersection + self.eps) / (union + self.eps)
        return 1. - dice.mean()

# ---- Tversky / Focal-Tversky ----
class TverskyLoss(nn.Module):
    def __init__(self, alpha=0.5, beta=0.5, eps=1e-6):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.eps = eps
    def forward(self, pred_logits, target):
        pred = torch.sigmoid(pred_logits)
        target = target.float()
        TP = (pred * target).sum(dim=[1,2,3])
        FP = (pred * (1-target)).sum(dim=[1,2,3])
        FN = ((1-pred) * target).sum(dim=[1,2,3])
        tversky = (TP + self.eps) / (TP + self.alpha*FP + self.beta*FN + self.eps)
        return 1.0 - tversky.mean()

class FocalTverskyLoss(nn.Module):
    def __init__(self, alpha=0.5, beta=0.5, gamma=0.75):
        super().__init__()
        self.tversky = TverskyLoss(alpha, beta)
        self.gamma = gamma
    def forward(self, pred_logits, target):
        t = 1.0 - self.tversky(pred_logits, target)  # tversky score
        return (1 - t**self.gamma)

# ---- Focal BCE ----
class FocalBCE(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    def forward(self, logits, target):
        bce = F.binary_cross_entropy_with_logits(logits, target.float(), reduction='none')
        pred_prob = torch.sigmoid(logits)
        p_t = target * pred_prob + (1-target) * (1-pred_prob)
        mod = (1 - p_t) ** self.gamma
        loss = self.alpha * mod * bce
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss

# ---- Combined loss ----
class CombinedLoss(nn.Module):
    def __init__(self, w_bce=1.0, w_dice=1.0, w_boundary=1.0):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()
        self.w_bce = w_bce
        self.w_dice = w_dice
        self.w_boundary = w_boundary
    def forward(self, logits, target, boundary_weight_map=None):
        # boundary_weight_map: same shape as target, optional per-pixel weights
        bce_loss = F.binary_cross_entropy_with_logits(logits, target.float(), reduction='none')
        if boundary_weight_map is not None:
            bce_loss = (bce_loss * boundary_weight_map).mean()
        else:
            bce_loss = bce_loss.mean()
        dice_loss = self.dice(logits, target)
        # Optionally add boundary/surface loss term here (see below)
        loss = self.w_bce * bce_loss + self.w_dice * dice_loss
        return loss


# ---------------------------------------------------------------------------
# Upscaler losses
# ---------------------------------------------------------------------------


class MSELoss(nn.Module):
    """
    Wrapper around mse_loss with the same interface as nn.Module losses.

    Example:
        loss_fn = MSELoss(reduction='mean')
        loss = loss_fn(pred, target)
    """

    def __init__(self, reduction: str = "mean"):
        super().__init__()
        self.reduction = reduction

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor
    ) -> torch.Tensor:
        return F.mse_loss(pred, target)


class L1Loss(nn.Module):
    """
    Wrapper around l1_loss with optional mask support.
    """

    def __init__(self, reduction: str = "mean"):
        super().__init__()
        self.reduction = reduction

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        return F.l1_loss(pred, target)


class SmoothL1Loss(nn.Module):
    """
    Wrapper around smooth_l1_loss with beta parameter.

    Example:
        loss_fn = SmoothL1Loss(beta=1.0, reduction='mean')
        loss = loss_fn(pred, target)
    """

    def __init__(self, beta: float = 1.0, reduction: str = "mean"):
        super().__init__()
        self.beta = beta
        self.reduction = reduction

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        return F.smooth_l1_loss(pred, target, beta=self.beta)


__all__ = [
    # Segmentation losses
    "DiceLoss",
    "TverskyLoss",
    "FocalTverskyLoss",
    "FocalBCE",
    "CombinedLoss"
    # Upscaler losses
    "MSELoss",
    "L1Loss",
    "SmoothL1Loss",
]

