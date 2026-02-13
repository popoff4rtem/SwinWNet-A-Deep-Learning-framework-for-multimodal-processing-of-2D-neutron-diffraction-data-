import torch
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm
import torch.nn as nn
from Diffraction_metrics import DiffractionMetricsCalculator
from supervised_losses import MSELoss, L1Loss, SmoothL1Loss
from RL_policy import apply_action


class RLTrainer:

    def __init__(
        self,
        model: nn.Module,
        policy: nn.Module,
        train_loader,
        metrics_calculator=DiffractionMetricsCalculator,
        d_centers=np.linspace(0.05318052, 7.49710258, 1241),
        upscaler_loss = 'SmoothL1Loss',
        device="cuda",
        optimizer_policy=None,
        optimizer_model=None,
        num_epochs=100,
        lambda_rec=10.0,
        lambda_intensity=2.0,
        lambda_peak=1.0,
        lambda_shape=0.5,
    ):

        self.model = model.to(device)
        self.policy = policy.to(device)
        self.train_loader = train_loader
        self.metrics_calculator = metrics_calculator(
            fixed_centers_pred=d_centers,
            fixed_centers_true=d_centers,
            device=device
        )
        self.upscaler_loss = upscaler_loss

        if self.upscaler_loss == 'MSELoss':
            self.upscaler_loss_fn = MSELoss()

        elif self.upscaler_loss == 'L1Loss':
            self.upscaler_loss_fn = L1Loss()

        elif self.upscaler_loss == 'SmoothL1Loss':
            self.upscaler_loss_fn = SmoothL1Loss()

        self.device = device


        # Freeze segmentation tower
        self._configure_trainable_layers()

        # Optimizers
        if optimizer_policy is None:
            self.optimizer_policy = self._build_default_policy_optimizer()
        else:
            self.optimizer_policy = optimizer_policy

        if optimizer_model is None:
            self.optimizer_model = self._build_default_model_optimizer()
        else:
            self.optimizer_model = optimizer_model
            self._rebuild_optimizer_with_trainable_params()

        self.num_epochs = num_epochs

        # Loss weights
        self.lambda_rec = lambda_rec
        self.lambda_intensity = lambda_intensity
        self.lambda_peak = lambda_peak
        self.lambda_shape = lambda_shape


    # ------------------------------------------------------------
    # 🔹 Freeze / Unfreeze logic
    # ------------------------------------------------------------

    def _set_requires_grad(self, module, flag):
        for p in module.parameters():
            p.requires_grad = flag

    def _configure_trainable_layers(self):

        # Freeze segmentator
        self._set_requires_grad(self.model.patch_embed, False)
        self._set_requires_grad(self.model.segmentator_encoder, False)
        self._set_requires_grad(self.model.segmentator_bottleneck, False)
        self._set_requires_grad(self.model.segmentator_decoder, False)
        self._set_requires_grad(self.model.segmentator_head, False)
        self._set_requires_grad(self.model.ca_sr_to_seg, False)

        # Unfreeze upscaler
        self._set_requires_grad(self.model.upscaler_encoder, True)
        self._set_requires_grad(self.model.upscaler_bottleneck, True)
        self._set_requires_grad(self.model.upscaler_decoder, True)
        self._set_requires_grad(self.model.upscaler_head, True)
        self._set_requires_grad(self.model.ca_seg_to_sr, True)


    def _rebuild_optimizer_with_trainable_params(self):
        trainable_params = filter(lambda p: p.requires_grad, self.model.parameters())

        optimizer_class = self.optimizer_model.__class__
        optimizer_defaults = self.optimizer_model.defaults

        self.optimizer_model = optimizer_class(
            trainable_params,
            **optimizer_defaults
        )

    # ------------------------------------------------------------
    # 🔹 Optimizer
    # ------------------------------------------------------------

    def _build_default_policy_optimizer(self):
        return torch.optim.Adam(self.policy.parameters(), lr=1e-4)

    def _build_default_model_optimizer(self):
        return torch.optim.Adam(
            [p for p in self.model.parameters() if p.requires_grad],
            lr=1e-5
        )

    # ------------------------------------------------------------
    # Normalization utils
    # ------------------------------------------------------------

    @staticmethod
    def ensure_2ch(x):
        if x.size(1) == 2:
            return x
        err = torch.sqrt(torch.abs(x))
        return torch.cat([x, err], dim=1)

    @staticmethod
    def normalize_piecewise(x, threshold=0.01, eps=1e-6):
        x_min = x.amin(dim=(2, 3), keepdim=True)
        x_max = x.amax(dim=(2, 3), keepdim=True)
        x01 = (x - x_min) / (x_max - x_min + eps)

        mask = x01 > threshold
        x_strong = torch.log1p(x01)
        x_norm = torch.where(mask, x_strong, x01)

        params = {
            "x_min": x_min,
            "x_max": x_max,
            "threshold": threshold
        }

        return x_norm, params

    @staticmethod
    def denormalize_piecewise(x_norm, params, eps=1e-6):
        x_min = params["x_min"]
        x_max = params["x_max"]
        threshold = params["threshold"]

        mask = x_norm > threshold
        x_strong = torch.expm1(x_norm)
        x01 = torch.where(mask, x_strong, x_norm)

        return x01 * (x_max - x_min + eps) + x_min

    def _rl_train_step(self, images, masks):

        model = self.model
        policy = self.policy
        device = self.device

        model.train()
        policy.train()

        images = images.to(device)
        images = self.ensure_2ch(images)

        # ------------------------------
        # PREPROCESS (no grad)
        # ------------------------------
        with torch.no_grad():
            seg, skips = model.segment_1(images)
            seg_images = images * torch.sigmoid(seg)

            lr = F.interpolate(seg_images, scale_factor=0.5,
                            mode="bilinear", align_corners=False)

            norm_lr, _ = self.normalize_piecewise(lr)
            norm_hr, params_hr = self.normalize_piecewise(seg_images)

        # ======================================================
        # 1) POLICY UPDATE
        # ======================================================
        mu, std = policy(norm_lr)
        dist = torch.distributions.Normal(mu, std)

        alpha = dist.rsample()
        log_prob = dist.log_prob(alpha).sum(dim=1)

        with torch.no_grad():
            sr_out, _ = model.upscale(norm_lr, skips)
            sr_out = apply_action(sr_out, alpha)

            denorm_pred = self.denormalize_piecewise(
                sr_out, params_hr
            )[:, 0:1]

            denorm_tgt = seg_images[:, 0:1]

            allm = self.metrics_calculator(
                batch_pred_2d=denorm_pred,
                batch_true_2d=denorm_tgt,
                peak_params_pred={"scale": False},
                peak_params_true={"scale": False},
                tol=0.05
            )

            int_per = torch.FloatTensor(allm['Integral Intensity']).to(device)
            peak_per = torch.FloatTensor(allm['Peak Intensity']).to(device)
            shape_per = torch.FloatTensor(allm['Shape']).to(device)

            total_per = (
                self.lambda_intensity * int_per +
                self.lambda_peak * peak_per +
                self.lambda_shape * shape_per
            )

            reward = -total_per

        if reward.size(0) != log_prob.size(0):
            padded_reward = torch.zeros_like(log_prob)
            min_size = min(reward.size(0), log_prob.size(0))
            padded_reward[:min_size] = reward.detach()[:min_size]
            policy_loss = -(log_prob * padded_reward).mean()
        else:
            policy_loss = -(log_prob * reward.detach()).mean()

        self.optimizer_policy.zero_grad(set_to_none=True)
        policy_loss.backward()
        self.optimizer_policy.step()

        # ======================================================
        # 2) MODEL UPDATE
        # ======================================================
        sr_out2, _ = model.upscale(norm_lr, skips)

        alpha_det = mu.detach()
        sr_out2 = apply_action(sr_out2, alpha_det)

        rec_per = F.l1_loss(sr_out2, norm_hr)

        sup_loss = self.lambda_rec * rec_per

        self.optimizer_model.zero_grad(set_to_none=True)
        sup_loss.backward()
        self.optimizer_model.step()

        return {
            "reward": reward.mean().item(),
            "rec": rec_per.item(),
            "integral": int_per.mean().item(),
            "peak": peak_per.mean().item(),
            "shape": shape_per.mean().item(),
            "alpha_mean": alpha.mean().item(),
            "alpha_std": alpha.std().item(),
            "policy_loss": policy_loss.item(),
            "sup_loss": sup_loss.item(),
        }

    def train_epoch(self):

        stats = {
            "reward": 0.0,
            "rec": 0.0,
            "integral": 0.0,
            "peak": 0.0,
            "shape": 0.0,
            "alpha_mean": 0.0,
            "alpha_std": 0.0,
        }

        for images, masks in tqdm(self.train_loader):

            out = self._rl_train_step(images, masks)

            for k in stats:
                stats[k] += out[k]

        for k in stats:
            stats[k] /= len(self.train_loader)

        return stats

    def fit(self):

        for ep in range(self.num_epochs):

            st = self.train_epoch()

            print(
                f"[RL] {ep:03d} "
                f"reward={st['reward']:.4f} rec={st['rec']:.4f} "
                f"int={st['integral']:.4f} peak={st['peak']:.4f} "
                f"shape={st['shape']:.6f} "
                f"alpha={st['alpha_mean']:.3f}±{st['alpha_std']:.3f}"
            )