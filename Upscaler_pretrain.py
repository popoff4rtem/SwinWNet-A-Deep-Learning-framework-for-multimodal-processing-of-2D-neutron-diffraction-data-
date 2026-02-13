import torch
import gc
import torch.nn.functional as F
import math
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler
from supervised_losses import MSELoss, L1Loss, SmoothL1Loss


class UpscalerTrainer:
    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        device,
        loss='SmoothL1Loss',
        optimizer=None,
        scheduler=None,
        num_epochs=50,
        warmup_epochs=10,
        lr=2e-4,
        weight_decay=1e-4,
        use_fp16=True,
        verbose=True
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.loss = loss
        self.num_epochs = num_epochs
        self.warmup_epochs = warmup_epochs
        self.lr = lr
        self.weight_decay = weight_decay
        self.use_fp16 = use_fp16
        self.verbose = verbose

        if self.loss == 'MSELoss':
            self.loss_fn = MSELoss()

        elif self.loss == 'L1Loss':
            self.loss_fn = L1Loss()

        elif self.loss == 'SmoothL1Loss':
            self.loss_fn = SmoothL1Loss()

        self._configure_trainable_layers()

        if optimizer is None:
            self.optimizer = self._build_default_optimizer()
        else:
            self.optimizer = optimizer
            self._rebuild_optimizer_with_trainable_params()

        self.scheduler = scheduler or self._build_default_scheduler()

        self.scaler = GradScaler() if use_fp16 else None

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

        # Unfreeze upscaler
        self._set_requires_grad(self.model.upscaler_encoder, True)
        self._set_requires_grad(self.model.upscaler_bottleneck, True)
        self._set_requires_grad(self.model.upscaler_decoder, True)
        self._set_requires_grad(self.model.upscaler_head, True)

        # Cross attention remains frozen
        self._set_requires_grad(self.model.ca_seg_to_sr, False)
        self._set_requires_grad(self.model.ca_sr_to_seg, False)

    def _rebuild_optimizer_with_trainable_params(self):
        trainable_params = filter(lambda p: p.requires_grad, self.model.parameters())

        optimizer_class = self.optimizer.__class__
        optimizer_defaults = self.optimizer.defaults

        self.optimizer = optimizer_class(
            trainable_params,
            **optimizer_defaults
        )

    # ------------------------------------------------------------
    # 🔹 Optimizer
    # ------------------------------------------------------------

    def _build_default_optimizer(self):
        return torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=self.lr,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=self.weight_decay
        )

    def _build_default_scheduler(self):
        def lr_lambda(epoch):
            if epoch < self.warmup_epochs:
                return (epoch + 1) / self.warmup_epochs
            else:
                progress = (epoch - self.warmup_epochs) / (
                    self.num_epochs - self.warmup_epochs
                )
                return 0.5 * (1 + math.cos(math.pi * progress))

        return torch.optim.lr_scheduler.LambdaLR(
            self.optimizer,
            lr_lambda
        )

    # ------------------------------------------------------------
    # 🔹 Normalization utils (твоя логика)
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

    # ------------------------------------------------------------
    # 🔹 One epoch
    # ------------------------------------------------------------

    def _run_epoch(self, loader, train=True):

        if train:
            self.model.train()
        else:
            self.model.eval()

        total_loss = 0.0

        for hr, _ in tqdm(loader, leave=False):

            hr = hr.to(self.device)
            hr = self.ensure_2ch(hr)

            # ------------------------------
            # segmentation inference
            # ------------------------------
            with torch.no_grad():
                if self.use_fp16:
                    with autocast():
                        seg_pred, skips = self.model.segment_1(hr)
                        seg_pred = torch.sigmoid(seg_pred)
                else:
                    seg_pred, skips = self.model.segment_1(hr)
                    seg_pred = torch.sigmoid(seg_pred)

            hr_masked = seg_pred * hr

            # downscale
            lr = F.interpolate(
                hr_masked,
                scale_factor=0.5,
                mode="bilinear",
                align_corners=False
            )

            norm_lr, _ = self.normalize_piecewise(lr)
            norm_hr, params_hr = self.normalize_piecewise(hr_masked)

            if train:
                self.optimizer.zero_grad()

            # ------------------------------
            # upscaler forward
            # ------------------------------
            if self.use_fp16:
                with autocast():
                    pred, _ = self.model.upscale(norm_lr, skips)
                    loss = self.loss_fn(pred, norm_hr)

                if train:
                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
            else:
                pred, _ = self.model.upscale(norm_lr, skips)
                loss = self.loss_fn(pred, norm_hr)

                if train:
                    loss.backward()
                    self.optimizer.step()

            total_loss += loss.item()

        return total_loss / len(loader)

    # ------------------------------------------------------------
    # 🔹 Training loop
    # ------------------------------------------------------------

    def train(self):

        for epoch in range(self.num_epochs):

            train_loss = self._run_epoch(self.train_loader, train=True)
            val_loss = self._run_epoch(self.val_loader, train=False)

            self.scheduler.step()

            if self.verbose:
                print(
                    f"Epoch [{epoch+1}/{self.num_epochs}] "
                    f"Train Loss: {train_loss:.6f} "
                    f"Val Loss: {val_loss:.6f} "
                    f"LR: {self.optimizer.param_groups[0]['lr']:.2e}"
                )

    # ------------------------------------------------------------
    # 🔹 Removing the optimizer and scheduler
    # ------------------------------------------------------------

    def release_training_state(self):

        # Optimizer
        if hasattr(self, "optimizer") and self.optimizer is not None:
            self.optimizer.state.clear()
            self.optimizer.param_groups.clear()
            del self.optimizer
            self.optimizer = None

        # Scheduler
        if hasattr(self, "scheduler") and self.scheduler is not None:
            del self.scheduler
            self.scheduler = None

        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()