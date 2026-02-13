import torch
import gc
import torch.nn.functional as F
import math
from tqdm import tqdm
from torch.cuda.amp import GradScaler, autocast
from supervised_losses import DiceLoss, TverskyLoss, FocalTverskyLoss, FocalBCE, CombinedLoss, MSELoss, L1Loss, SmoothL1Loss


class FullModelTrainer:

    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        device,
        segmentator_loss='CombinedLoss',
        upscaler_loss='SmoothL1Loss',
        optimizer=None,
        scheduler=None,
        num_epochs=100,
        warmup_epochs=10,
        lr=2e-4,
        weight_decay=1e-4,
        seg_weight_lr=1.0,
        seg_weight_hr=1.0,
        rec_weight=1.0,
        verbose=True
    ):

        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device

        self.segmentator_loss = segmentator_loss
        self.upscaler_loss = upscaler_loss

        self.num_epochs = num_epochs
        self.seg_weight_lr = seg_weight_lr
        self.seg_weight_hr = seg_weight_hr
        self.rec_weight = rec_weight

        self.warmup_epochs = warmup_epochs
        self.lr = lr
        self.weight_decay = weight_decay
        self.verbose = verbose


        if self.upscaler_loss == 'MSELoss':
            self.loss_fn_upscaler = MSELoss()

        elif self.upscaler_loss == 'L1Loss':
            self.loss_fn_upscaler = L1Loss()

        elif self.upscaler_loss == 'SmoothL1Loss':
            self.loss_fn_upscaler = SmoothL1Loss()

        if self.segmentator_loss == 'CombinedLoss':
            self.loss_fn_segmentator = CombinedLoss()

        elif self.segmentator_loss == 'DiceLoss':
            self.loss_fn_segmentator = DiceLoss()

        elif self.segmentator_loss == 'TverskyLoss':
            self.loss_fn_segmentator = TverskyLoss()

        elif self.segmentator_loss == 'FocalTverskyLoss':
            self.loss_fn_segmentator = FocalTverskyLoss()

        elif self.segmentator_loss == 'FocalBCE':
            self.loss_fn_segmentator = FocalBCE()

        self.optimizer = optimizer or self._build_default_optimizer()
        self.scheduler = scheduler or self._build_default_scheduler()

        self.scaler = GradScaler()

    # ==========================
    # Optimizer & Scheduler
    # ==========================

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
            self.optimizer, lr_lambda
        )

    # ------------------------------------------------------------
    # 🔹 Normalization utils
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

    # ==========================
    # Training
    # ==========================

    def train(self):

        for epoch in range(self.num_epochs):
            train_metrics = self._run_epoch(epoch, train=True)
            val_metrics = self._run_epoch(epoch, train=False)

            self.scheduler.step()

            if self.verbose:
                self._print_epoch_summary(epoch, train_metrics, val_metrics)

    # ==========================
    # Single epoch
    # ==========================

    def _run_epoch(self, epoch, train=True):

        loader = self.train_loader if train else self.val_loader

        if train:
            self.model.train()
        else:
            self.model.eval()

        total_loss = 0
        total_seg_lr = 0
        total_seg_hr = 0
        total_rec = 0

        context = torch.enable_grad() if train else torch.no_grad()

        with context:

            for batch_idx, (images, masks) in enumerate(
                tqdm(loader, leave=False)
            ):

                images = images.to(self.device)
                masks = masks.unsqueeze(1).to(self.device)

                images = self.ensure_2ch(images)

                is_even = (batch_idx % 2 == 0)

                if train:
                    self.optimizer.zero_grad(set_to_none=True)

                with autocast():

                    if is_even:
                        loss, seg_lr, rec = self._even_step(images, masks)
                        seg_hr = 0
                    else:
                        loss, seg_lr, seg_hr = self._odd_step(images, masks)
                        rec = 0

                if train:
                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()

                total_loss += loss.item()
                total_seg_lr += seg_lr
                total_seg_hr += seg_hr
                total_rec += rec

        n = len(loader)

        return {
            "loss": total_loss / n,
            "seg_lr": total_seg_lr / n,
            "seg_hr": total_seg_hr / n,
            "rec": total_rec / n
        }

    # ==========================
    # Even batch logic
    # ==========================

    def _even_step(self, images, masks):
        seg, skips_seg = self.model.segment_1(images)
        loss_seg = self.loss_fn_segmentator(seg, masks)

        seg = torch.sigmoid(seg)
        images_masked = images * seg

        lr = F.interpolate(images_masked, scale_factor=0.5,
                           mode='bilinear', align_corners=False)

        norm_lr, _ = self.normalize_piecewise(lr)
        norm_hr, params_hr = self.normalize_piecewise(images_masked)

        sr_out, _ = self.model.upscale(norm_lr, skips_seg)

        rec = self.loss_fn_upscaler(sr_out, norm_hr)

        loss = (
            loss_seg * self.seg_weight_lr +
            rec * self.rec_weight
        )

        return loss, loss_seg.item(), rec.item()

    # ==========================
    # Odd batch logic
    # ==========================

    def _odd_step(self, images, masks):

        seg, skips_seg = self.model.segment_1(images)
        loss_low = self.loss_fn_segmentator(seg, masks)

        seg = torch.sigmoid(seg)
        images_masked = seg * images

        norm_hr, params_hr = self.normalize_piecewise(images_masked)

        sr_out, skips_sr = self.model.upscale(norm_hr, skips_seg)

        denorm_pred = self.denormalize_piecewise(sr_out, params_hr)

        seg_high, _ = self.model.segment_2(denorm_pred, skips_sr)

        masks_up = F.interpolate(
            masks.float(),
            scale_factor=2,
            mode='nearest-exact'
        ).long()

        loss_high = self.loss_fn_segmentator(seg_high, masks_up)

        loss = (
            loss_low * self.seg_weight_lr +
            loss_high * self.seg_weight_hr
        )

        return loss, loss_low.item(), loss_high.item()

    # ==========================
    # Logging
    # ==========================

    def _print_epoch_summary(self, epoch, train_m, val_m):

        print(
            f"Epoch [{epoch+1}/{self.num_epochs}] "
            f"Summary AVG Train Loss: {train_m['loss']:.4f} | "
            f"AVG Low Res Segmentation Train Loss: {train_m['seg_lr']:.4f} | "
            f"AVG Upscaler Train Loss: {train_m['rec']:.4f} | "
            f"AVG High Res Segmentation Train Loss: {train_m['seg_hr']:.4f} | "
            f"Summary AVG Val Loss: {val_m['loss']:.4f} | "
            f"AVG Low Res Segmentation Val Loss: {val_m['seg_lr']:.4f} | "
            f"AVG Upscaler Val Loss: {val_m['rec']:.4f} | "
            f"AVG High Res Segmentation Val Loss: {val_m['seg_hr']:.4f} | "
            f"LR: {self.optimizer.param_groups[0]['lr']:.2e}"
        )

    # ==========================
    # Memory cleanup
    # ==========================

    def release_training_state(self):

        if hasattr(self, "optimizer") and self.optimizer is not None:
            self.optimizer.state.clear()
            self.optimizer.param_groups.clear()
            del self.optimizer
            self.optimizer = None

        if hasattr(self, "scheduler") and self.scheduler is not None:
            del self.scheduler
            self.scheduler = None

        for p in self.model.parameters():
            p.grad = None

        gc.collect()
        torch.cuda.empty_cache()