import torch
import gc
import math
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler
from supervised_losses import DiceLoss, TverskyLoss, FocalTverskyLoss, FocalBCE, CombinedLoss


class SegmentatorTrainer:
    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        device,
        loss = 'CombinedLoss',
        optimizer=None,
        scheduler=None,
        num_epochs=300,
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


        if self.loss == 'CombinedLoss':
            self.loss_fn = CombinedLoss()

        elif self.loss == 'DiceLoss':
            self.loss_fn = DiceLoss()

        elif self.loss == 'TverskyLoss':
            self.loss_fn = TverskyLoss()

        elif self.loss == 'FocalTverskyLoss':
            self.loss_fn = FocalTverskyLoss()

        elif self.loss == 'FocalBCE':
            self.loss_fn = FocalBCE()

        # self._freeze_non_segmentator()
        self._configure_trainable_layers()

        if optimizer is None:
            self.optimizer = self._build_default_optimizer()
        else:
            self.optimizer = optimizer
            self._rebuild_optimizer_with_trainable_params()

        self.scheduler = scheduler or self._build_default_scheduler()

        self.scaler = GradScaler() if use_fp16 else None

        self.history_train = []
        self.history_val = []

    # ------------------------------------------------------------
    # 🔹 Freeze logic
    # ------------------------------------------------------------

    def _set_requires_grad(self, module, flag):
        for p in module.parameters():
            p.requires_grad = flag

    def _configure_trainable_layers(self):

        # Unfreeze segmentator
        self._set_requires_grad(self.model.patch_embed, True)
        self._set_requires_grad(self.model.segmentator_encoder, True)
        self._set_requires_grad(self.model.segmentator_bottleneck, True)
        self._set_requires_grad(self.model.segmentator_decoder, True)
        self._set_requires_grad(self.model.segmentator_head, True)

        # Freeze upscaler and Cross attention
        self._set_requires_grad(self.model.upscaler_encoder, False)
        self._set_requires_grad(self.model.upscaler_bottleneck, False)
        self._set_requires_grad(self.model.upscaler_decoder, False)
        self._set_requires_grad(self.model.upscaler_head, False)
        self._set_requires_grad(self.model.ca_seg_to_sr, False)
        self._set_requires_grad(self.model.ca_sr_to_seg, False)


    # def _freeze_non_segmentator(self):
    #     modules_to_freeze = [
    #         self.model.ca_seg_to_sr,
    #         self.model.ca_sr_to_seg,
    #         self.model.upscaler_encoder,
    #         self.model.upscaler_bottleneck,
    #         self.model.upscaler_decoder,
    #         self.model.upscaler_head,
    #     ]

    #     for module in modules_to_freeze:
    #         for p in module.parameters():
    #             p.requires_grad = False

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

    # ------------------------------------------------------------
    # 🔹 Scheduler (warmup + cosine)
    # ------------------------------------------------------------

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
    # 🔹 Error matrix check
    # ------------------------------------------------------------

    def _ensure_2ch(self, x):
        # x: [B,1,H,W] -> [B,2,H,W] (diffraction + error_matrix)
        if x.size(1) == 2:
            return x
        err = torch.sqrt(torch.abs(x))
        return torch.cat([x, err], dim=1)

    # ------------------------------------------------------------
    # 🔹 One epoch
    # ------------------------------------------------------------

    def _train_one_epoch(self, epoch):
        self.model.train()
        total_loss = 0.0

        for images, masks in tqdm(
            self.train_loader,
            desc=f"Epoch {epoch+1} Train",
            leave=False
        ):
            images = images.to(self.device)
            images = self._ensure_2ch(images)
            masks = masks.unsqueeze(1).to(self.device)

            self.optimizer.zero_grad()

            if self.use_fp16:
                with autocast():
                    preds, _ = self.model.segment_1(images)
                    loss = self.loss_fn(preds, masks)

                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                preds, _ = self.model.segment_1(images)
                loss = self.loss_fn(preds, masks)

                loss.backward()
                self.optimizer.step()

            total_loss += loss.item() * images.size(0)

        return total_loss / len(self.train_loader.dataset)

    # ------------------------------------------------------------
    # 🔹 Validation
    # ------------------------------------------------------------

    def _validate(self, epoch):
        self.model.eval()
        total_loss = 0.0

        with torch.no_grad():
            for images, masks in tqdm(
                self.val_loader,
                desc=f"Epoch {epoch+1} Val",
                leave=False
            ):
                images = images.to(self.device)
                images = self._ensure_2ch(images)
                masks = masks.unsqueeze(1).to(self.device)

                if self.use_fp16:
                    with autocast():
                        preds, _ = self.model.segment_1(images)
                        loss = self.loss_fn(preds, masks)
                else:
                    preds, _ = self.model.segment_1(images)
                    loss = self.loss_fn(preds, masks)

                total_loss += loss.item() * images.size(0)

        return total_loss / len(self.val_loader.dataset)

    # ------------------------------------------------------------
    # 🔹 Main training loop
    # ------------------------------------------------------------

    def train(self):
        for epoch in range(self.num_epochs):

            train_loss = self._train_one_epoch(epoch)
            val_loss = self._validate(epoch)

            self.scheduler.step()

            self.history_train.append(train_loss)
            self.history_val.append(val_loss)

            if self.verbose:
                print(
                    f"Epoch [{epoch+1}/{self.num_epochs}] "
                    f"Train Loss: {train_loss:.6f} "
                    f"Val Loss: {val_loss:.6f} "
                    f"LR: {self.optimizer.param_groups[0]['lr']:.2e}"
                )

        return {
            "train_loss": self.history_train,
            "val_loss": self.history_val
        }

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
