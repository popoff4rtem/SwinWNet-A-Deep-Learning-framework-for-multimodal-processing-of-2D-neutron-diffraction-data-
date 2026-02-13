from Segmentator_pretrain import SegmentatorTrainer
from Upscaler_pretrain import UpscalerTrainer
from FullModel_supervised_trainer import FullModelTrainer

class SwinWNetTrainingPipeline:

    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        device,

        seg_epochs=300,
        sr_epochs=50,
        full_epochs=100,

        seg_lr=2e-4,
        sr_lr=2e-4,
        full_lr=1e-4,

        seg_weight_lr=1.0,
        seg_weight_hr=1.0,
        rec_weight=1.0,
    ):

        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device

        self.seg_epochs = seg_epochs
        self.sr_epochs = sr_epochs
        self.full_epochs = full_epochs

        self.seg_lr = seg_lr
        self.sr_lr = sr_lr
        self.full_lr = full_lr

        self.seg_weight_lr = seg_weight_lr
        self.seg_weight_hr = seg_weight_hr
        self.rec_weight = rec_weight


    # ==========================================
    # Main entry point
    # ==========================================

    def run(self):

        print("\n========== STAGE 1: Segmentator Pretraining ==========\n")
        self._run_segmentator_pretrain()

        print("\n========== STAGE 2: Upscaler Pretraining ==========\n")
        self._run_upscaler_pretrain()

        print("\n========== STAGE 3: Full Model Joint Training ==========\n")
        self._run_full_training()

        print("\n========== TRAINING COMPLETE ==========\n")

    # ==========================================
    # Stage 1
    # ==========================================

    def _run_segmentator_pretrain(self):

        trainer = SegmentatorTrainer(
            model=self.model,
            train_loader=self.train_loader,
            val_loader=self.val_loader,
            device=self.device,
            loss='CombinedLoss',      # DiceLoss, TverskyLoss, FocalTverskyLoss, FocalBCE, CombinedLoss
            optimizer=None,      # if None → create default
            scheduler=None,      # if None → create default
            num_epochs=self.seg_epochs,
            lr=self.seg_lr,
        )


        trainer.train()
        trainer.release_training_state()

    # ==========================================
    # Stage 2
    # ==========================================

    def _run_upscaler_pretrain(self):

        trainer = UpscalerTrainer(
            model=self.model,
            train_loader=self.train_loader,
            val_loader=self.val_loader,
            device=self.device,
            loss='SmoothL1Loss',      # MSELoss, L1Loss, SmoothL1Loss
            optimizer=None,      # if None → create default
            scheduler=None,      # if None → create default
            num_epochs=self.seg_epochs,
            lr=self.sr_lr,
        )

        trainer.train()
        trainer.release_training_state()

    # ==========================================
    # Stage 3
    # ==========================================

    def _run_full_training(self):


        trainer = FullModelTrainer(
            model=self.model,
            train_loader=self.train_loader,
            val_loader=self.val_loader,
            device=self.device,
            segmentator_loss='CombinedLoss',      # DiceLoss, TverskyLoss, FocalTverskyLoss, FocalBCE, CombinedLoss
            upscaler_loss='SmoothL1Loss',         # MSELoss, L1Loss, SmoothL1Loss
            optimizer=None,      # if None → create default
            scheduler=None,      # if None → create default
            num_epochs=self.full_epochs,
            seg_weight_lr=self.seg_weight_lr,
            seg_weight_hr=self.seg_weight_hr,
            rec_weight=self.rec_weight,
            base_lr=self.full_lr
        )

        trainer.train()
        trainer.release_training_state()