import torch


class SwinWNetInference:

    def __init__(self, model, device):
        self.model = model.to(device)
        self.device = device
        self.model.eval()

        self._reset_outputs()

    # ==========================================
    # Reset all stored attributes
    # ==========================================

    def _reset_outputs(self):

        self.images = None
        self.seg_map_lr = None
        self.images_masked_lr = None
        self.norm = None
        self.upscaled_norm = None
        self.upscaled_denorm = None
        self.seg_map_hr = None
        self.images_masked_hr = None

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

    # ==========================================
    # Forward pipeline
    # ==========================================

    def __call__(self, images):

        self._reset_outputs()

        with torch.no_grad():

            # ----------------------------------
            # 1. Ensure 2 channels
            # ----------------------------------
            images = images.to(self.device)
            images = self.ensure_2ch(images)
            self.images = images

            # ----------------------------------
            # 2. Low-resolution segmentation
            # ----------------------------------
            seg, skips_seg = self.model.segment_1(images)
            seg_map_lr = torch.sigmoid(seg)
            self.seg_map_lr = seg_map_lr

            # ----------------------------------
            # 3. Filtering (LR)
            # ----------------------------------
            images_masked_lr = images * seg_map_lr
            self.images_masked_lr = images_masked_lr

            # ----------------------------------
            # 4. Normalization
            # ----------------------------------
            norm, params = self.normalize_piecewise(
                images_masked_lr
            )
            self.norm  = norm

            # ----------------------------------
            # 5. Upscaling
            # ----------------------------------
            upscaled_norm, skips_sr = self.model.upscale(norm, skips_seg)
            self.upscaled_norm = upscaled_norm

            # ----------------------------------
            # 6. Denormalization
            # ----------------------------------
            upscaled_denorm = self.denormalize_piecewise(
                upscaled_norm, params
            )
            self.upscaled_denorm = upscaled_denorm

            # ----------------------------------
            # 7. High-resolution segmentation
            # ----------------------------------
            seg_high, _ = self.model.segment_2(
                upscaled_denorm, skips_sr
            )
            seg_map_hr = torch.sigmoid(seg_high)
            self.seg_map_hr = seg_map_hr

            # ----------------------------------
            # 8. Final filtering
            # ----------------------------------
            images_masked_hr = upscaled_denorm * seg_map_hr
            self.images_masked_hr = images_masked_hr

        return self.images_masked_hr