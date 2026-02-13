import torch
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
from ST_Inference_Pipline import SwinWNetInference
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from Diffraction_metrics import DiffractionMetricsCalculator


def binarize_prediction(pred_probs, threshold=0.5):
    """
    pred_probs: torch.Tensor [B,1,H,W], float in [0,1]
    """
    return (pred_probs >= threshold).to(torch.uint8)

def confusion_matrix_binary(pred_bin, gt_bin):
    """
    pred_bin, gt_bin: [B,1,H,W], uint8 {0,1}
    """
    pred = pred_bin.view(-1)
    gt   = gt_bin.view(-1)

    TP = torch.sum((pred == 1) & (gt == 1)).float()
    TN = torch.sum((pred == 0) & (gt == 0)).float()
    FP = torch.sum((pred == 1) & (gt == 0)).float()
    FN = torch.sum((pred == 0) & (gt == 1)).float()

    return TP, TN, FP, FN

def pixel_accuracy(TP, TN, FP, FN, eps=1e-8):
    return (TP + TN) / (TP + TN + FP + FN + eps)

def iou_score(TP, FP, FN, eps=1e-8):
    return TP / (TP + FP + FN + eps)

def dice_score(TP, FP, FN, eps=1e-8):
    return (2 * TP) / (2 * TP + FP + FN + eps)

def precision_score(TP, FP, eps=1e-8):
    return TP / (TP + FP + eps)

def recall_score(TP, FN, eps=1e-8):
    return TP / (TP + FN + eps)

import numpy as np
from scipy.spatial.distance import cdist
from scipy.ndimage import binary_erosion

def extract_boundary(mask):
    """
    mask: [H,W] binary numpy array
    """
    eroded = binary_erosion(mask)
    boundary = mask ^ eroded
    return np.argwhere(boundary)

def compute_all_metrics(pred_probs, gt_mask, threshold=0.5):
    pred_bin = binarize_prediction(pred_probs, threshold)
    gt_bin   = gt_mask.to(torch.uint8)

    TP, TN, FP, FN = confusion_matrix_binary(pred_bin, gt_bin)

    metrics = {
        "PixelAccuracy": pixel_accuracy(TP, TN, FP, FN).item(),
        "IoU": iou_score(TP, FP, FN).item(),
        "Dice": dice_score(TP, FP, FN).item(),
        "Precision": precision_score(TP, FP).item(),
        "Recall": recall_score(TP, FN).item(),
    }
    
    return metrics

def extract_metric(metrics_list, metric_name):
    return np.array([m[metric_name] for m in metrics_list])

def calculate_statistics(data, metric_name):
    """Calculates the mean and standard deviation"""
    mean_val = np.mean(data)
    std_val = np.std(data, ddof=1)  # ddof=1 for an unbiased estimator
    
    print(f"{metric_name}:")
    print(f"  Mean: {mean_val:.4f}")
    print(f"  Std: {std_val:.4f}")
    print(f"  Min: {np.min(data):.4f}")
    print(f"  Max: {np.max(data):.4f}")
    print(f"  Number of samples: {len(data)}")
    print("-" * 40)
    
    return mean_val, std_val

def summarize_errors(errs: np.ndarray):
    if errs.size == 0:
        return {"mean": np.nan, "median": np.nan, "p95": np.nan}
    return {
        "mean": float(np.mean(errs)),
        "median": float(np.median(errs)),
        "p95": float(np.percentile(errs, 95)),
    }

def plot_metric_distributions(all_metrics, bins=None, xlims=None, title="Metrics Distribution"):
    """
    all_metrics = {"integral": [], "peak": [], "shape": []}
    """
    sns.set_theme(style="white")  # без сетки

    metrics = ["integral", "peak", "shape"]
    if bins is None:
        bins = {"integral": 200, "peak": 200, "shape": 150}
    if xlims is None:
        xlims = {"integral": None, "peak": None, "shape": None}

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(title, fontsize=16)

    # Единый цвет для всех графиков
    color = "#1f77b4"  # синий

    for ax, m in zip(axes, metrics):
        data = np.asarray(all_metrics[m])
        data = data[np.isfinite(data)]

        # histogram with KDE
        sns.histplot(data, bins=bins[m], stat="density", kde=True, 
                    color=color, alpha=0.35, ax=ax, label=m)

        ax.set_title(m)
        ax.set_xlabel("Abs Error")
        ax.set_ylabel("Density")
        ax.tick_params(axis="both", which="both", direction="in")
        ax.grid(False)
        
        if xlims.get(m) is not None:
            ax.set_xlim(*xlims[m])

        stats = summarize_errors(data)
        ax.text(
            0.02, 0.98,
            f"mean={stats['mean']:.3g}\n"
            f"med ={stats['median']:.3g}\n"
            f"p95 ={stats['p95']:.3g}",
            transform=ax.transAxes,
            ha="left", va="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8)
        )
        ax.legend()

    plt.tight_layout()
    plt.show()


class MetricsCalculator:
    def __init__(
        self,
        model,
        val_loader,
        device):

        self.model = model.to(device)
        self.val_loader = val_loader
        self.device = device
        self.infer = SwinWNetInference(model, device)

        self.psnr_metric = PeakSignalNoiseRatio(data_range=1.0).to(device)  # data_range=1.0 для диапазона [0, 1]
        self.ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)

        self.d_centers_lr=np.linspace(0.0546658, 7.49180085, 832)
        self.d_centers_hr=np.linspace(0.05318052, 7.49710258, 1241)
        self.physycal_metrics_calculator=DiffractionMetricsCalculator(fixed_centers_pred=self.d_centers_hr,
                                                                      fixed_centers_true=self.d_centers_lr,
                                                                      device=device)

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


    def CalculateSegmentationMetrics(self):
        
        metrics_25_lr = []
        metrics_50_lr = []
        metrics_75_lr = []

        metrics_25_hr = []
        metrics_50_hr = []
        metrics_75_hr = []

        metrics_names = ["PixelAccuracy", "IoU", "Dice", "Precision", "Recall"]

        with torch.no_grad():
            for images, masks in tqdm(
                self.val_loader,
                desc=f"Calculating Segmentation Metrics",
                leave=False
            ):

                images = images.to(self.device)
                masks = masks.unsqueeze(1).to(self.device)

                result = self.infer(images)

                masks_up = F.interpolate(
                    masks.float(),
                    scale_factor=2,
                    mode='nearest-exact'
                ).long()

                for probability_map_lr, probability_map_hr, binary_mask_lr, binary_mask_hr in zip(self.infer.seg_map_lr, self.infer.seg_map_hr, masks, masks_up):

                    probability_map_lr = probability_map_lr.unsqueeze(0)
                    probability_map_hr = probability_map_hr.unsqueeze(0)
                    binary_mask_lr = binary_mask_lr.unsqueeze(0)
                    binary_mask_hr = binary_mask_hr.unsqueeze(0)

                    metrics_25_lr.append(compute_all_metrics(probability_map_lr, binary_mask_lr, threshold=0.25))
                    metrics_50_lr.append(compute_all_metrics(probability_map_lr, binary_mask_lr, threshold=0.5))
                    metrics_75_lr.append(compute_all_metrics(probability_map_lr, binary_mask_lr, threshold=0.75))

                    metrics_25_hr.append(compute_all_metrics(probability_map_hr, binary_mask_hr, threshold=0.25))
                    metrics_50_hr.append(compute_all_metrics(probability_map_hr, binary_mask_hr, threshold=0.5))
                    metrics_75_hr.append(compute_all_metrics(probability_map_hr, binary_mask_hr, threshold=0.75))

        all_metrics = {'Low Res': {'0.25 thrashold': metrics_25_lr, '0.50 thrashold': metrics_50_lr, '0.75 thrashold': metrics_75_lr},
                       'High Res': {'0.25 thrashold': metrics_25_hr, '0.50 thrashold': metrics_50_hr, '0.75 thrashold': metrics_75_hr}}


        print("\n========== Segmentation Metrics Low Res ==========\n")

        for metric_name in metrics_names:
            print(metric_name)
            arr_25 = extract_metric(metrics_25_lr, metric_name)
            mean_25 = np.mean(arr_25)
            std_25 = np.std(arr_25)
            print(f'0.25 thrashold {metric_name}: mean: {mean_25}, std: {std_25}')

            arr_50 = extract_metric(metrics_50_lr, metric_name)
            mean_50 = np.mean(arr_50)
            std_50 = np.std(arr_50)
            print(f'0.50 thrashold {metric_name}: mean: {mean_50}, std: {std_50}')

            arr_75 = extract_metric(metrics_75_lr, metric_name)
            mean_75 = np.mean(arr_75)
            std_75 = np.std(arr_75)
            print(f'0.75 thrashold {metric_name}: mean: {mean_75}, std: {std_75}')

            print('-----------------------------------------------------------------------')


        print("\n========== Segmentation Metrics High Res ==========\n")

        for metric_name in metrics_names:
            print(metric_name)
            arr_25 = extract_metric(metrics_25_lr, metric_name)
            mean_25 = np.mean(arr_25)
            std_25 = np.std(arr_25)
            print(f'0.25 thrashold {metric_name}: mean: {mean_25}, std: {std_25}')

            arr_50 = extract_metric(metrics_50_lr, metric_name)
            mean_50 = np.mean(arr_50)
            std_50 = np.std(arr_50)
            print(f'0.50 thrashold {metric_name}: mean: {mean_50}, std: {std_50}')

            arr_75 = extract_metric(metrics_75_lr, metric_name)
            mean_75 = np.mean(arr_75)
            std_75 = np.std(arr_75)
            print(f'0.75 thrashold {metric_name}: mean: {mean_75}, std: {std_75}')

            print('-----------------------------------------------------------------------')

        
        return all_metrics

    def CalculateUpscalerMetrics(self):

        PSNRs_summary = []
        SSIMs_summary = []
        PSNRs_diffraction = []
        SSIMs_diffraction = []
        PSNRs_error_matrix = []
        SSIMs_error_matrix = []

        with torch.no_grad():
            for images, _ in tqdm(
                self.val_loader,
                desc=f"Calculating Upscaleing Metrics",
                leave=False
            ):

                images = images.to(self.device)
                images = self.ensure_2ch(images)

                seg, skips_seg = self.model.segment_1(images)

                # prepare data for SR
                seg = torch.sigmoid(seg)
                images = images * seg

                images_downscaled = F.interpolate(
                    images, 
                    scale_factor=0.5, 
                    mode='bilinear', 
                    align_corners=False
                )  # размер: [32, 1, 125, 240]

                norm_downscaled, params_downscaled = self.normalize_piecewise(images_downscaled)
                denorm_downscaled = self.denormalize_piecewise(norm_downscaled, params_downscaled)

                norm_images, params_images = self.normalize_piecewise(images)
                denorm_images = self.denormalize_piecewise(norm_images, params_images)

                # SR @ full scale
                sr_out, _ = self.model.upscale(norm_downscaled.to(self.device), skips_seg)
                denorm_sr_out = self.denormalize_piecewise(sr_out, params_images)
                
                for gt_normed_diffraction, pred_normed_diffraction in zip(norm_images, sr_out):

                    gt_normed_diffraction = gt_normed_diffraction.unsqueeze(0).to(self.device)
                    pred_normed_diffraction = pred_normed_diffraction.unsqueeze(0).to(self.device)

                    gt_normed_diffraction = torch.clamp(gt_normed_diffraction, 0, 1)
                    pred_normed_diffraction = torch.clamp(pred_normed_diffraction, 0, 1)

                    psnr_value_summary = self.psnr_metric(gt_normed_diffraction, pred_normed_diffraction) # img_pred, img_gt
                    # ssim_value_summary = ssim_metric(pred_normed_diffraction, gt_normed_diffraction)
                    ssim_value_summary = self.ssim_metric(gt_normed_diffraction, pred_normed_diffraction)

                    # Вычисляем метрики
                    psnr_value_diffraction = self.psnr_metric(gt_normed_diffraction[:, 0:1, :, :], pred_normed_diffraction[:, 0:1, :, :]) # img_pred, img_gt
                    ssim_value_diffraction = self.ssim_metric(gt_normed_diffraction[:, 0:1, :, :], pred_normed_diffraction[:, 0:1, :, :])

                    psnr_value_error_matrix = self.psnr_metric(gt_normed_diffraction[:, 1:2, :, :], pred_normed_diffraction[:, 1:2, :, :]) # img_pred, img_gt
                    ssim_value_error_matrix = self.ssim_metric(gt_normed_diffraction[:, 1:2, :, :], pred_normed_diffraction[:, 1:2, :, :])

                    PSNRs_summary.append(psnr_value_summary.cpu().item())
                    SSIMs_summary.append(ssim_value_summary.cpu().item())

                    PSNRs_diffraction.append(psnr_value_diffraction.cpu().item())
                    SSIMs_diffraction.append(ssim_value_diffraction.cpu().item())

                    PSNRs_error_matrix.append(psnr_value_error_matrix.cpu().item())
                    SSIMs_error_matrix.append(ssim_value_error_matrix.cpu().item())


        all_metrics = {'Summary Metrics': {'PSNR': PSNRs_summary, 'SSIM': SSIMs_summary},
                       'Only Diffraction Metrics': {'PSNR': PSNRs_diffraction, 'SSIM': SSIMs_diffraction},
                       'Only Error Matrix Metrics': {'PSNR': PSNRs_error_matrix, 'SSIM': SSIMs_error_matrix}}


        print("\n========== Upscaling Metrics Summary Diffraction + Error Matrix ==========\n")

        psnr_summary_mean, psnr_summary_std = calculate_statistics(PSNRs_summary, "PSNR (dB)")
        ssim_summary_mean, ssim_summary_std = calculate_statistics(SSIMs_summary, "SSIM")

        print("\n========== Upscaling Metrics Only Diffraction ==========\n")

        psnr_diffraction_mean, psnr_diffraction_std = calculate_statistics(PSNRs_diffraction, "PSNR (dB)")
        ssim_diffraction_mean, ssim_diffraction_std = calculate_statistics(SSIMs_diffraction, "SSIM")

        print("\n========== Upscaling Metrics Only Error Matrix ==========\n")

        psnr_error_matrix_mean, psnr_error_matrix_std = calculate_statistics(PSNRs_error_matrix, "PSNR (dB)")
        ssim_error_matrix_mean, ssim_error_matrix_std = calculate_statistics(SSIMs_error_matrix, "SSIM")


        return all_metrics


    def CalculatePhysycalMetrics(self):

        all_metrics = {"integral": [], "peak": [], "shape": []}

        with torch.no_grad():
            for images, _ in tqdm(
                self.val_loader,
                desc=f"Calculating Physycal Metrics",
                leave=False
            ):

                images = images.to(self.device)
                images = self.ensure_2ch(images)

                seg, skips_seg = self.model.segment_1(images)

                # prepare data for SR
                seg = torch.sigmoid(seg)
                images = images * seg

                images_downscaled = F.interpolate(
                    images, 
                    scale_factor=0.5, 
                    mode='bilinear', 
                    align_corners=False
                )  # размер: [32, 1, 125, 240]

                norm_downscaled, params_downscaled = self.normalize_piecewise(images_downscaled)
                denorm_downscaled = self.denormalize_piecewise(norm_downscaled, params_downscaled)

                norm_images, params_images = self.normalize_piecewise(images)
                denorm_images = self.denormalize_piecewise(norm_images, params_images)

                # SR @ full scale
                sr_out, _ = self.model.upscale(norm_downscaled.to(self.device), skips_seg)
                denorm_sr_out = self.denormalize_piecewise(sr_out, params_images)
                
                for downscaled_diffraction, upscaled_diffraction in zip(images_downscaled, denorm_sr_out):

                    allm = self.physycal_metrics_calculator(
                        batch_pred_2d=upscaled_diffraction.unsqueeze(0),     # predicted 2D diffraction
                        batch_true_2d=downscaled_diffraction.unsqueeze(0),     # ground-truth 2D diffraction
                        peak_params_pred={"scale": True},
                        peak_params_true={"scale": False},
                        tol=0.05
                    )

                    int_per   = np.array(allm['Integral Intensity'])
                    peak_per  = np.array(allm['Peak Intensity'])
                    shape_per = np.array(allm['Shape'])

                    all_metrics["integral"].append(int_per)
                    all_metrics["peak"].append(peak_per)
                    all_metrics["shape"].append(shape_per)

        for k in all_metrics:
            all_metrics[k] = np.concatenate(all_metrics[k], axis=0) if len(all_metrics[k]) else np.array([])

        print("\n========== Metric of divergence in integrated intensity ==========\n")

        calculate_statistics(all_metrics["integral"], "Integral intensity")

        print("\n========== Peak intensity divergence metric ==========\n")

        calculate_statistics(all_metrics["peak"], "Peak intensity")

        print("\n========== Peak shape divergence metric ==========\n")

        calculate_statistics(all_metrics["shape"], "Peak shape")

        plot_metric_distributions(all_metrics)

        
        return all_metrics





