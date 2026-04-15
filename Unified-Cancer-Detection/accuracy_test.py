"""
Automation script for model accuracy testing.
Compares predicted masks with ground truth pixel annotations for at least 100 images.
Calculates accuracy metrics and generates plots.
"""

import os
import sys
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import tifffile
from datetime import datetime

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from resnet50.inference import run_resnet50_inference
from vgg16.inference import run_vgg16_inference
from utils.config import Config

def load_ground_truth_mask(mask_path):
    """
    Load ground truth mask from pixel annotation.
    """
    try:
        arr = tifffile.imread(mask_path)
        if arr.ndim == 3:
            arr = arr[:, :, 0]
        # Convert to binary
        mask = (arr > 127).astype(np.uint8)
        return mask
    except Exception as e:
        print(f"Error loading GT mask {mask_path}: {e}")
        return None

def calculate_metrics(pred_mask, gt_mask):
    """
    Calculate IoU, Dice, Precision, Recall.
    """
    if pred_mask.shape != gt_mask.shape:
        # Resize pred to match gt
        pred_mask = cv2.resize(pred_mask.astype(np.uint8), (gt_mask.shape[1], gt_mask.shape[0]), interpolation=cv2.INTER_NEAREST)
    
    pred_flat = pred_mask.flatten()
    gt_flat = gt_mask.flatten()
    
    intersection = np.sum(pred_flat * gt_flat)
    union = np.sum(pred_flat) + np.sum(gt_flat) - intersection
    
    iou = intersection / union if union > 0 else 0
    
    dice = 2 * intersection / (np.sum(pred_flat) + np.sum(gt_flat)) if (np.sum(pred_flat) + np.sum(gt_flat)) > 0 else 0
    
    # Precision and Recall
    tp = intersection
    fp = np.sum(pred_flat) - tp
    fn = np.sum(gt_flat) - tp
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    
    return {
        'iou': iou,
        'dice': dice,
        'precision': precision,
        'recall': recall
    }

def run_accuracy_test(num_images=100):
    """
    Run accuracy testing on up to num_images that have ground truth.
    """
    tiff_dir = os.path.join(Config.PROJECT_ROOT, "media", "TIFF Images")
    mask_dir = os.path.join(Config.PROJECT_ROOT, "media", "Pixel-level annotation", "Pixel-level annotation")
    
    # Get common filenames
    tiff_files = set(f for f in os.listdir(tiff_dir) if f.lower().endswith(('.tif', '.tiff')))
    mask_files = set(os.listdir(mask_dir))
    
    common_files = sorted(tiff_files & mask_files)
    
    if len(common_files) < num_images:
        print(f"Warning: Only {len(common_files)} images have ground truth. Using all available.")
        num_images = len(common_files)
    
    selected_files = common_files[:num_images]
    
    results = []
    
    print(f"Running accuracy test on {len(selected_files)} images...")
    
    for i, fname in enumerate(selected_files):
        print(f"Processing {i+1}/{len(selected_files)}: {fname}")
        
        tiff_path = os.path.join(tiff_dir, fname)
        mask_path = os.path.join(mask_dir, fname)
        
        # Run inference
        resnet_result = run_resnet50_inference(tiff_path, output_dir="results")
        vgg_result = run_vgg16_inference(tiff_path, output_dir="results")
        
        # Load GT
        gt_mask = load_ground_truth_mask(mask_path)
        if gt_mask is None:
            continue
        
        # Get predicted masks
        pred_resnet = resnet_result.get('binary_mask') if resnet_result else None
        pred_vgg = vgg_result.get('binary_mask') if vgg_result else None
        
        if pred_resnet is None or pred_vgg is None:
            print(f"Skipping {fname}: No prediction masks")
            continue
        
        # Calculate metrics
        resnet_metrics = calculate_metrics(pred_resnet, gt_mask)
        vgg_metrics = calculate_metrics(pred_vgg, gt_mask)
        
        result = {
            'filename': fname,
            'resnet_iou': resnet_metrics['iou'],
            'resnet_dice': resnet_metrics['dice'],
            'resnet_precision': resnet_metrics['precision'],
            'resnet_recall': resnet_metrics['recall'],
            'vgg_iou': vgg_metrics['iou'],
            'vgg_dice': vgg_metrics['dice'],
            'vgg_precision': vgg_metrics['precision'],
            'vgg_recall': vgg_metrics['recall']
        }
        results.append(result)
    
    # Save results
    df = pd.DataFrame(results)
    output_path = os.path.join(Config.RESULTS_DIR, "accuracy_test_results.csv")
    df.to_csv(output_path, index=False)
    
    # Generate plots
    generate_accuracy_plots(df)
    
    print(f"Accuracy test complete. Results saved to {output_path}")
    
    # Print summary
    print("\nSummary Statistics:")
    print(df.describe())

def generate_accuracy_plots(df):
    """
    Generate and save accuracy plots.
    """
    metrics = ['iou', 'dice', 'precision', 'recall']
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    for i, metric in enumerate(metrics):
        resnet_col = f'resnet_{metric}'
        vgg_col = f'vgg_{metric}'
        
        axes[i].hist(df[resnet_col], alpha=0.7, label='ResNet50', bins=20)
        axes[i].hist(df[vgg_col], alpha=0.7, label='VGG16', bins=20)
        axes[i].set_title(f'{metric.upper()} Distribution')
        axes[i].set_xlabel(metric.upper())
        axes[i].set_ylabel('Frequency')
        axes[i].legend()
    
    plt.tight_layout()
    plot_path = os.path.join(Config.RESULTS_DIR, "accuracy_plots.png")
    plt.savefig(plot_path)
    plt.close()
    
    print(f"Plots saved to {plot_path}")

if __name__ == "__main__":
    run_accuracy_test(num_images=100)