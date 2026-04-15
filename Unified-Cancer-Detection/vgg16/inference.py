import os
import cv2
import torch
import numpy as np
import sys

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.patching import split_into_patches, reconstruct_from_patches
from src.preprocessing import preprocess_patch
from src.segmentation import UNet
from vgg16.model import BreastCancerClassifier
from src.contour_detection import detect_contours
from src.metrics import calculate_tumor_metrics, get_relative_coordinates, get_formatted_coordinates
from src.visualization import overlay_contours, save_output_image
from utils.config import Config
from utils.helpers import load_model

def run_vgg16_inference(image_path, output_dir="results", seg_model=None, cls_model=None):
    """
    Complete inference pipeline using VGG16 as the classification backbone.
    Parity with original VGG16 standalone implementation.
    """
    # 1. Load Image
    try:
        from PIL import Image
        image_pil = Image.open(image_path).convert('RGB')
        # Maintain aspect ratio caps at 2048px for optimization
        max_dim = 2048
        if max(image_pil.size) > max_dim:
            ratio = max_dim / max(image_pil.size)
            new_size = (int(image_pil.size[0] * ratio), int(image_pil.size[1] * ratio))
            image_pil = image_pil.resize(new_size, Image.Resampling.LANCZOS)
        image_rgb = np.array(image_pil)
    except Exception as e:
        print(f"Error reading image {image_path}: {e}")
        return None
    
    h, w, _ = image_rgb.shape
    
    # 2. Patching
    patches, metadata = split_into_patches(image_rgb, patch_size=Config.PATCH_SIZE, stride=Config.STRIDE)
    
    # 3. Models
    if seg_model is None:
        seg_model_path = os.path.join(Config.VGG16_DIR, "vgg16_seg.pth")
        seg_model = load_model(UNet, seg_model_path, Config.DEVICE)
    if cls_model is None:
        cls_model_path = os.path.join(Config.VGG16_DIR, "vgg16_cls.pth")
        cls_model = load_model(BreastCancerClassifier, cls_model_path, Config.DEVICE)
    
    seg_model.eval()
    cls_model.eval()
    
    # 4. Segmentation
    patch_masks = []
    for patch in patches:
        if np.mean(patch) < 10 or np.std(patch) < 3: # Skip background
            patch_masks.append(np.zeros((Config.SEG_INPUT_SIZE[1], Config.SEG_INPUT_SIZE[0]), dtype=np.float32))
            continue
        input_tensor = preprocess_patch(patch, img_size=Config.SEG_INPUT_SIZE).to(Config.DEVICE)
        with torch.no_grad():
            mask = seg_model(input_tensor)
        patch_masks.append(mask.squeeze().cpu().numpy())
        
    full_mask = reconstruct_from_patches(patch_masks, metadata, (h, w))
    
    # Refine with grayscale intensity
    grayscale = np.mean(image_rgb, axis=2) / 255.0
    refined_mask = full_mask * grayscale
    
    # Heuristic fallback if mask is empty
    if np.sum(refined_mask > Config.SEG_THRESHOLD) < 1000:
        # Simple intensity-based fallback
        binary_mask = (grayscale > 0.8).astype(np.uint8)
    else:
        binary_mask = (refined_mask > Config.SEG_THRESHOLD).astype(np.uint8)
    
    # 5. Classification & Staging
    findings = []
    contour_info = detect_contours(binary_mask)
    
    for info in contour_info:
        x, y, cw, ch = info['bbox']
        region_crop = image_rgb[y:y+ch, x:x+cw]
        if region_crop.size == 0: continue
        
        # Classification via Model
        input_tensor = preprocess_patch(cv2.resize(region_crop, (224, 224)), img_size=(224, 224)).to(Config.DEVICE)
        with torch.no_grad():
            binary_out, stage_out = cls_model(input_tensor)
            is_mal = torch.argmax(binary_out, dim=1).item() == 1
            stage_val = torch.argmax(stage_out, dim=1).item() + 1
            
        rel_coord = get_relative_coordinates(info['centroid'], (h, w))
        findings.append({
            'classification': "Malignant" if is_mal else "Benign",
            'stage': f"Stage {stage_val}",
            'location_pct': get_formatted_coordinates(rel_coord),
            'area_pixels': info['area'],
            'is_malignant': is_mal
        })
        # Add to info for visualization
        info['is_malignant'] = is_mal
    
    # 6. Visualization
    output_image = overlay_contours(image_rgb, contour_info)
    out_path = os.path.join(output_dir, "prediction_vgg16.jpg")
    save_output_image(output_image, out_path)
    
    # Overall Ratio
    metrics = calculate_tumor_metrics(binary_mask, image_rgb)
    
    return {
        'tumor_present': len(findings) > 0,
        'findings': findings,
        'overall_ratio': metrics['ratio'],
        'output_path': out_path,
        'binary_mask': binary_mask  # Add binary mask for accuracy testing
    }
