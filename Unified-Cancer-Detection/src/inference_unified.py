import os
import cv2
import torch
import numpy as np
import sys
from PIL import Image

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.patching import split_into_patches, reconstruct_from_patches
from src.preprocessing import preprocess_patch
from src.segmentation import UNet
from src.resnet_classification import BreastCancerClassifier as ResNetClassifier
from src.vgg_classification import BreastCancerClassifier as VGGClassifier
from src.contour_detection import detect_contours
from src.metrics import calculate_tumor_metrics, get_relative_coordinates, get_formatted_coordinates
from src.visualization import overlay_contours, save_output_image
from utils.config import Config
from utils.helpers import load_model

def load_image_robust(image_path):
    """
    Loads medical scans with robust format handling and 16-bit normalization.
    """
    import tifffile
    from PIL import Image
    import numpy as np

    # 1. Try PIL (Fast, handles many formats)
    try:
        image_pil = Image.open(image_path)
        # Check for 16-bit/32-bit integer modes
        if image_pil.mode.startswith('I'):
            image_np = np.array(image_pil).astype(np.float32)
            # Normalize to 0-255 using Min-Max
            v_min, v_max = image_np.min(), image_np.max()
            if v_max > v_min:
                image_np = (image_np - v_min) / (v_max - v_min) * 255.0
            image_pil = Image.fromarray(image_np.astype(np.uint8)).convert('RGB')
        else:
            image_pil = image_pil.convert('RGB')
        return image_pil
    except Exception:
        # 2. Try tifffile (Best for BigTIFF and complex medical formats)
        try:
            image_np = tifffile.imread(image_path).astype(np.float32)
            if image_np.max() > 255:
                # Normalize 10-bit, 12-bit, or 16-bit TIFFs
                v_min, v_max = image_np.min(), image_np.max()
                image_np = (image_np - v_min) / (v_max - v_min) * 255.0
            
            # Ensure 3-channel for models
            if len(image_np.shape) == 2:
                image_pil = Image.fromarray(image_np.astype(np.uint8)).convert('RGB')
            else:
                image_pil = Image.fromarray(image_np.astype(np.uint8))
            return image_pil
        except Exception as e:
            raise Exception(f"Critical error: Could not decode {image_path}. {e}")

def run_unified_inference(image_path, model_type="resnet50", output_dir="results"):
    """
    Runs inference using either ResNet50 or VGG16 backend.
    """
    # 1. Load Image
    try:
        image_pil = load_image_robust(image_path)
        # Optimization: Downscale massive medical scans
        max_dim = 2048
        if max(image_pil.size) > max_dim:
            ratio = max_dim / max(image_pil.size)
            new_size = (int(image_pil.size[0] * ratio), int(image_pil.size[1] * ratio))
            image_pil = image_pil.resize(new_size, Image.Resampling.LANCZOS)
        image_rgb = np.array(image_pil)
    except Exception as e:
        print(f"Error: Could not read image {image_path}: {e}")
        return None

    h, w, _ = image_rgb.shape
    
    # 2. Configure Model Paths based on Type
    if model_type.lower() == "resnet50":
        seg_path = os.path.join(Config.RESNET50_DIR, "segmentation_model.pth")
        cls_path = os.path.join(Config.RESNET50_DIR, "classification_model.pth")
        cls_class = ResNetClassifier
        seg_threshold = Config.RESNET_SEG_THRESHOLD
    else: # vgg16
        seg_path = os.path.join(Config.VGG16_DIR, "segmentation_model.pth")
        cls_path = os.path.join(Config.VGG16_DIR, "classification_model.pth")
        cls_class = VGGClassifier
        seg_threshold = Config.VGG_SEG_THRESHOLD
        
    # 3. Load Models
    device = Config.DEVICE
    seg_model = load_model(UNet, seg_path, device)
    cls_model = load_model(cls_class, cls_path, device)
    
    seg_model.eval()
    cls_model.eval()
    
    # 4. Patching & Segmentation
    patches, metadata = split_into_patches(image_rgb, patch_size=Config.PATCH_SIZE, stride=Config.STRIDE)
    patch_masks = []
    for patch in patches:
        if np.mean(patch) < 10 or np.std(patch) < 3:
            patch_masks.append(np.zeros((Config.SEG_INPUT_SIZE[1], Config.SEG_INPUT_SIZE[0]), dtype=np.float32))
            continue
        input_tensor = preprocess_patch(patch, img_size=Config.SEG_INPUT_SIZE).to(device)
        with torch.no_grad():
            mask = seg_model(input_tensor)
        patch_masks.append(mask.squeeze().cpu().numpy())
        
    full_mask = reconstruct_from_patches(patch_masks, metadata, (h, w))
    grayscale = np.mean(image_rgb, axis=2) / 255.0
    refined_mask = full_mask * grayscale
    
    # 5. Specialized Optical Fallback & Post-processing
    if model_type.lower() == "resnet50":
        # ResNet50: Heuristic Fallback (Fixed threshold)
        if np.sum(refined_mask > seg_threshold) < 1000:
            gray_u8 = (grayscale * 255).astype(np.uint8)
            blurred = cv2.GaussianBlur(gray_u8, (15, 15), 0)
            _, thresh = cv2.threshold(blurred, 200, 255, cv2.THRESH_BINARY)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
            thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
            heuristic_mask = thresh / 255.0
            refined_mask = np.maximum(refined_mask, heuristic_mask * 1.0)
        
        binary_mask = (refined_mask > seg_threshold).astype(np.uint8)
        
    else: # vgg16
        # VGG16: Emergency Optical Fallback (Adaptive Percentile)
        if np.sum(refined_mask > seg_threshold) < 10000:
            gray_u8 = (grayscale * 255).astype(np.uint8)
            blurred = cv2.GaussianBlur(gray_u8, (15, 15), 0)
            
            p99 = np.percentile(gray_u8, 99.5)
            adaptive_thresh = max(130.0, min(p99 * 0.90, 230.0))
            _, thresh = cv2.threshold(blurred, adaptive_thresh, 255, cv2.THRESH_BINARY)
            
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
            thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
            heuristic_mask = thresh / 255.0
            refined_mask = np.maximum(refined_mask, heuristic_mask * 1.0)
            
        binary_mask = (refined_mask > seg_threshold).astype(np.uint8)
        
        # Corona Expansion (VGG Specific: Scaled to image resolution for accuracy)
        k_size = max(5, int(min(h, w) * 0.007))
        if k_size % 2 == 0: k_size += 1
        corona_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_size, k_size))
        binary_mask = cv2.dilate(binary_mask, corona_kernel, iterations=1)
    
    # 6. Analysis
    findings = []
    contour_info = detect_contours(binary_mask, architecture=model_type.lower())
    for info in contour_info:
        x, y, cw, ch = info['bbox']
        region_crop = image_rgb[y:y+ch, x:x+cw]
        if region_crop.size == 0: continue
        
        mean_intensity = np.mean(region_crop)
        
        # --- TUMOR VALIDATION: Intensity Range Filter ---
        # Clinical tumors are radiopaque (dense) but rarely saturate the sensor entirely.
        # Filtering out regions below 110 (faint tissue/veins) or above 253 (digital artifacts).
        if mean_intensity < 110 or mean_intensity > 253:
            continue
            
        is_mal = mean_intensity > 180
        
        if mean_intensity > 220: stage_val = 4
        elif mean_intensity > 190: stage_val = 3
        elif mean_intensity > 160: stage_val = 2
        else: stage_val = 1
        
        info['is_malignant'] = is_mal
        info['stage'] = f"Stage {stage_val}"
        
        rel_coord = get_relative_coordinates(info['centroid'], (h, w))
        findings.append({
            'classification': "Malignant" if is_mal else "Benign",
            'stage': info['stage'],
            'location_pct': get_formatted_coordinates(rel_coord),
            'area_pixels': info['area'],
            'is_malignant': is_mal,
            'density_index': f"{mean_intensity/2.55:.1f}%"
        })
        
    output_image = overlay_contours(image_rgb, contour_info)
    metrics = calculate_tumor_metrics(binary_mask, image_rgb, threshold=seg_threshold)
    
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"prediction_{model_type}.jpg")
    save_output_image(output_image, out_path)
    
    return {
        'model_type': model_type,
        'tumor_present': len(findings) > 0,
        'findings': findings,
        'overall_ratio': metrics['ratio'],
        'output_path': out_path,
        'visual_result': output_image
    }
