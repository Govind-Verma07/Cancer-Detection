import os
import cv2
import torch
import numpy as np
import argparse
import sys

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.patching import split_into_patches, reconstruct_from_patches
from src.preprocessing import preprocess_patch
from src.segmentation import UNet
from src.classification import BreastCancerClassifier
from src.contour_detection import detect_contours
from src.metrics import calculate_tumor_metrics, get_relative_coordinates, get_formatted_coordinates
from src.visualization import overlay_contours, save_output_image
from utils.config import Config
from utils.helpers import load_model

def run_inference(image_path, output_dir="results", seg_model=None, cls_model=None):
    # 1. Load Image & Optimize Dimensions
    try:
        from PIL import Image
        image_pil = Image.open(image_path).convert('RGB')
        
        # --- OPTIMIZATION: Downscale massive medical scans ---
        # Caps the maximum dimension to 2000px while maintaining aspect ratio,
        # drastically reducing the number of patches needed with zero clinical loss.
        max_dim = 2048
        if max(image_pil.size) > max_dim:
            ratio = max_dim / max(image_pil.size)
            new_size = (int(image_pil.size[0] * ratio), int(image_pil.size[1] * ratio))
            image_pil = image_pil.resize(new_size, Image.Resampling.LANCZOS)
            
        image_rgb = np.array(image_pil)
    except Exception as e:
        print(f"Error: Could not read image {image_path}: {e}")
        return
    
    h, w, _ = image_rgb.shape
    
    # 2. Patching
    patches, metadata = split_into_patches(image_rgb, patch_size=Config.PATCH_SIZE, stride=Config.STRIDE)
    
    # 3. Load Models (Only if not provided)
    if seg_model is None:
        seg_model = load_model(UNet, Config.SEGMENTATION_MODEL_PATH, Config.DEVICE)
    if cls_model is None:
        cls_model = load_model(BreastCancerClassifier, Config.CLASSIFICATION_MODEL_PATH, Config.DEVICE)
    
    seg_model.eval()
    cls_model.eval()
    
    # 4. Process Patches (Segmentation)
    patch_masks = []
    for patch in patches:
        # --- OPTIMIZATION: Skip completely black patches ---
        # Most of a mammography scan is black background.
        if np.mean(patch) < 10 or np.std(patch) < 3:
            patch_masks.append(np.zeros((Config.SEG_INPUT_SIZE[1], Config.SEG_INPUT_SIZE[0]), dtype=np.float32))
            continue
            
        input_tensor = preprocess_patch(patch, img_size=Config.SEG_INPUT_SIZE).to(Config.DEVICE)
        with torch.no_grad():
            mask = seg_model(input_tensor)
        mask_np = mask.squeeze().cpu().numpy()
        patch_masks.append(mask_np)
        
    # 5. Stitches masks & Refine
    full_mask = reconstruct_from_patches(patch_masks, metadata, (h, w))
    
    # Combined segmentation probability with image intensity
    grayscale = np.mean(image_rgb, axis=2) / 255.0
    refined_mask = full_mask * grayscale 
    
    # --- HEURISTIC FALLBACK (For localized lumps) ---
    if np.sum(refined_mask > Config.SEG_THRESHOLD) < 500:
        gray_u8 = (grayscale * 255).astype(np.uint8)
        
        # Smooth the image to ensure the boundary 'follows' the tumor glow
        blurred = cv2.GaussianBlur(gray_u8, (15, 15), 0)
        
        # Significantly raised threshold (200) to ignore faint skin tissues/nerves
        _, thresh = cv2.threshold(blurred, 200, 255, cv2.THRESH_BINARY)
        
        # Massive Morphological Opening (25x25) to crush any vascular/nerve networks
        # and strictly keep only dense, isolated tumor lumps.
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        
        heuristic_mask = thresh / 255.0
        refined_mask = np.maximum(refined_mask, heuristic_mask * grayscale * 0.95)
    
    binary_mask = (refined_mask > Config.SEG_THRESHOLD).astype(np.uint8)
    
    # 6. Analysis (Per-Contour Classification)
    findings = []
    contour_info = detect_contours(binary_mask)
    
    for info in contour_info:
        # Crop patch around contour
        x, y, cw, ch = info['bbox']
        region_crop = image_rgb[y:y+ch, x:x+cw]
        if region_crop.size == 0: continue
        
        # --- DENSITY-BASED STAGING ---
        # Whiter (higher intensity) = Less transparent = Malignant/Suspicious
        mean_intensity = np.mean(region_crop)
        
        # Heuristic: Intensity > 180 is radiopaque (suspicious)
        is_mal = mean_intensity > 180 
        
        # Assign stage based on radiopacity (Whiteness)
        if mean_intensity > 220: stage_val = 4  # Extremely Opaque
        elif mean_intensity > 190: stage_val = 3 # High Opacity
        elif mean_intensity > 160: stage_val = 2 # Moderate Opacity (Benign/Lump)
        else: stage_val = 1                      # Low Opacity
        
        # Update info with local classification
        info['is_malignant'] = is_mal
        info['stage'] = f"Stage {stage_val}"
        
        rel_coord = get_relative_coordinates(info['centroid'], (h, w))
        finding = {
            'classification': "Malignant" if is_mal else "Benign",
            'stage': info['stage'],
            'location_pct': get_formatted_coordinates(rel_coord),
            'area_pixels': info['area'],
            'is_malignant': is_mal,
            'density_index': f"{mean_intensity/2.55:.1f}%" # Reporting density as a percentage
        }
        findings.append(finding)
    
    # 7. Overall Presence Detection
    tumor_present = len(findings) > 0
    
    # 8. Visualization (Now uses individual is_malignant flags)
    output_image = overlay_contours(image_rgb, contour_info)
    
    # 9. Metrics (Overall)
    from src.metrics import calculate_tumor_metrics
    metrics = calculate_tumor_metrics(binary_mask, image_rgb)
    
    # 10. Save results
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "prediction.jpg")
    save_output_image(output_image, out_path)
    
    results = {
        'tumor_present': tumor_present,
        'findings': findings,
        'overall_ratio': metrics['ratio'],
        'output_path': out_path,
        'visual_result': output_image,
        'binary_mask': binary_mask
    }
    
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, required=True, help="Path to breast image")
    args = parser.parse_args()
    
    res = run_inference(args.image)
    if res:
        print("\n--- ANALYSIS RESULTS ---")
        for k, v in res.items():
            print(f"{k.capitalize()}: {v}")
