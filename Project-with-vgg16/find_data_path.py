import os
import numpy as np
from PIL import Image
from utils.config import Config

def check_datasets():
    print(f"🔍 Checking Image Directory: {Config.IMAGE_DIR}")
    print(f"🔍 Checking Mask Directory: {Config.MASK_DIR}")
    
    if not os.path.exists(Config.IMAGE_DIR):
        print("❌ ERROR: Image directory does not exist.")
        return
    if not os.path.exists(Config.MASK_DIR):
        print("❌ ERROR: Mask directory does not exist.")
        return

    images = [f for f in os.listdir(Config.IMAGE_DIR) if f.lower().endswith(('.tif', '.tiff', '.png', '.jpg'))]
    print(f"📁 Images found: {len(images)}")
    
    missing_masks = 0
    empty_masks = 0
    non_zero_masks = 0
    sample_mask_vals = None
    
    for i, img_name in enumerate(images[:5]): # Sample first 5
        mask_path = os.path.join(Config.MASK_DIR, img_name)
        if not os.path.exists(mask_path):
            missing_masks += 1
            continue
            
        try:
            mask = np.array(Image.open(mask_path).convert('L'))
            max_val = mask.max()
            if max_val == 0:
                empty_masks += 1
            else:
                non_zero_masks += 1
                if sample_mask_vals is None:
                    sample_mask_vals = (mask.min(), mask.max(), np.unique(mask)[:5])
        except Exception as e:
            print(f"❌ Error reading mask {img_name}: {e}")
            
    print(f"📊 Results (from sample of {min(100, len(images))}):")
    print(f"   - Missing masks: {missing_masks}")
    print(f"   - Empty masks (all 0): {empty_masks}")
    print(f"   - Non-zero masks: {non_zero_masks}")
    
    if sample_mask_vals:
        print(f"💡 Sample mask stats: min={sample_mask_vals[0]}, max={sample_mask_vals[1]}, unique={sample_mask_vals[2]}")
        
    if missing_masks > 50:
        print("⚠️ WARNING: Most images have no matching masks. Check filenames/extensions.")
    elif empty_masks > non_zero_masks:
        print("⚠️ WARNING: Most masks are empty. Is this expected?")

if __name__ == "__main__":
    check_datasets()
