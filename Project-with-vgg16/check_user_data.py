import os
import sys
from utils.config import Config
from src.dataset import BreastCancerDataset
from src.preprocessing import get_transforms

def check_data():
    print("--- Data Verification Script ---")
    print(f"IMAGE_DIR: {Config.IMAGE_DIR}")
    print(f"MASK_DIR: {Config.MASK_DIR}")
    
    # Check if directories exist
    if not os.path.exists(Config.IMAGE_DIR):
        print(f"Error: IMAGE_DIR does not exist!")
        return
    if not os.path.exists(Config.MASK_DIR):
        print(f"Error: MASK_DIR does not exist!")
        return
        
    # Check for images
    transform = get_transforms(img_size=Config.SEG_INPUT_SIZE, train=False)
    try:
        dataset = BreastCancerDataset(Config.IMAGE_DIR, mask_dir=Config.MASK_DIR, transform=transform)
        count = len(dataset)
        print(f"Total matching images found: {count}")
        
        if count > 0:
            print(f"Example file: {dataset.image_files[0]}")
            # Try loading one sample
            sample = dataset[0]
            print(f"Successfully loaded a sample. Image shape: {sample['image'].shape}")
            if 'mask' in sample:
                print(f"Successfully loaded a mask. Mask shape: {sample['mask'].shape}")
            else:
                print("Warning: No mask found for the example file.")
        else:
            print("Error: No images found in the IMAGE_DIR!")
            
    except Exception as e:
        print(f"Error during dataset initialization/loading: {e}")

if __name__ == "__main__":
    check_data()
