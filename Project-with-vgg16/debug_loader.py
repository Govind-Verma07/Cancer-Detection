import os
import sys
import torch
from torch.utils.data import DataLoader

# Add project root to path
sys.path.append(os.getcwd())

from src.dataset import BreastCancerDataset
from src.preprocessing import get_transforms
from utils.config import Config

def debug_loading():
    img_dir = os.path.join("data", "Pixel-level annotation")
    mask_dir = os.path.join("data", "ROI Masks")
    
    transform = get_transforms(img_size=Config.SEG_INPUT_SIZE, train=True)
    dataset = BreastCancerDataset(img_dir, mask_dir=mask_dir, transform=transform)
    
    print(f"Dataset length: {len(dataset)}")
    if len(dataset) > 0:
        print("Attempting to load first sample...")
        try:
            sample = dataset[0]
            print("Sample loaded successfully!")
            print(f"Image shape: {sample['image'].shape}")
            print(f"Mask shape: {sample['mask'].shape}")
        except Exception as e:
            print(f"Error loading sample: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    debug_loading()
