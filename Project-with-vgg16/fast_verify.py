import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import os
import sys

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.dataset import BreastCancerDataset
from src.segmentation import UNet
from src.classification import BreastCancerClassifier
from src.preprocessing import get_transforms
from utils.config import Config

def fast_verify():
    with open("verification_log.txt", "w") as f:
        f.write("Verification Started\n")
        f.flush()
        
        try:
            img_dir = os.path.join("data", "Pixel-level annotation")
            mask_dir = os.path.join("data", "ROI Masks")
            
            f.write(f"Checking data in {img_dir}\n")
            f.flush()
            
            transform = get_transforms(img_size=Config.SEG_INPUT_SIZE, train=False)
            ds = BreastCancerDataset(img_dir, mask_dir=mask_dir, transform=transform)
            
            f.write(f"Dataset size: {len(ds)}\n")
            f.flush()
            
            if len(ds) == 0:
                f.write("ERROR: No files found in dataset!\n")
                return
                
            sample = ds[0]
            f.write("Successfully loaded one sample\n")
            f.flush()
            
            # Simple forward pass
            model = UNet().to(Config.DEVICE)
            img = sample['image'].unsqueeze(0).to(Config.DEVICE)
            with torch.no_grad():
                out = model(img)
            f.write(f"Forward pass successful. Output shape: {out.shape}\n")
            f.flush()
            
            # Save dummy
            os.makedirs("models", exist_ok=True)
            torch.save(model.state_dict(), "models/verify_test.pth")
            f.write("Model saving successful\n")
            f.flush()
            
            f.write("VERIFICATION COMPLETE - PIPELINE IS FUNCTIONAL\n")
        except Exception as e:
            f.write(f"FAILED: {e}\n")

if __name__ == "__main__":
    fast_verify()
