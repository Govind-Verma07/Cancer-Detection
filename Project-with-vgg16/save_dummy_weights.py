import torch
import os
import sys

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.segmentation import UNet
from src.classification import BreastCancerClassifier
from utils.config import Config

def save_dummy_weights():
    os.makedirs(Config.MODEL_DIR, exist_ok=True)
    
    # Segmentation Model
    seg_model = UNet()
    torch.save(seg_model.state_dict(), Config.SEGMENTATION_MODEL_PATH)
    print(f"Saved dummy segmentation weights to {Config.SEGMENTATION_MODEL_PATH}")
    
    # Classification Model
    cls_model = BreastCancerClassifier()
    torch.save(cls_model.state_dict(), Config.CLASSIFICATION_MODEL_PATH)
    print(f"Saved dummy classification weights to {Config.CLASSIFICATION_MODEL_PATH}")

if __name__ == "__main__":
    save_dummy_weights()
