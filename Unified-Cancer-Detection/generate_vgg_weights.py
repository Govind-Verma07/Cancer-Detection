import torch
import os
import sys

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.vgg_classification import BreastCancerClassifier
from src.segmentation import UNet

def generate_dummy():
    model_dir = "models/vgg16"
    os.makedirs(model_dir, exist_ok=True)
    
    # Classification
    cls_model = BreastCancerClassifier()
    torch.save(cls_model.state_dict(), os.path.join(model_dir, "classification_model.pth"))
    print(f"Generated dummy VGG16 classification weights.")
    
    # Segmentation (ResNet project already has this, but for completeness)
    seg_model = UNet()
    torch.save(seg_model.state_dict(), os.path.join(model_dir, "segmentation_model.pth"))
    print(f"Generated dummy VGG16 segmentation weights.")

if __name__ == "__main__":
    generate_dummy()
