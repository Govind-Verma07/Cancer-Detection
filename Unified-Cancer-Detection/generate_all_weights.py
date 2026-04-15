import torch
import os
import sys

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.segmentation import UNet
from resnet50.model import BreastCancerClassifier as ResNetClassifier
from vgg16.model import BreastCancerClassifier as VGGClassifier
from utils.config import Config

def generate_dummy_weights():
    print("Generating dummy weights for Unified Cancer Detection Suite...")
    
    # 1. ResNet50 Models
    os.makedirs(Config.RESNET50_DIR, exist_ok=True)
    
    resnet_seg = UNet()
    torch.save(resnet_seg.state_dict(), Config.RESNET50_SEG_PATH)
    print("[OK] ResNet50 Segmentation -> " + Config.RESNET50_SEG_PATH)
    
    resnet_cls = ResNetClassifier()
    torch.save(resnet_cls.state_dict(), Config.RESNET50_CLS_PATH)
    print("[OK] ResNet50 Classification -> " + Config.RESNET50_CLS_PATH)
    
    # 2. VGG16 Models
    os.makedirs(Config.VGG16_DIR, exist_ok=True)
    
    vgg_seg = UNet()
    torch.save(vgg_seg.state_dict(), Config.VGG16_SEG_PATH)
    print("[OK] VGG16 Segmentation -> " + Config.VGG16_SEG_PATH)
    
    vgg_cls = VGGClassifier()
    torch.save(vgg_cls.state_dict(), Config.VGG16_CLS_PATH)
    print("[OK] VGG16 Classification -> " + Config.VGG16_CLS_PATH)
    
    print("All weights generated successfully!")

if __name__ == "__main__":
    generate_dummy_weights()
