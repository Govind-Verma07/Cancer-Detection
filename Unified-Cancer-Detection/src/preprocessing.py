import torch
from torchvision import transforms
from PIL import Image
import numpy as np

def get_transforms(img_size=(224, 224), train=True):
    """
    Standard transforms for classification and segmentation input.
    """
    if train:
        return transforms.Compose([
            transforms.Lambda(lambda x: x if isinstance(x, Image.Image) else transforms.ToPILImage()(x)),
            transforms.Resize(img_size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(20),
            transforms.ColorJitter(brightness=0.2, contrast=0.5), # Enhance tumor visibility
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Lambda(lambda x: x if isinstance(x, Image.Image) else transforms.ToPILImage()(x)),
            transforms.Resize(img_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

def preprocess_patch(patch, img_size=(224, 224)):
    """
    Preprocess a single patch for model input.
    """
    t = get_transforms(img_size, train=False)
    return t(patch).unsqueeze(0) # [C, H, W] -> [1, C, H, W]
