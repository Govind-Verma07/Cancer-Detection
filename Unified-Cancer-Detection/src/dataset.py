import os
import cv2
import torch
from torch.utils.data import Dataset
import numpy as np
from utils.config import Config

try:
    import tifffile
    HAS_TIFFFILE = True
except ImportError:
    HAS_TIFFFILE = False

class BreastCancerDataset(Dataset):
    """
    Dataset class for Breast Cancer images (Segmentation and Classification).
    Expects data/raw to have images and optionally masks/labels.
    """
    def __init__(self, image_dir, mask_dir=None, labels_dict=None, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.labels_dict = labels_dict # {image_name: {'class': 0, 'stage': 1}}
        self.transform = transform
        self.image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg', '.tif', '.tiff'))]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.image_dir, img_name)
        
        try:
            from PIL import Image
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Fallback or returning a zero tensor could be done here
            # For now, let's assume images are mostly readable
            image = Image.new('RGB', (Config.SEG_INPUT_SIZE[0], Config.SEG_INPUT_SIZE[1]))

        # Load mask if available
        mask = None
        if self.mask_dir:
            mask_path = os.path.join(self.mask_dir, img_name)
            if os.path.exists(mask_path):
                try:
                    if HAS_TIFFFILE:
                        m_data = tifffile.imread(mask_path)
                        m_pil = Image.fromarray(m_data).convert('L').resize(Config.SEG_INPUT_SIZE, Image.NEAREST)
                    else:
                        m_pil = Image.open(mask_path).convert('L').resize(Config.SEG_INPUT_SIZE, Image.NEAREST)
                    mask = (np.array(m_pil) > 127).astype(np.float32)
                except Exception as e:
                    print(f"Error loading mask {mask_path}: {e}")
                    mask = np.zeros((Config.SEG_INPUT_SIZE[1], Config.SEG_INPUT_SIZE[0]), dtype=np.float32)
        
        # Load labels if available
        label_class = -1
        label_stage = -1
        if self.labels_dict and img_name in self.labels_dict:
            label_class = self.labels_dict[img_name].get('class', 0)
            label_stage = self.labels_dict[img_name].get('stage', 0)
            
        if self.transform:
            image = self.transform(image)
            if mask is not None:
                mask = torch.tensor(mask).unsqueeze(0)
            else:
                # If no mask, provide a blank one to keep batch shape consistent
                mask = torch.zeros((1, Config.SEG_INPUT_SIZE[1], Config.SEG_INPUT_SIZE[0]), dtype=torch.float32)
                
        sample = {'image': image}
        if mask is not None: sample['mask'] = mask
        if label_class != -1: sample['label_class'] = torch.tensor(label_class)
        if label_stage != -1: sample['label_stage'] = torch.tensor(label_stage)
        
        return sample
