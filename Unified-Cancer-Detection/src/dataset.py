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
    def __init__(self, image_dir, pixel_mask_dir=None, roi_mask_dir=None, labels_dict=None, transform=None):
        self.image_dir = image_dir
        self.pixel_mask_dir = pixel_mask_dir
        self.roi_mask_dir = roi_mask_dir
        self.labels_dict = labels_dict # {image_name: {'class': 0, 'stage': 1}}
        self.transform = transform
        
        # Only use images that have a pixel mask
        if self.pixel_mask_dir and os.path.exists(self.pixel_mask_dir):
            valid_files = os.listdir(self.pixel_mask_dir)
        else:
            valid_files = os.listdir(image_dir)
            
        self.image_files = [f for f in valid_files if f.lower().endswith(('.jpg', '.png', '.jpeg', '.tif', '.tiff'))]

    def __len__(self):
        return len(self.image_files)

    def _load_mask(self, mask_dir, img_name):
        mask_path = os.path.join(mask_dir, img_name)
        if os.path.exists(mask_path):
            try:
                from PIL import Image
                if HAS_TIFFFILE:
                    m_data = tifffile.imread(mask_path)
                    m_pil = Image.fromarray(m_data).convert('L').resize(Config.SEG_INPUT_SIZE, Image.NEAREST)
                else:
                    m_pil = Image.open(mask_path).convert('L').resize(Config.SEG_INPUT_SIZE, Image.NEAREST)
                m_arr = np.array(m_pil)
                m_thresh = 127 if m_arr.max() > 1 else 0.5
                return (m_arr > m_thresh).astype(np.float32)
            except Exception as e:
                print("[WARN] Error loading mask from " + str(mask_dir) + " for " + str(img_name) + ": " + str(e))
        return np.zeros((Config.SEG_INPUT_SIZE[1], Config.SEG_INPUT_SIZE[0]), dtype=np.float32)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.image_dir, img_name)
        
        try:
            from PIL import Image
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            try:
                # TIF files sometimes have compression profiles PIL cannot read natively.
                # Fallback to cv2, then tifffile.
                img_arr = cv2.imread(img_path)
                if img_arr is not None:
                    from PIL import Image
                    image = Image.fromarray(cv2.cvtColor(img_arr, cv2.COLOR_BGR2RGB))
                elif HAS_TIFFFILE:
                    import tifffile
                    from PIL import Image
                    img_arr = tifffile.imread(img_path)
                    image = Image.fromarray(img_arr).convert('RGB')
                else:
                    raise ValueError("cv2 returned None and tifffile is unavailable.")
            except Exception as e2:
                # Silently catch and substitute with blank canvas for corrupted files
                from PIL import Image
                image = Image.new('RGB', (Config.SEG_INPUT_SIZE[0], Config.SEG_INPUT_SIZE[1]))

        pixel_mask = self._load_mask(self.pixel_mask_dir, img_name) if self.pixel_mask_dir else np.zeros((Config.SEG_INPUT_SIZE[1], Config.SEG_INPUT_SIZE[0]), dtype=np.float32)
        roi_mask = self._load_mask(self.roi_mask_dir, img_name) if self.roi_mask_dir else np.zeros((Config.SEG_INPUT_SIZE[1], Config.SEG_INPUT_SIZE[0]), dtype=np.float32)

        label_class = -1
        label_stage = -1
        if self.labels_dict and img_name in self.labels_dict:
            label_class = self.labels_dict[img_name].get('class', 0)
            label_stage = self.labels_dict[img_name].get('stage', 0)
            
        if self.transform:
            image = self.transform(image)
            pixel_mask = torch.tensor(pixel_mask).unsqueeze(0)
            roi_mask = torch.tensor(roi_mask).unsqueeze(0)
        else:
            pixel_mask = torch.tensor(pixel_mask).unsqueeze(0)
            roi_mask = torch.tensor(roi_mask).unsqueeze(0)
                
        sample = {
            'image': image,
            'pixel_mask': pixel_mask,
            'roi_mask': roi_mask
        }
        if label_class != -1: sample['label_class'] = torch.tensor(label_class)
        if label_stage != -1: sample['label_stage'] = torch.tensor(label_stage)
        
        return sample
