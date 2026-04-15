import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import os
import sys

# Add project root to path to access common src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.dataset import BreastCancerDataset
from src.segmentation import UNet
from resnet50.model import BreastCancerClassifier
from src.preprocessing import get_transforms
from utils.config import Config
from utils.helpers import save_model

def train_resnet50_pipeline(epochs=10, batch_size=2):
    device = Config.DEVICE
    print(f"RESNET50 TRAINING START (Device: {device})")
    
    # Paths (Unified via Config)
    img_dir = Config.DATA_RAW
    pixel_mask_dir = os.path.join(Config.PROJECT_ROOT, "data", "pixel_masks")
    roi_mask_dir = os.path.join(Config.PROJECT_ROOT, "data", "roi_masks")
    
    # 1. Dataset & Transforms
    transform = get_transforms(img_size=Config.SEG_INPUT_SIZE, train=True)
    full_dataset = BreastCancerDataset(img_dir, pixel_mask_dir=pixel_mask_dir, roi_mask_dir=roi_mask_dir, transform=transform)
    print(f"Total images found: {len(full_dataset)}")
    
    if len(full_dataset) == 0:
        print("ERROR: No images found! Check paths and extensions.")
        return
    
    # Selection of fresh images
    fresh_indices = list(range(len(full_dataset)))
    if len(fresh_indices) > 50:
        fresh_indices = fresh_indices[:50]
    
    fresh_dataset = Subset(full_dataset, fresh_indices)
    
    total_samples = len(fresh_dataset)
    num_train = int(total_samples * 0.8)
    num_test = total_samples - num_train
    
    # Split
    generator = torch.Generator().manual_seed(42)
    train_ds, test_ds = torch.utils.data.random_split(fresh_dataset, [num_train, num_test], generator=generator)
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False)
    
    # 2. Initialize Models
    print("Initializing ResNet50 Model Suite...")
    seg_model = UNet().to(device)
    cls_model = BreastCancerClassifier().to(device)
    
    # 3. Optimization
    seg_optimizer = optim.Adam(seg_model.parameters(), lr=1e-4)
    cls_optimizer = optim.Adam(cls_model.parameters(), lr=1e-4)
    criterion_seg = nn.BCELoss()
    criterion_cls = nn.CrossEntropyLoss()
    
    # 4. Training Loop
    print("Starting ResNet50 Training...")
    for epoch in range(epochs):
        seg_model.train()
        cls_model.train()
        epoch_loss = 0
        
        for b_idx, batch in enumerate(train_loader):
            images = batch['image'].to(device)
            pixel_masks = batch['pixel_mask'].to(device)
            
            # Step A: Segmentation
            seg_optimizer.zero_grad()
            pred_masks = seg_model(images)
            loss_seg = criterion_seg(pred_masks, pixel_masks)
            loss_seg.backward()
            seg_optimizer.step()
            
            # Step B: Classification
            cls_optimizer.zero_grad()
            labels_class = batch.get('label_class', torch.zeros(images.size(0), dtype=torch.long)).to(device)
            labels_stage = batch.get('label_stage', torch.zeros(images.size(0), dtype=torch.long)).to(device)
            
            pred_class, pred_stage = cls_model(images)
            loss_class = criterion_cls(pred_class, labels_class)
            loss_stage = criterion_cls(pred_stage, labels_stage)
            loss_cls = loss_class + loss_stage
            loss_cls.backward()
            cls_optimizer.step()
            
            b_loss = loss_seg.item() + loss_cls.item()
            epoch_loss += b_loss
            print(f"\r   Batch [{b_idx+1}/{len(train_loader)}] Loss: {b_loss:.4f}", end="", flush=True)
            
        print(f"\nEpoch [{epoch+1}/{epochs}] - Avg Loss: {epoch_loss/len(train_loader):.4f}")
        
    # 5. Save models
    os.makedirs(Config.RESNET50_DIR, exist_ok=True)
    save_model(seg_model, os.path.join(Config.RESNET50_DIR, "resnet50_seg.pth"))
    save_model(cls_model, os.path.join(Config.RESNET50_DIR, "resnet50_cls.pth"))
    print("ResNet50 Models saved successfully.")

if __name__ == "__main__":
    train_resnet50_pipeline(epochs=1, batch_size=2)
