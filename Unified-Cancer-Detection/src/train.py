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
from utils.helpers import save_model

def train_pipeline(num_train=15, num_test=5, epochs=20):
    device = Config.DEVICE
    print(f"Using device: {device}")
    
    # Paths (Configured for the user's dataset)
    # Using 'Pixel-level annotation' as Image and 'ROI Masks' as Mask
    img_dir = os.path.join("data", "Pixel-level annotation")
    mask_dir = os.path.join("data", "ROI Masks")
    
    # 1. Dataset & Transforms
    transform = get_transforms(img_size=Config.SEG_INPUT_SIZE, train=True)
    full_dataset = BreastCancerDataset(img_dir, mask_dir=mask_dir, transform=transform)
    print(f"Total images found: {len(full_dataset)}")
    
    if len(full_dataset) == 0:
        print("ERROR: No images found! Check paths and extensions.")
        return
    
    # 2. Split (Dynamic based on data size)
    total_samples = len(full_dataset)
    if total_samples < 2:
        print("ERROR: Need at least 2 images (1 train, 1 test).")
        return
        
    num_train = min(num_train, total_samples // 2)
    num_test = min(num_test, total_samples - num_train)
    
    print(f"Split: {num_train} training, {num_test} testing.")
    
    indices = list(range(total_samples))
    train_indices = indices[:num_train]
    test_indices = indices[num_train:num_train+num_test]
    
    train_ds = Subset(full_dataset, train_indices)
    test_ds = Subset(full_dataset, test_indices)
    
    train_loader = DataLoader(train_ds, batch_size=2, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False)
    
    # 3. Initialize Models
    seg_model = UNet().to(device)
    cls_model = BreastCancerClassifier().to(device)
    
    # 4. Optimization
    seg_optimizer = optim.Adam(seg_model.parameters(), lr=1e-4)
    cls_optimizer = optim.Adam(cls_model.parameters(), lr=1e-4)
    criterion_seg = nn.BCELoss()
    criterion_cls = nn.CrossEntropyLoss()
    
    # 5. Training Loop (Simplified for demo)
    print("Starting Training...")
    for epoch in range(epochs):
        seg_model.train()
        cls_model.train()
        total_loss = 0
        
        for batch in train_loader:
            images = batch['image'].to(device)
            masks = batch['mask'].to(device) # [B, 1, H, W]
            
            # Step A: Segmentation
            seg_optimizer.zero_grad()
            pred_masks = seg_model(images)
            loss_seg = criterion_seg(pred_masks, masks)
            loss_seg.backward()
            seg_optimizer.step()
            
            # Step B: Classification (Mock labels for now if not provided)
            cls_optimizer.zero_grad()
            # If labels are not in dataset, we use dummy labels just to ensure the architecture works
            labels_class = batch.get('label_class', torch.zeros(images.size(0), dtype=torch.long)).to(device)
            labels_stage = batch.get('label_stage', torch.zeros(images.size(0), dtype=torch.long)).to(device)
            
            pred_class, pred_stage = cls_model(images)
            loss_class = criterion_cls(pred_class, labels_class)
            loss_stage = criterion_cls(pred_stage, labels_stage)
            loss_cls = loss_class + loss_stage
            loss_cls.backward()
            cls_optimizer.step()
            
            total_loss += loss_seg.item() + loss_cls.item()
            
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(train_loader):.4f}")
        
    # 6. Save models
    save_model(seg_model, Config.SEGMENTATION_MODEL_PATH)
    save_model(cls_model, Config.CLASSIFICATION_MODEL_PATH)
    print("Models saved successfully.")
    
    # 7. Verification (Testing on the 10 testing samples)
    print("\nStarting Verification on Testing Set...")
    seg_model.eval()
    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            images = batch['image'].to(device)
            masks = batch['mask'].to(device)
            
            pred_masks = seg_model(images)
            # Binary mask with threshold
            pred_binary = (pred_masks > Config.SEG_THRESHOLD).float()
            
            # Calculate simple IOU for verification
            intersection = (pred_binary * masks).sum()
            union = pred_binary.sum() + masks.sum() - intersection
            iou = intersection / (union + 1e-6)
            
            print(f"Test Sample {indices[num_train+i]}: IOU = {iou.item():.4f}")

if __name__ == "__main__":
    # Retraining as requested: 15 for training, 5 for testing
    train_pipeline(num_train=15, num_test=5, epochs=20)
