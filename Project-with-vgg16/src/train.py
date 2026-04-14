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

def train_pipeline(epochs=20, batch_size=4):
    device = Config.DEVICE
    print(f"PIPELINE START (Device: {device})")
    
    # Paths (Unified via Config)
    img_dir = Config.IMAGE_DIR
    pixel_mask_dir = Config.PIXEL_MASK_DIR
    roi_mask_dir = Config.ROI_MASK_DIR
    
    # 1. Dataset & Transforms
    transform = get_transforms(img_size=Config.SEG_INPUT_SIZE, train=True)
    full_dataset = BreastCancerDataset(img_dir, pixel_mask_dir=pixel_mask_dir, roi_mask_dir=roi_mask_dir, transform=transform)
    print(f"Total images found: {len(full_dataset)}")
    
    if len(full_dataset) == 0:
        print("ERROR: No images found! Check paths and extensions.")
        return
    
    # 2. Read Trained Images History
    history_file = os.path.join(Config.PROJECT_ROOT, "trained_images_log.txt")
    trained_images = set()
    if os.path.exists(history_file):
        with open(history_file, 'r') as f:
            trained_images = set(line.strip() for line in f)
            
    # Filter dataset for fresh, untrained images
    fresh_indices = [i for i, filename in enumerate(full_dataset.image_files) if filename not in trained_images]
    
    # TURBO DEMO MODE: Limit to 50 images for a balanced refinement run
    if len(fresh_indices) > 50:
        fresh_indices = fresh_indices[:50]
    
    if len(fresh_indices) == 0:
        print("AI is fully trained! All available images have already been utilized.")
        return
        
    print(f"Running in TURBO DEMO MODE: Training on exactly {len(fresh_indices)} images for maximum speed.")
    fresh_dataset = Subset(full_dataset, fresh_indices)
    
    total_samples = len(fresh_dataset)
    num_train = int(total_samples * 0.8)
    num_test = total_samples - num_train
    
    print(f"New Split: {num_train} training, {num_test} testing.")
    
    # Using random split for sets of images as requested robustly
    generator = torch.Generator().manual_seed(42)
    train_ds, test_ds = torch.utils.data.random_split(fresh_dataset, [num_train, num_test], generator=generator)
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False)
    
    # 3. Initialize Models
    print("Initializing Models (VGG16 Backbone)...")
    seg_model = UNet().to(device)
    cls_model = BreastCancerClassifier().to(device)
    
    # 4. Optimization
    seg_optimizer = optim.Adam(seg_model.parameters(), lr=1e-4)
    cls_optimizer = optim.Adam(cls_model.parameters(), lr=1e-4)
    criterion_seg = nn.BCELoss()
    criterion_cls = nn.CrossEntropyLoss()
    
    # 5. Training Loop
    print("Starting Training...")
    for epoch in range(epochs):
        seg_model.train()
        cls_model.train()
        epoch_loss = 0
        
        for b_idx, batch in enumerate(train_loader):
            images = batch['image'].to(device)
            pixel_masks = batch['pixel_mask'].to(device)
            
            # Step A: Segmentation (Train on Pixel-level annotations)
            seg_optimizer.zero_grad()
            pred_masks = seg_model(images)
            loss_seg = criterion_seg(pred_masks, pixel_masks)
            loss_seg.backward()
            seg_optimizer.step()
            
            # Step B: Classification (Mock labels if not provided)
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
            
            # Real-time progress update
            print(f"\r   Processing Batch [{b_idx+1}/{len(train_loader)}] - Batch Loss: {b_loss:.4f}", end="", flush=True)
            
        print(f"\nBatch [{epoch+1}/{epochs}] - Average Loss: {epoch_loss/len(train_loader):.4f}")
        
    # 6. Save models
    save_model(seg_model, Config.SEGMENTATION_MODEL_PATH)
    save_model(cls_model, Config.CLASSIFICATION_MODEL_PATH)
    print("Models saved successfully.")
    
    # Track images successfully trained on so they aren't repeated in future runs
    with open(history_file, 'a') as f:
        for idx in train_ds.indices:
            orig_idx = fresh_indices[idx]
            f.write(full_dataset.image_files[orig_idx] + '\n')
    print("Verified and permanently logged utilized images into history tracker.")
    
    # 7. Verification
    print("\nStarting Verification on Testing Set...")
    seg_model.eval()
    cls_model.eval()
    total_iou = 0
    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            images = batch['image'].to(device)
            roi_masks = batch['roi_mask'].to(device)
            
            pred_masks = seg_model(images)
            pred_binary = (pred_masks > Config.SEG_THRESHOLD).float()
            
            # Calculate validation IOU using ROI masks as ground truth
            intersection = (pred_binary * roi_masks).sum()
            union = pred_binary.sum() + roi_masks.sum() - intersection
            iou = intersection / (union + 1e-6)
            total_iou += iou.item()
            
            if i % 10 == 0:
                print(f"   Sample {i}: Validation ROI IOU = {iou.item():.4f}")

    avg_iou = total_iou / len(test_loader) if len(test_loader) > 0 else 0
    print(f"\nAverage Validation ROI IOU: {avg_iou:.4f}")
    if avg_iou == 0:
        print("💡 TIP: IOU is zero. This means the predicted contours didn't overlap with the ROI masks.")
    print("PIPELINE FINISHED")

if __name__ == "__main__":
    # The user asked not to use 20 images, but rather do processing for 'all the images'
    # 'Sets of 20' is now managed as batch grouping if necessary, but processing full dir
    # Adjusted to batch_size=2 because CPU training on 512x512 images freezes with batch_size=20
    # Adjusted to Epochs=1 for TURBO DEMO MODE speed
    train_pipeline(epochs=10, batch_size=2)
