import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import os
import sys
import cv2
import numpy as np
from tqdm import tqdm
from PIL import Image

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.dataset import BreastCancerDataset
from src.segmentation import UNet
from src.classification import BreastCancerClassifier
from src.preprocessing import get_transforms
from src.inference import run_inference
from src.visualization import overlay_contours
from utils.config import Config
from utils.helpers import load_model, save_model

def refine_and_test(num_refine=10, epochs=5):
    print("🚀 PIPELINE START", flush=True)
    # Create progress file immediately
    with open("batch_progress.txt", "w") as f:
        f.write("Batch process initialized.\n")
        
    device = Config.DEVICE
    print(f"🚀 Initializing Batch Verification... (Device: {device})", flush=True)
    
    # 1. Paths
    test_img_dir = os.path.join("data", "test-images")
    gt_mask_dir = os.path.join("data", "Pixel-level annotation")
    output_dir = os.path.join("data", "test-outputs")
    os.makedirs(output_dir, exist_ok=True)
    
    # 2. Dataset for Refinement (Fine-tuning)
    transform = get_transforms(img_size=Config.SEG_INPUT_SIZE, train=True)
    refine_dataset = BreastCancerDataset(test_img_dir, mask_dir=gt_mask_dir, transform=transform)
    
    # Reload models for inference (Load once)
    print("📂 Loading Models...", flush=True)
    seg_model = load_model(UNet().to(device), Config.SEGMENTATION_MODEL_PATH, device=device)
    cls_model = load_model(BreastCancerClassifier().to(device), Config.CLASSIFICATION_MODEL_PATH, device=device)
    
    if len(refine_dataset) == 0:
        print("⚠️ No matching images and ground truth masks found for refinement. Skipping training stage...")
    else:
        num_refine = min(num_refine, len(refine_dataset))
        refine_loader = DataLoader(Subset(refine_dataset, range(num_refine)), batch_size=2, shuffle=True)
        
        # 4. Refinement Stage (Mini-Training)
        print(f"\n🧠 STAGE 1: Refining Segmentation Model on {num_refine} samples...", flush=True)
        optimizer = optim.Adam(seg_model.parameters(), lr=5e-5) # Low learning rate for refinement
        criterion = nn.BCELoss()
        
        for epoch in range(epochs):
            seg_model.train()
            epoch_loss = 0
            for batch in refine_loader:
                imgs = batch['image'].to(device)
                msks = batch['mask'].to(device)
                
                optimizer.zero_grad()
                preds = seg_model(imgs)
                loss = criterion(preds, msks)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            
            print(f"   Epoch {epoch+1}/{epochs} | Refinement Loss: {epoch_loss/len(refine_loader):.6f}", flush=True)
            
        save_model(seg_model, Config.SEGMENTATION_MODEL_PATH)
        print("💾 Refined Segmentation Model saved.", flush=True)
        
    seg_model.eval()
    
    # 5. Batch Testing & Visualization
    print(f"📸 STAGE 2: Batch Testing & Visualization...", flush=True)
    all_files = [f for f in os.listdir(test_img_dir) if f.lower().endswith(('.tif', '.png', '.jpg'))]
    
    transform = get_transforms(img_size=Config.SEG_INPUT_SIZE, train=False)
    
    for filename in tqdm(all_files, desc="Batch Inference"):
        # Log progress
        with open("batch_progress.txt", "a") as logf:
            logf.write(f"Analyzing {filename}\n")
            
        img_path = os.path.join(test_img_dir, filename)
        print(f"📄 Analyzing {filename}...", flush=True)
        
        # Run Inference (Pass pre-loaded models)
        results = run_inference(img_path, seg_model=seg_model, cls_model=cls_model)
        if results is None: continue
        
        # Save visualized output
        output_path = os.path.join(output_dir, f"RES_{os.path.splitext(filename)[0]}.png")
        cv2.imwrite(output_path, cv2.cvtColor(results['visual_result'], cv2.COLOR_RGB2BGR))
        
    print(f"\n✅ All results saved to {output_dir}", flush=True)

if __name__ == "__main__":
    # Refine on 20 samples, 4 epochs
    refine_and_test(num_refine=20, epochs=4)
