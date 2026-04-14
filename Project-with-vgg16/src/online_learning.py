import torch
import torch.nn as nn
import torch.optim as optim
import os
import cv2
import numpy as np

from src.segmentation import UNet
from src.preprocessing import get_transforms
from utils.config import Config
from utils.helpers import load_model, save_model

def refine_from_feedback(image_path, results, feedback_type):
    """
    Continually refines the model based on user feedback.
    feedback_type: "Correct", "False Positive", or "False Negative"
    """
    device = Config.DEVICE
    
    # 1. Load Image and Transform
    try:
        from PIL import Image
        img_orig = Image.open(image_path).convert('RGB')
    except Exception as e:
        print(f"Error reading image: {e}")
        return False
        
    transform = get_transforms(img_size=Config.SEG_INPUT_SIZE, train=True)
    img_tensor = transform(img_orig).unsqueeze(0).to(device)
    
    # 2. Get the mask Target based on Feedback
    if feedback_type == "Correct":
        # Pseudo-labeling: reinforce the model's own correct prediction
        mask_np = results.get('binary_mask')
        if mask_np is None:
            return False
            
        mask_pil = Image.fromarray((mask_np * 255).astype(np.uint8)).convert('L').resize(Config.SEG_INPUT_SIZE, Image.NEAREST)
        mask_tensor = torch.tensor((np.array(mask_pil) > 127).astype(np.float32)).unsqueeze(0).unsqueeze(0).to(device)
        
    elif feedback_type == "False Positive":
        # Model predicted a tumor, but there is none. Set target to ALL ZEROES.
        mask_tensor = torch.zeros((1, 1, Config.SEG_INPUT_SIZE[1], Config.SEG_INPUT_SIZE[0]), dtype=torch.float32).to(device)
        
    elif feedback_type == "False Negative":
        # Missed a tumor. Without manual drawing tools, we can't easily guess the shape.
        # But we can try to force the heuristic fallback's mask if available, or just skip.
        # For a simplified online learning demo, we'll reinforce using the raw heuristic intensity.
        grayscale = np.mean(np.array(img_orig), axis=2) / 255.0
        heur_mask = (grayscale > 0.6).astype(np.float32)
        mask_pil = Image.fromarray((heur_mask * 255).astype(np.uint8)).convert('L').resize(Config.SEG_INPUT_SIZE, Image.NEAREST)
        mask_tensor = torch.tensor((np.array(mask_pil) > 127).astype(np.float32)).unsqueeze(0).unsqueeze(0).to(device)
    else:
        return False

    # 3. Load Model
    seg_model = load_model(UNet().to(device), Config.SEGMENTATION_MODEL_PATH, device=device)
    seg_model.train()
    
    # 4. Single-Step Refinement (Online Learning)
    optimizer = optim.Adam(seg_model.parameters(), lr=1e-5) # Very low LR for micro-adjustments
    criterion = nn.BCELoss()
    
    # Do a quick 2-epoch micro-training on this single example
    for _ in range(2):
        optimizer.zero_grad()
        preds = seg_model(img_tensor)
        loss = criterion(preds, mask_tensor)
        loss.backward()
        optimizer.step()
        
    # 5. Save the updated weights
    save_model(seg_model, Config.SEGMENTATION_MODEL_PATH)
    
    return True
