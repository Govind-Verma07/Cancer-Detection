import torch
import os

def load_model(model_or_class, model_path, device="cpu"):
    """Load a model's state dict from a path. Handles both classes and instances."""
    if isinstance(model_or_class, type):
        model = model_or_class()
    else:
        model = model_or_class
        
    if os.path.exists(model_path):
        try:
            model.load_state_dict(torch.load(model_path, map_location=device))
            print("[OK] Loaded weights from " + model_path)
        except Exception as e:
            print("[WARN] Error loading weights from " + model_path + ": " + str(e))
    else:
        print("[WARN] Model path not found, using random weights: " + model_path)
    
    model.to(device)
    model.eval()
    return model

def save_model(model, model_path):
    """Save a model's state dict to a path."""
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    torch.save(model.state_dict(), model_path)
