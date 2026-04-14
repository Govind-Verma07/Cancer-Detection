import os
import torch

class Config:
    # Path settings
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    # Data Paths (Redirected from D: drive to workspace dataset)
    IMAGE_DIR = os.path.join(PROJECT_ROOT, "..", "Project-with-resnet50", "data", "Pixel-level annotation")
    PIXEL_MASK_DIR = os.path.join(PROJECT_ROOT, "..", "Project-with-resnet50", "data", "Pixel-level annotation")
    ROI_MASK_DIR = os.path.join(PROJECT_ROOT, "..", "Project-with-resnet50", "data", "ROI Masks")
    
    MODEL_DIR = os.path.join(PROJECT_ROOT, "models")
    
    # Model Paths
    SEGMENTATION_MODEL_PATH = os.path.join(MODEL_DIR, "segmentation_model.pth")
    CLASSIFICATION_MODEL_PATH = os.path.join(MODEL_DIR, "classification_model.pth")
    
    # Patch settings
    PATCH_SIZE = 512 # or 224
    STRIDE = 512
    
    # Segmentation settings
    SEG_INPUT_SIZE = (512, 512)
    SEG_THRESHOLD = 0.50   # Lowered to 0.50 for better detection in early training
    
    # Classification settings
    CLS_INPUT_SIZE = (224, 224)
    NUM_CLASSES = 2 # Benign, Malignant
    NUM_STAGES = 4 # Stage 1, 2, 3, 4
    
    # Device (Forced to 'cpu' for stability on this environment)
    DEVICE = "cpu"
    
    # Visualization Colors (RGB)
    COLOR_BENIGN = (0, 255, 0)       # Green
    COLOR_STAGE_1 = (255, 255, 0)    # Yellow
    COLOR_STAGE_2 = (255, 165, 0)    # Orange
    COLOR_STAGE_3 = (255, 0, 0)      # Red
    COLOR_STAGE_4 = (139, 0, 0)      # Dark Red
