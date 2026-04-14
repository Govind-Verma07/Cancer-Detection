import os
import torch

class Config:
    # Path settings
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_RAW = os.path.join(PROJECT_ROOT, "data", "raw")
    DATA_PROCESSED = os.path.join(PROJECT_ROOT, "data", "processed")
    DATA_PATCHES = os.path.join(PROJECT_ROOT, "data", "patches")
    MODEL_DIR = os.path.join(PROJECT_ROOT, "models")
    RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
    COMPARISON_LOG = os.path.join(RESULTS_DIR, "comparison_log.csv")
    
    # Model-specific Paths
    RESNET50_DIR = os.path.join(MODEL_DIR, "resnet50")
    VGG16_DIR = os.path.join(MODEL_DIR, "vgg16")
    
    # Defaults (Backward compatibility)
    SEGMENTATION_MODEL_PATH = os.path.join(RESNET50_DIR, "segmentation_model.pth")
    CLASSIFICATION_MODEL_PATH = os.path.join(RESNET50_DIR, "classification_model.pth")
    
    # Thresholds
    RESNET_SEG_THRESHOLD = 0.50
    VGG_SEG_THRESHOLD = 0.50
    
    # Patch settings
    PATCH_SIZE = 512 # or 224
    STRIDE = 512
    
    # Segmentation settings
    SEG_INPUT_SIZE = (512, 512)
    SEG_THRESHOLD = 0.70   # Lowered to 0.70 to improve sensitivity
    
    # Classification settings
    CLS_INPUT_SIZE = (224, 224)
    NUM_CLASSES = 2 # Benign, Malignant
    NUM_STAGES = 4 # Stage 1, 2, 3, 4
    
    # Device (Forced to 'cpu' for stability on this environment)
    DEVICE = "cpu"
    
    # Visualization
    COLOR_MALIGNANT = (255, 0, 0) # Red in RGB
    COLOR_BENIGN = (0, 255, 0)    # Green in RGB
