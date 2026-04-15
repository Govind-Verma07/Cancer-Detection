import os
import torch

class Config:
    # Path settings
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_RAW = os.path.join(PROJECT_ROOT, "data", "raw")
    DATA_PROCESSED = os.path.join(PROJECT_ROOT, "data", "processed")
    DATA_PATCHES = os.path.join(PROJECT_ROOT, "data", "patches")
    RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
    COMPARISON_LOG = os.path.join(RESULTS_DIR, "comparison_log.csv")
    MEDIA_DIR = os.path.join(PROJECT_ROOT, "media")
    
    # Model-specific Paths
    RESNET50_DIR = os.path.join(PROJECT_ROOT, "resnet50")
    VGG16_DIR = os.path.join(PROJECT_ROOT, "vgg16")
    
    # ResNet50 Models
    RESNET50_SEG_PATH = os.path.join(RESNET50_DIR, "resnet50_seg.pth")
    RESNET50_CLS_PATH = os.path.join(RESNET50_DIR, "resnet50_cls.pth")
    
    # VGG16 Models
    VGG16_SEG_PATH = os.path.join(VGG16_DIR, "vgg16_seg.pth")
    VGG16_CLS_PATH = os.path.join(VGG16_DIR, "vgg16_cls.pth")
    
    # Defaults (Backward compatibility)
    SEGMENTATION_MODEL_PATH = RESNET50_SEG_PATH
    CLASSIFICATION_MODEL_PATH = RESNET50_CLS_PATH
    
    # Thresholds
    RESNET_SEG_THRESHOLD = 0.50
    VGG_SEG_THRESHOLD = 0.50
    
    # Patch settings
    PATCH_SIZE = 256
    STRIDE = 256
    
    # Segmentation settings
    SEG_INPUT_SIZE = (256, 256)
    SEG_THRESHOLD = 0.70   # Lowered to 0.70 to improve sensitivity
    
    # Classification settings
    CLS_INPUT_SIZE = (224, 224)
    NUM_CLASSES = 2 # Benign, Malignant
    NUM_STAGES = 4 # Stage 1, 2, 3, 4
    
    # Device (auto-detect GPU, fall back to CPU)
    DEVICE = "cpu"  # Force CPU for testing
    
    # Visualization
    COLOR_MALIGNANT = (255, 0, 0) # Red in RGB
    COLOR_BENIGN = (0, 255, 0)    # Green in RGB
