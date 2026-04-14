import os
import sys

# Add current directory to path
sys.path.append(os.getcwd())

print("Testing imports...")
try:
    from src.dataset import BreastCancerDataset
    from src.segmentation import UNet
    from src.classification import BreastCancerClassifier
    from src.preprocessing import get_transforms, preprocess_patch
    from src.patching import split_into_patches, reconstruct_from_patches
    from src.inference import run_inference
    from src.metrics import calculate_tumor_metrics, get_relative_coordinates
    from src.visualization import overlay_contours, save_output_image
    from utils.config import Config
    from utils.helpers import load_model, save_model
    print("✅ All core modules imported successfully.")
except Exception as e:
    print(f"❌ Import test failed: {e}")
    sys.exit(1)

print("\nFinal codebase check complete.")
