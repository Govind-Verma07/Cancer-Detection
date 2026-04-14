import sys
import os

# Add project root
sys.path.append(os.getcwd())

try:
    print("🚀 Starting Smoke Test...")
    from src.dataset import BreastCancerDataset
    from src.segmentation import UNet
    from src.classification import BreastCancerClassifier
    from utils.config import Config
    print("✅ Imports successful.")
    
    print("🏗️ Testing model instantiation...")
    model = BreastCancerClassifier()
    print("✅ Model instantiated.")
    
    print(f"📁 Checking image dir: {Config.IMAGE_DIR}")
    if os.path.exists(Config.IMAGE_DIR):
        print(f"✅ Image dir exists: {len(os.listdir(Config.IMAGE_DIR))} files found.")
    else:
        print(f"❌ Image dir NOT found: {Config.IMAGE_DIR}")
        
    print("🎯 Smoke test finished.")
except Exception as e:
    print(f"❌ FAILED: {e}")
    import traceback
    traceback.print_exc()
