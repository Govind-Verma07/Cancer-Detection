import torch
import torch.nn as nn
import os
import sys
import cv2
import numpy as np
from PIL import Image

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.inference import run_inference
from utils.config import Config

def test_single():
    print("🧪 Running Single-Sample Verification...")
    test_img = os.path.join("data", "test-images", "IMG001.tif")
    if not os.path.exists(test_img):
        print(f"❌ Test image not found: {test_img}")
        return
        
    print(f"📄 Processing: {test_img}...")
    try:
        results = run_inference(test_img)
        output_path = "test_verify.png"
        cv2.imwrite(output_path, cv2.cvtColor(results['visual_result'], cv2.COLOR_RGB2BGR))
        print(f"✅ Success! Output saved to {output_path}")
        print(f"📊 Findings: {len(results['findings'])}")
        for f in results['findings']:
            print(f"   - {f['classification']} | {f['stage']} | Loc: {f['location_pct']}")
    except Exception as e:
        print(f"❌ Error during inference: {e}")

if __name__ == "__main__":
    test_single()
