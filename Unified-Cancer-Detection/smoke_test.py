"""
Quick smoke-test for the ensemble pipeline.
Creates a synthetic 512x512 white image and runs end-to-end inference.
"""
import os
import sys
import numpy as np
from PIL import Image

print("[TEST] Starting smoke test...")

# Root on path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("[TEST] Path added")

# Save a synthetic test image
os.makedirs("results", exist_ok=True)
test_img_path = os.path.join("results", "smoke_test.jpg")
Image.fromarray(np.random.randint(100, 200, (512, 512, 3), dtype=np.uint8)).save(test_img_path)
print("[TEST] Synthetic test image saved ->", test_img_path)

print("[TEST] Importing ensemble_learning...")
from src.ensemble_learning import predict_ensemble

print("[TEST] Running ensemble inference...")
result = predict_ensemble(test_img_path, ground_truth=None)

print("[RESULT] ResNet50 tumor burden:", result["resnet"]["tumor_burden"])
print("[RESULT] VGG16 tumor_burden:   ", result["vgg"]["tumor_burden"])
print("[RESULT] Ensemble score:        ", result["ensemble"]["score"])
print("[RESULT] Status:                ", result["ensemble"]["status"])
print()
print("[PASS] Smoke test complete.")
