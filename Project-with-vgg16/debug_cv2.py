import cv2
import os

def debug_cv2():
    path = os.path.join("data", "test-images", "IMG001.tif")
    print(f"🔍 Testing CV2 on: {path}")
    if not os.path.exists(path):
        print("❌ Path does not exist!")
        return
        
    img = cv2.imread(path)
    if img is not None:
        print(f"✅ CV2 Success! Shape: {img.shape}")
    else:
        print("❌ CV2 Failed to read image.")

if __name__ == "__main__":
    debug_cv2()
