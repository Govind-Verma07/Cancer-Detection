import cv2
import numpy as np
import os

def create_mock_image(output_path, size=(512, 512), is_mask=False):
    if is_mask:
        # Create a black background with a white "tumor" mask
        img = np.zeros((size[0], size[1]), dtype=np.uint8)
        center = (size[1]//2, size[0]//2)
        axes = (size[1]//4, size[0]//3)
        cv2.ellipse(img, center, axes, 45, 0, 360, 255, -1)
    else:
        # Create a gray background (breast tissue simulation)
        img = np.zeros((size[0], size[1], 3), dtype=np.uint8) + 50
        center = (size[1]//2, size[0]//2)
        axes = (size[1]//4, size[0]//3)
        cv2.ellipse(img, center, axes, 45, 0, 360, (200, 200, 200), -1)
        # Add some noise
        noise = np.random.normal(0, 15, img.shape).astype(np.int16)
        img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, img)

def setup_mock_dataset(num_samples=20):
    print(f"Setting up mock dataset with {num_samples} samples...")
    img_dir = os.path.join("data", "Pixel-level annotation")
    mask_dir = os.path.join("data", "ROI Masks")
    
    for i in range(num_samples):
        name = f"sample_{i:03d}.png"
        create_mock_image(os.path.join(img_dir, name), is_mask=False)
        create_mock_image(os.path.join(mask_dir, name), is_mask=True)
        
    print(f"Mock dataset created at {img_dir} and {mask_dir}")

if __name__ == "__main__":
    setup_mock_dataset()
