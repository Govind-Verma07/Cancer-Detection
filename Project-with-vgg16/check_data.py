import PIL.Image
import os
import numpy as np
import sys

def check_image(path):
    try:
        img = PIL.Image.open(path)
        img_arr = np.array(img)
        unique_vals = len(np.unique(img_arr)) if img_arr.size < 10000000 else "Too large"
        return {
            'size': img.size,
            'mode': img.mode,
            'dtype': str(img_arr.dtype),
            'unique_values': unique_vals,
            'min': int(img_arr.min()),
            'max': int(img_arr.max())
        }
    except Exception as e:
        return f"Error: {e}"

p1 = r"c:\Users\Govind Verma\OneDrive\Desktop\Breast-Cancer-Detection\data\Pixel-level annotation\IMG001.tif"
p2 = r"c:\Users\Govind Verma\OneDrive\Desktop\Breast-Cancer-Detection\data\ROI Masks\IMG001.tif"

print(f"Checking {os.path.basename(os.path.dirname(p1))}/{os.path.basename(p1)}:")
print(check_image(p1))
sys.stdout.flush()

print(f"\nChecking {os.path.basename(os.path.dirname(p2))}/{os.path.basename(p2)}:")
print(check_image(p2))
sys.stdout.flush()
