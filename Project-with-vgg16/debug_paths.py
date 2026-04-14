import os

img_dir = os.path.join("data", "Pixel-level annotation")
mask_dir = os.path.join("data", "ROI Masks")

print(f"Checking img_dir: {img_dir}")
print(f"Exists: {os.path.exists(img_dir)}")
if os.path.exists(img_dir):
    files = os.listdir(img_dir)
    print(f"Found {len(files)} files.")
    tif_files = [f for f in files if f.endswith('.tif')]
    print(f"Found {len(tif_files)} .tif files.")
    if len(files) > 0:
        print(f"First file: {files[0]}")

print(f"\nChecking mask_dir: {mask_dir}")
print(f"Exists: {os.path.exists(mask_dir)}")
if os.path.exists(mask_dir):
    files = os.listdir(mask_dir)
    print(f"Found {len(files)} files.")
