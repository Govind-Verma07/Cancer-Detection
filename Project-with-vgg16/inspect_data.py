import os

def inspect_folders():
    base_path = r"D:\dataset for dl\Dataset DL"
    folders = ["Pixel-level annotation", "ROI Masks", "TIFF Images"]
    
    for folder in folders:
        full_path = os.path.join(base_path, folder)
        print(f"\n--- Folder: {folder} ---")
        if not os.path.exists(full_path):
            print("  Does not exist!")
            continue
            
        files = os.listdir(full_path)
        print(f"  Total files: {len(files)}")
        if len(files) > 0:
            print(f"  First 5 files: {files[:5]}")
            extensions = set([os.path.splitext(f)[1].lower() for f in files])
            print(f"  Extensions found: {extensions}")

if __name__ == "__main__":
    inspect_folders()
