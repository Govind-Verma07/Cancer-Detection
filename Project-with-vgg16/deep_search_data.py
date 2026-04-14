import os

def deep_search():
    base_path = r"D:\dataset for dl\Dataset DL"
    print(f"Deep searching in {base_path}...")
    
    if not os.path.exists(base_path):
        print("Error: Base path not found.")
        return

    extensions = ('.jpg', '.png', '.jpeg', '.tif', '.tiff')
    found_info = {}

    for root, dirs, files in os.walk(base_path):
        image_files = [f for f in files if f.lower().endswith(extensions)]
        if image_files:
            rel_path = os.path.relpath(root, base_path)
            found_info[rel_path] = {
                'count': len(image_files),
                'first_file': image_files[0],
                'ext': os.path.splitext(image_files[0])[1]
            }

    if not found_info:
        print("No images found anywhere in the tree!")
    else:
        print("\n--- Summary of Images Found ---")
        for path, info in found_info.items():
            print(f"Path: ...\\Dataset DL\\{path}")
            print(f"  Count: {info['count']}")
            print(f"  Example: {info['first_file']}")
            print("-" * 30)

if __name__ == "__main__":
    deep_search()
