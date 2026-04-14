import os

directories = [
    "data/raw",
    "data/processed",
    "data/patches",
    "models",
    "src",
    "ui",
    "utils"
]

for directory in directories:
    os.makedirs(directory, exist_ok=True)
    print(f"Created directory: {directory}")

with open("requirements.txt", "w") as f:
    f.write("# Requirements for Breast Cancer Tumor Analysis\n")
    f.write("torch\ntorchvision\nopencv-python\nnumpy\npillow\nmatplotlib\nstreamlit\nscikit-learn\n")

with open("README.md", "w") as f:
    f.write("# Breast Cancer Tumor Analysis Pipeline\n")
    f.write("\nAn end-to-end medical image analysis system for breast cancer detection and staging.\n")
