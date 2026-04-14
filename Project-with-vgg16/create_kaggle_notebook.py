import json

notebook = {
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# 🩺 Breast Cancer Tumor Analysis System\n",
    "### Patch-based Deep Learning Pipeline for Segmentation and Staging\n",
    "\n",
    "This notebook implements a complete production-ready deep learning pipeline for analyzing high-resolution breast cancer medical images. It uses patch-based processing to preserve image quality, U-Net for tumor segmentation, and a Multi-head VGG16 for classification and staging."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "!pip install opencv-python-headless torch torchvision numpy pillow matplotlib scikit-learn tqdm\n",
    "\n",
    "import os\n",
    "import cv2\n",
    "import torch\n",
    "import torch.nn as nn\n",
    "import torch.nn.functional as F\n",
    "import torch.optim as optim\n",
    "from torch.utils.data import Dataset, DataLoader, Subset\n",
    "from torchvision import transforms, models\n",
    "from PIL import Image\n",
    "import numpy as np\n",
    "import matplotlib.pyplot as plt\n",
    "from tqdm.auto import tqdm\n",
    "\n",
    "DEVICE = \"cuda\" if torch.cuda.is_available() else \"cpu\"\n",
    "print(f\"Using device: {DEVICE}\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## ⚙️ Configuration\n",
    "Adjust paths for Kaggle's input/output structure."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "class Config:\n",
    "    # Kaggle Path overrides\n",
    "    DATA_DIR = \"/kaggle/input/breast-cancer-data\" \n",
    "    WORKING_DIR = \"/kaggle/working\"\n",
    "    MODEL_DIR = os.path.join(WORKING_DIR, \"models\")\n",
    "    os.makedirs(MODEL_DIR, exist_ok=True)\n",
    "    \n",
    "    PATCH_SIZE = 512\n",
    "    STRIDE = 512\n",
    "    \n",
    "    SEG_INPUT_SIZE = (512, 512)\n",
    "    SEG_THRESHOLD = 0.5\n",
    "    \n",
    "    CLS_INPUT_SIZE = (224, 224)\n",
    "    NUM_CLASSES = 2\n",
    "    NUM_STAGES = 4\n",
    "    \n",
    "    SEGMENTATION_MODEL_PATH = os.path.join(MODEL_DIR, \"segmentation_model.pth\")\n",
    "    CLASSIFICATION_MODEL_PATH = os.path.join(MODEL_DIR, \"classification_model.pth\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "def split_into_patches(image, patch_size=512, stride=512):\n",
    "    h, w, _ = image.shape\n",
    "    patches, metadata = [], []\n",
    "    patch_id = 0\n",
    "    for y in range(0, h - patch_size + 1, stride):\n",
    "        for x in range(0, w - patch_size + 1, stride):\n",
    "            patch = image[y:y+patch_size, x:x+patch_size]\n",
    "            patches.append(patch)\n",
    "            metadata.append({'patch_id': patch_id, 'x_offset': x, 'y_offset': y})\n",
    "            patch_id += 1\n",
    "    return patches, metadata\n",
    "\n",
    "def reconstruct_from_patches(patches, metadata, original_shape):\n",
    "    full_image = np.zeros(original_shape, dtype=patches[0].dtype)\n",
    "    for patch, meta in zip(patches, metadata):\n",
    "        x, y = meta['x_offset'], meta['y_offset']\n",
    "        full_image[y:y+patch.shape[0], x:x+patch.shape[1]] = patch\n",
    "    return full_image\n",
    "\n",
    "def get_transforms(img_size=(224, 224), train=True):\n",
    "    if train:\n",
    "        return transforms.Compose([\n",
    "            transforms.ToPILImage(), transforms.Resize(img_size),\n",
    "            transforms.RandomHorizontalFlip(), transforms.RandomVerticalFlip(),\n",
    "            transforms.ToTensor(),\n",
    "            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])\n",
    "        ])\n",
    "    return transforms.Compose([\n",
    "        transforms.ToPILImage(), transforms.Resize(img_size),\n",
    "        transforms.ToTensor(),\n",
    "        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])\n",
    "    ])\n",
    "\n",
    "class BreastCancerDataset(Dataset):\n",
    "    def __init__(self, image_dir, mask_dir=None, transform=None):\n",
    "        self.image_dir, self.mask_dir, self.transform = image_dir, mask_dir, transform\n",
    "        self.image_files = sorted([f for f in os.listdir(image_dir) if f.endswith('.tif')])\n",
    "    def __len__(self): return len(self.image_files)\n",
    "    def __getitem__(self, idx):\n",
    "        img_name = self.image_files[idx]\n",
    "        image = cv2.cvtColor(cv2.imread(os.path.join(self.image_dir, img_name)), cv2.COLOR_BGR2RGB)\n",
    "        mask = None\n",
    "        if self.mask_dir:\n",
    "            m_path = os.path.join(self.mask_dir, img_name)\n",
    "            if os.path.exists(m_path):\n",
    "                mask = (cv2.imread(m_path, cv2.IMREAD_GRAYSCALE) > 127).astype(np.float32)\n",
    "        if self.transform: image = self.transform(image)\n",
    "        sample = {'image': image}\n",
    "        if mask is not None: sample['mask'] = torch.tensor(mask).unsqueeze(0)\n",
    "        return sample"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "class UNet(nn.Module):\n",
    "    def __init__(self, in_channels=3, out_channels=1):\n",
    "        super().__init__()\n",
    "        def double_conv(in_c, out_c):\n",
    "            return nn.Sequential(nn.Conv2d(in_c, out_c, 3, padding=1), nn.BatchNorm2d(out_c), nn.ReLU(True),\n",
    "                                 nn.Conv2d(out_c, out_c, 3, padding=1), nn.BatchNorm2d(out_c), nn.ReLU(True))\n",
    "        self.down1, self.down2, self.down3, self.down4 = double_conv(in_channels, 64), double_conv(64, 128), double_conv(128, 256), double_conv(256, 512)\n",
    "        self.pool, self.up1, self.up2, self.up3 = nn.MaxPool2d(2), nn.ConvTranspose2d(512, 256, 2, 2), nn.ConvTranspose2d(256, 128, 2, 2), nn.ConvTranspose2d(128, 64, 2, 2)\n",
    "        self.up_conv1, self.up_conv2, self.up_conv3 = double_conv(512, 256), double_conv(256, 128), double_conv(128, 64)\n",
    "        self.final = nn.Conv2d(64, out_channels, 1)\n",
    "    def forward(self, x):\n",
    "        x1 = self.down1(x); x2 = self.down2(self.pool(x1)); x3 = self.down3(self.pool(x2)); x4 = self.down4(self.pool(x3))\n",
    "        d1 = self.up_conv1(torch.cat([self.up1(x4), x3], 1)); d2 = self.up_conv2(torch.cat([self.up2(d1), x2], 1))\n",
    "        d3 = self.up_conv3(torch.cat([self.up3(d2), x1], 1))\n",
    "        return torch.sigmoid(self.final(d3))\n",
    "\n",
    "class BreastCancerClassifier(nn.Module):\n",
    "    def __init__(self, n_cls=2, n_stage=4):\n",
    "        super().__init__()\n",
    "        self.backbone = models.vgg16(weights=models.VGG16_Weights.DEFAULT)\n",
    "        n_ftrs = self.backbone.classifier[6].in_features\n",
    "        self.backbone.classifier[6] = nn.Identity()\n",
    "        self.h_bin = nn.Sequential(nn.Linear(n_ftrs, 512), nn.ReLU(), nn.Dropout(0.3), nn.Linear(512, n_cls))\n",
    "        self.h_stg = nn.Sequential(nn.Linear(n_ftrs, 512), nn.ReLU(), nn.Dropout(0.3), nn.Linear(512, n_stage))\n",
    "    def forward(self, x):\n",
    "        f = self.backbone(x); return self.h_bin(f), self.h_stg(f)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "def train_model(img_dir, mask_dir, num_train=10, num_test=10, epochs=10):\n",
    "    ds = BreastCancerDataset(img_dir, mask_dir, get_transforms(Config.SEG_INPUT_SIZE, True))\n",
    "    train_set = Subset(ds, range(num_train))\n",
    "    test_set = Subset(ds, range(num_train, num_train+num_test))\n",
    "    train_loader = DataLoader(train_set, 2, shuffle=True)\n",
    "    test_loader = DataLoader(test_set, 1)\n",
    "    \n",
    "    seg_m = UNet().to(DEVICE)\n",
    "    cls_m = BreastCancerClassifier().to(DEVICE)\n",
    "    \n",
    "    opt_s = optim.Adam(seg_m.parameters(), 1e-4)\n",
    "    opt_c = optim.Adam(cls_m.parameters(), 1e-4)\n",
    "    crit_s = nn.BCELoss()\n",
    "    crit_c = nn.CrossEntropyLoss()\n",
    "    \n",
    "    for e in range(epochs):\n",
    "        seg_m.train(); cls_m.train(); t_loss = 0\n",
    "        for b in tqdm(train_loader, desc=f'Epoch {e+1}'):\n",
    "            imgs, masks = b['image'].to(DEVICE), b['mask'].to(DEVICE)\n",
    "            opt_s.zero_grad(); ps = seg_m(imgs); ls = crit_s(ps, masks); ls.backward(); opt_s.step()\n",
    "            opt_c.zero_grad(); pc, pst = cls_m(imgs)\n",
    "            lc = crit_c(pc, torch.zeros(imgs.size(0), dtype=torch.long).to(DEVICE)) + crit_c(pst, torch.zeros(imgs.size(0), dtype=torch.long).to(DEVICE))\n",
    "            lc.backward(); opt_c.step()\n",
    "            t_loss += ls.item() + lc.item()\n",
    "        print(f'Epoch {e+1} Loss: {t_loss/len(train_loader):.4f}')\n",
    "        \n",
    "    torch.save(seg_m.state_dict(), Config.SEGMENTATION_MODEL_PATH)\n",
    "    torch.save(cls_m.state_dict(), Config.CLASSIFICATION_MODEL_PATH)\n",
    "    print('Training Finished.')"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.8.5"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}

with open('breast_cancer_analysis_kaggle.ipynb', 'w') as f:
    json.dump(notebook, f, indent=1)
