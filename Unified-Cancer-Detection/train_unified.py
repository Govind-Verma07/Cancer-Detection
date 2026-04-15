"""
Unified GPU training script for Breast Cancer Detection Suite.
Trains ResNet50 and VGG16 sequentially per epoch to stay within 6GB VRAM.

Label strategy:
  - Pixel mask exists       -> label_class = 1 (Malignant)
  - No pixel mask           -> label_class = 0 (Benign)
  - label_stage defaulted to 0 (no staging GT available)
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
from PIL import Image
import time
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from src.segmentation import UNet
from resnet50.model import BreastCancerClassifier as ResNetClassifier
from vgg16.model import BreastCancerClassifier as VGGClassifier
from utils.config import Config
from utils.helpers import save_model

IMG_DIR = os.path.join(ROOT, "media", "TIFF Images")
PIX_DIR = os.path.join(ROOT, "media", "Pixel-level annotation", "Pixel-level annotation")

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class MediaTIFFDataset(Dataset):
    def __init__(self, img_dir, pix_dir, img_size=(256, 256), cls_size=(224, 224)):
        self.img_size = img_size
        self.cls_size = cls_size
        self.pix_dir  = pix_dir
        pix_masks = set(os.listdir(pix_dir))
        all_imgs  = sorted([f for f in os.listdir(img_dir) if f.lower().endswith(('.tif', '.tiff'))])
        self.samples = []
        for fname in all_imgs:
            self.samples.append({
                'img_path':  os.path.join(img_dir, fname),
                'mask_path': os.path.join(pix_dir, fname) if fname in pix_masks else None,
                'label':     1 if fname in pix_masks else 0,
            })
        mal = sum(1 for s in self.samples if s['label'] == 1)
        print(f"Dataset: {len(self.samples)} images | Malignant: {mal} | Benign: {len(self.samples)-mal}")

    def __len__(self):
        return len(self.samples)

    def _load_img(self, path):
        try:
            import tifffile
            arr = tifffile.imread(path)
            if arr.dtype != np.uint8:
                mx = arr.max()
                arr = (arr / mx * 255).astype(np.uint8) if mx > 0 else arr.astype(np.uint8)
            if arr.ndim == 2:
                arr = np.stack([arr]*3, axis=-1)
            elif arr.ndim == 3 and arr.shape[2] > 3:
                arr = arr[:, :, :3]
            return Image.fromarray(arr).convert('RGB')
        except Exception:
            try:
                return Image.open(path).convert('RGB')
            except Exception:
                return Image.new('RGB', self.img_size)

    def _load_mask(self, path):
        if path is None:
            return np.zeros((self.img_size[1], self.img_size[0]), dtype=np.float32)
        try:
            import tifffile
            arr = tifffile.imread(path)
            if arr.ndim == 3:
                arr = arr[:, :, 0]
            pil = Image.fromarray(arr).convert('L').resize(self.img_size, Image.NEAREST)
            arr = np.array(pil, dtype=np.float32)
            arr = (arr > 127).astype(np.float32) if arr.max() > 1 else (arr > 0.5).astype(np.float32)
            return arr
        except Exception:
            return np.zeros((self.img_size[1], self.img_size[0]), dtype=np.float32)

    def __getitem__(self, idx):
        s = self.samples[idx]
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

        img_pil = self._load_img(s['img_path'])

        img_seg = img_pil.resize(self.img_size, Image.LANCZOS)
        img_seg_t = torch.tensor(np.array(img_seg), dtype=torch.float32).permute(2, 0, 1) / 255.0
        img_seg_t = (img_seg_t - mean) / std

        img_cls = img_pil.resize(self.cls_size, Image.LANCZOS)
        img_cls_t = torch.tensor(np.array(img_cls), dtype=torch.float32).permute(2, 0, 1) / 255.0
        img_cls_t = (img_cls_t - mean) / std

        mask = self._load_mask(s['mask_path'])
        mask_t = torch.tensor(mask).unsqueeze(0)

        return {
            'img_seg': img_seg_t,
            'img_cls': img_cls_t,
            'mask':    mask_t,
            'label':   torch.tensor(s['label'], dtype=torch.long),
        }

# ---------------------------------------------------------------------------
# Train one model for one epoch
# ---------------------------------------------------------------------------
def train_epoch(seg_model, cls_model, loader, seg_opt, cls_opt,
                crit_seg, crit_cls, device, name):
    seg_model.train()
    cls_model.train()
    total_loss = 0.0
    for i, batch in enumerate(loader):
        img_seg = batch['img_seg'].to(device)
        img_cls = batch['img_cls'].to(device)
        masks   = batch['mask'].to(device)
        labels  = batch['label'].to(device)
        stages  = torch.zeros_like(labels)

        seg_opt.zero_grad(set_to_none=True)
        loss_seg = crit_seg(seg_model(img_seg), masks)
        loss_seg.backward()
        seg_opt.step()

        cls_opt.zero_grad(set_to_none=True)
        pc, ps = cls_model(img_cls)
        loss_cls = crit_cls(pc, labels) + crit_cls(ps, stages)
        loss_cls.backward()
        cls_opt.step()

        batch_loss = loss_seg.item() + loss_cls.item()
        total_loss += batch_loss
        print(f"\r  [{name}] Batch {i+1}/{len(loader)} "
              f"seg={loss_seg.item():.4f} cls={loss_cls.item():.4f}", end="", flush=True)
    print()
    return total_loss / max(len(loader), 1)

# ---------------------------------------------------------------------------
# Validate one model
# ---------------------------------------------------------------------------
def validate(seg_model, cls_model, loader, crit_seg, crit_cls, device, name):
    seg_model.eval()
    cls_model.eval()
    total_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for batch in loader:
            img_seg = batch['img_seg'].to(device)
            img_cls = batch['img_cls'].to(device)
            masks   = batch['mask'].to(device)
            labels  = batch['label'].to(device)
            stages  = torch.zeros_like(labels)
            loss_seg = crit_seg(seg_model(img_seg), masks)
            pc, ps   = cls_model(img_cls)
            loss_cls = crit_cls(pc, labels) + crit_cls(ps, stages)
            total_loss += loss_seg.item() + loss_cls.item()
            correct    += (torch.argmax(pc, 1) == labels).sum().item()
            total      += labels.size(0)
    acc = correct / max(total, 1) * 100
    avg = total_loss / max(len(loader), 1)
    print(f"  [{name}] Val loss={avg:.4f} | Accuracy={acc:.1f}% ({correct}/{total})")
    return avg, acc

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def train_all(epochs=10, batch_size=10):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"=== Unified GPU Training Pipeline ===")
    print(f"Device: {device}"
          + (f" ({torch.cuda.get_device_name(0)})" if device == "cuda" else ""))
    print(f"Batch size: {batch_size} | Epochs: {epochs}")
    print()

    dataset = MediaTIFFDataset(IMG_DIR, PIX_DIR)
    n_train = int(len(dataset) * 0.8)
    n_val   = len(dataset) - n_train
    gen = torch.Generator().manual_seed(42)
    train_ds, val_ds = random_split(dataset, [n_train, n_val], generator=gen)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  drop_last=True,  num_workers=0, pin_memory=(device=="cuda"))
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, drop_last=False, num_workers=0, pin_memory=(device=="cuda"))
    print(f"Train: {n_train} | Val: {n_val} | Train batches: {len(train_loader)}")
    print()

    crit_seg = nn.BCELoss()
    crit_cls = nn.CrossEntropyLoss()

    # ---- ResNet50 ----
    resnet_seg = UNet().to(device)
    resnet_cls = ResNetClassifier().to(device)
    rn_seg_opt = optim.Adam(resnet_seg.parameters(), lr=1e-4, weight_decay=1e-5)
    rn_cls_opt = optim.Adam(resnet_cls.parameters(), lr=1e-4, weight_decay=1e-5)
    rn_seg_sch = optim.lr_scheduler.StepLR(rn_seg_opt, step_size=3, gamma=0.5)
    rn_cls_sch = optim.lr_scheduler.StepLR(rn_cls_opt, step_size=3, gamma=0.5)

    # ---- VGG16 ---- (init but move to CPU first to save VRAM until needed)
    vgg_seg = UNet()
    vgg_cls = VGGClassifier()
    vg_seg_opt = optim.Adam(vgg_seg.parameters(), lr=1e-4, weight_decay=1e-5)
    vg_cls_opt = optim.Adam(vgg_cls.parameters(), lr=1e-4, weight_decay=1e-5)
    vg_seg_sch = optim.lr_scheduler.StepLR(vg_seg_opt, step_size=3, gamma=0.5)
    vg_cls_sch = optim.lr_scheduler.StepLR(vg_cls_opt, step_size=3, gamma=0.5)

    best_rn_acc, best_vg_acc = 0.0, 0.0

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        print(f"=== Epoch {epoch}/{epochs} ===")

        # -- Train ResNet50 (GPU) --
        resnet_seg.to(device); resnet_cls.to(device)
        rn_loss = train_epoch(resnet_seg, resnet_cls, train_loader,
                              rn_seg_opt, rn_cls_opt, crit_seg, crit_cls, device, "ResNet50")
        rn_seg_sch.step(); rn_cls_sch.step()
        _, rn_acc = validate(resnet_seg, resnet_cls, val_loader, crit_seg, crit_cls, device, "ResNet50")

        # Offload ResNet50 to CPU + free VRAM before VGG16
        resnet_seg.cpu(); resnet_cls.cpu()
        torch.cuda.empty_cache()

        # -- Train VGG16 (GPU) --
        vgg_seg.to(device); vgg_cls.to(device)
        vg_loss = train_epoch(vgg_seg, vgg_cls, train_loader,
                              vg_seg_opt, vg_cls_opt, crit_seg, crit_cls, device, "VGG16")
        vg_seg_sch.step(); vg_cls_sch.step()
        _, vg_acc = validate(vgg_seg, vgg_cls, val_loader, crit_seg, crit_cls, device, "VGG16")

        # Offload VGG16 to CPU + free VRAM
        vgg_seg.cpu(); vgg_cls.cpu()
        torch.cuda.empty_cache()

        elapsed = time.time() - t0
        print(f"  Epoch time: {elapsed:.1f}s")

        # Save best weights (move back to GPU briefly to save)
        if rn_acc > best_rn_acc:
            best_rn_acc = rn_acc
            os.makedirs(os.path.dirname(Config.RESNET50_SEG_PATH), exist_ok=True)
            resnet_seg.to(device)
            save_model(resnet_seg, Config.RESNET50_SEG_PATH)
            resnet_cls.to(device)
            save_model(resnet_cls, Config.RESNET50_CLS_PATH)
            resnet_seg.cpu(); resnet_cls.cpu()
            torch.cuda.empty_cache()
            print(f"  [ResNet50] Best acc {rn_acc:.1f}% -> weights saved.")

        if vg_acc > best_vg_acc:
            best_vg_acc = vg_acc
            os.makedirs(os.path.dirname(Config.VGG16_SEG_PATH), exist_ok=True)
            vgg_seg.to(device)
            save_model(vgg_seg, Config.VGG16_SEG_PATH)
            vgg_cls.to(device)
            save_model(vgg_cls, Config.VGG16_CLS_PATH)
            vgg_seg.cpu(); vgg_cls.cpu()
            torch.cuda.empty_cache()
            print(f"  [VGG16] Best acc {vg_acc:.1f}% -> weights saved.")

        print()

    print(f"=== Training Complete ===")
    print(f"Best ResNet50 Accuracy: {best_rn_acc:.1f}%")
    print(f"Best VGG16 Accuracy:    {best_vg_acc:.1f}%")

if __name__ == "__main__":
    train_all(epochs=10, batch_size=10)
