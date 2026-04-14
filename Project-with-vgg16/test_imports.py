import sys
import os
sys.path.append(os.getcwd())

print("Testing utils.config...")
from utils.config import Config
print("OK")

print("Testing src.segmentation...")
from src.segmentation import UNet
print("OK")

print("Testing src.classification...")
from src.classification import BreastCancerClassifier
print("OK")

print("Testing src.preprocessing...")
from src.preprocessing import get_transforms
print("OK")

print("Testing src.dataset...")
from src.dataset import BreastCancerDataset
print("OK")

print("Testing src.train...")
from src.train import train_pipeline
print("OK")
