import torch
import torch.nn as nn
from torchvision import models

class BreastCancerClassifier(nn.Module):
    """
    VGG16-based classifier with multiple heads for binary classification and cancer staging.
    Designed for high-accuracy tumor detection and staging in breast cancer images.
    """
    def __init__(self, num_classes=2, num_stages=4):
        super(BreastCancerClassifier, self).__init__()
        
        # Base model (VGG16 with pre-trained weights)
        self.backbone = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
        
        # Remove only the last layer of the classifier to keep pre-trained FC features (4096-dimensional)
        # The VGG16 classifier has: [Linear(25088, 4096), ReLU, Dropout, Linear(4096, 4096), ReLU, Dropout, Linear(4096, 1000)]
        # We replace the last Linear(4096, 1000) with Identity to get the 4096 features.
        num_ftrs = self.backbone.classifier[6].in_features 
        self.backbone.classifier[6] = nn.Identity()
        
        # High-performance heads with BatchNorm and Dropout
        
        # Benign/Malignant classification head
        self.head_binary = nn.Sequential(
            nn.Linear(num_ftrs, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
        
        # Cancer Staging head (Stage 1-4)
        self.head_stage = nn.Sequential(
            nn.Linear(num_ftrs, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, num_stages)
        )

    def forward(self, x):
        """
        Forward pass returns both binary classification and staging predictions.
        """
        # Feature extraction via VGG16 backbone
        features = self.backbone(x)
        
        # Multi-head output
        binary_out = self.head_binary(features)
        stage_out = self.head_stage(features)
        
        return binary_out, stage_out
