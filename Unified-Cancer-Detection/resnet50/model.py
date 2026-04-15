import torch
import torch.nn as nn
from torchvision import models

class BreastCancerClassifier(nn.Module):
    """
    ResNet50-based classifier with multiple heads for binary classification and cancer staging.
    Re-implemented from scratch to match VGG16 parity.
    """
    def __init__(self, num_classes=2, num_stages=4):
        super(BreastCancerClassifier, self).__init__()
        
        # Base model (ResNet50 with pre-trained weights)
        self.backbone = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        
        # Remove original terminal FC layer (2048 in_features)
        num_ftrs = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        
        # Binary Classification Head
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
        Multi-head forward pass.
        """
        features = self.backbone(x)
        binary_out = self.head_binary(features)
        stage_out = self.head_stage(features)
        return binary_out, stage_out
