import torch
import torch.nn as nn
from torchvision import models

class BreastCancerClassifier(nn.Module):
    def __init__(self, num_classes=2, num_stages=4):
        super(BreastCancerClassifier, self).__init__()
        
        # Base model (ResNet50)
        self.backbone = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        # Remove original fully connected layer
        num_ftrs = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        
        # Benign/Malignant head
        self.head_binary = nn.Sequential(
            nn.Linear(num_ftrs, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
        
        # Cancer Staging head
        self.head_stage = nn.Sequential(
            nn.Linear(num_ftrs, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_stages)
        )

    def forward(self, x):
        features = self.backbone(x)
        binary_out = self.head_binary(features)
        stage_out = self.head_stage(features)
        return binary_out, stage_out
