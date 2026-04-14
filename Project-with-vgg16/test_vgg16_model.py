import torch
import torch.nn as nn
from src.classification import BreastCancerClassifier

def test_model():
    print("Testing VGG16 BreastCancerClassifier...")
    
    # Instantiate model
    num_classes = 2
    num_stages = 4
    model = BreastCancerClassifier(num_classes=num_classes, num_stages=num_stages)
    
    # Check backbone type
    from torchvision import models
    is_vgg = isinstance(model.backbone, models.VGG)
    print(f"Backbone is VGG: {is_vgg}")
    
    if not is_vgg:
         print("Error: Backbone is not VGG!")
         return
    
    # Check if classifier[6] is Identity
    is_identity = isinstance(model.backbone.classifier[6], nn.Identity)
    print(f"Backbone classifier[6] is Identity: {is_identity}")
    
    # Create dummy input: [Batch, Channels, Height, Width]
    dummy_input = torch.randn(2, 3, 224, 224)
    
    # Forward pass
    model.eval()
    with torch.no_grad():
        binary_out, stage_out = model(dummy_input)
    
    print(f"Input shape: {dummy_input.shape}")
    print(f"Binary output shape: {binary_out.shape} (Expected: [2, {num_classes}])")
    print(f"Stage output shape: {stage_out.shape} (Expected: [2, {num_stages}])")
    
    assert binary_out.shape == (2, num_classes)
    assert stage_out.shape == (2, num_stages)
    
    print("\nSUCCESS: VGG16 model instantiated and forward pass completed successfully!")

if __name__ == "__main__":
    test_model()
