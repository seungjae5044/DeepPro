import torch
from models.xception_resnext import XceptionResNeXtModel

# Create model instance
model = XceptionResNeXtModel(num_classes=15)

# Test with dummy input
x = torch.randn(1, 3, 48, 48)
output = model(x)

print(f"Input shape: {x.shape}")
print(f"Output shape: {output.shape}")
print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
print("Model created successfully!")

# Print model architecture
print("\nModel architecture:")
print(model)