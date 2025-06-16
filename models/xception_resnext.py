import torch
import torch.nn as nn
import torch.nn.functional as F

class XceptionResNeXtModel(nn.Module):
    def __init__(self, num_classes=15, cardinality=8, base_width=64, dropout=0.3, fc_dropout=0.5):
        super(XceptionResNeXtModel, self).__init__()
        
        self.cardinality = cardinality
        self.base_width = base_width
        
        # Stem block with pre-activation style
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # Hybrid blocks - mixing Xception and ResNeXt concepts
        self.layer1 = self._make_layer(64, 64, 2, stride=1)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)
        self.layer4 = self._make_layer(256, 512, 2, stride=2)
        
        # Classification head
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(512, 256)
        self.fc1_relu = nn.ReLU(inplace=True)
        self.fc_dropout = nn.Dropout(fc_dropout)
        self.fc2 = nn.Linear(256, num_classes)
        
        self._initialize_weights()
    
    def _make_layer(self, in_channels, out_channels, blocks, stride=1):
        layers = []
        
        # First block handles stride and channel change
        layers.append(XceptionResNeXtBlock(
            in_channels, out_channels, self.cardinality, self.base_width, 
            stride=stride, downsample=(stride != 1 or in_channels != out_channels)
        ))
        
        # Remaining blocks
        for _ in range(1, blocks):
            layers.append(XceptionResNeXtBlock(
                out_channels, out_channels, self.cardinality, self.base_width
            ))
        
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.stem(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc1(x)
        x = self.fc1_relu(x)
        x = self.fc_dropout(x)
        x = self.fc2(x)
        
        return x
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


class XceptionResNeXtBlock(nn.Module):
    """Hybrid block combining Xception's depthwise separable conv with ResNeXt's grouped conv"""
    
    def __init__(self, in_channels, out_channels, cardinality, base_width, stride=1, downsample=False):
        super(XceptionResNeXtBlock, self).__init__()
        
        # Calculate group width for ResNeXt
        D = int(out_channels / 4)  # Bottleneck dimension
        group_width = cardinality * D
        
        # Pre-activation structure: BN → ReLU → Conv
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.relu1 = nn.ReLU(inplace=True)
        
        # First 1x1 conv (like ResNeXt bottleneck)
        self.conv1 = nn.Conv2d(in_channels, group_width, kernel_size=1, bias=False)
        
        # Hybrid middle layer: Xception-style depthwise + ResNeXt-style grouped
        self.bn2 = nn.BatchNorm2d(group_width)
        self.relu2 = nn.ReLU(inplace=True)
        
        # Depthwise separable convolution (Xception)
        self.depthwise_conv = nn.Conv2d(
            group_width, group_width, kernel_size=3, stride=stride, 
            padding=1, groups=group_width, bias=False
        )
        self.pointwise_conv = nn.Conv2d(group_width, group_width, kernel_size=1, bias=False)
        
        # Final 1x1 conv to output channels
        self.bn3 = nn.BatchNorm2d(group_width)
        self.relu3 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(group_width, out_channels, kernel_size=1, bias=False)
        
        # Shortcut connection
        self.shortcut = nn.Sequential()
        if downsample:
            self.shortcut = nn.Sequential(
                nn.BatchNorm2d(in_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)
            )
    
    def forward(self, x):
        # Store input for residual connection
        residual = x
        
        # Pre-activation
        out = self.bn1(x)
        out = self.relu1(out)
        
        # First 1x1 conv
        out = self.conv1(out)
        
        # Hybrid depthwise separable + grouped conv
        out = self.bn2(out)
        out = self.relu2(out)
        out = self.depthwise_conv(out)
        out = self.pointwise_conv(out)
        
        # Final 1x1 conv
        out = self.bn3(out)
        out = self.relu3(out)
        out = self.conv3(out)
        
        # Add shortcut
        residual = self.shortcut(residual)
        out += residual
        
        return out