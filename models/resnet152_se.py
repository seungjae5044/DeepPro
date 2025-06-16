import torch
import torch.nn as nn


class SEBlock(nn.Module):
    """Squeeze-and-Excitation Block"""
    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(channels, channels // reduction)
        self.fc2 = nn.Linear(channels // reduction, channels)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.global_pool(x).view(b, c)
        y = self.fc1(y)
        y = self.relu(y)
        y = self.fc2(y)
        y = self.sigmoid(y)
        y = y.view(b, c, 1, 1)
        return x * y


class Bottleneck(nn.Module):
    """ResNet-152 Bottleneck Block with SE Module"""
    expansion = 4
    
    def __init__(self, in_channels, out_channels, stride=1, downsample=None, reduction=16):
        super(Bottleneck, self).__init__()
        
        # 1x1 conv
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        # 3x3 conv
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, 
                              stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # 1x1 conv (expansion)
        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion, 
                              kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)
        
        # SE Block
        self.se = SEBlock(out_channels * self.expansion, reduction)
        
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        
    def forward(self, x):
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        
        out = self.conv3(out)
        out = self.bn3(out)
        
        # Apply SE Block
        out = self.se(out)
        
        if self.downsample is not None:
            identity = self.downsample(x)
        
        out += identity
        out = self.relu(out)
        
        return out


class ResNet50SE(nn.Module):
    """ResNet-50 with SE Blocks optimized for 48x48 input"""
    
    def __init__(self, num_classes=15, reduction=16, init_weights=True):
        super(ResNet50SE, self).__init__()
        
        # Initial convolution - optimized for 48x48
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=1, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # ResNet-50 layer configuration: [3, 4, 6, 3]
        self.layer1 = self._make_layer(64, 64, 3, stride=1, reduction=reduction)
        self.layer2 = self._make_layer(256, 128, 4, stride=2, reduction=reduction)
        self.layer3 = self._make_layer(512, 256, 6, stride=2, reduction=reduction)
        self.layer4 = self._make_layer(1024, 512, 3, stride=2, reduction=reduction)
        
        # Classification head
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(2048, num_classes)
        
        if init_weights:
            self._initialize_weights()
    
    def _make_layer(self, in_channels, out_channels, blocks, stride=1, reduction=16):
        downsample = None
        if stride != 1 or in_channels != out_channels * Bottleneck.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * Bottleneck.expansion,
                         kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * Bottleneck.expansion),
            )
        
        layers = []
        layers.append(Bottleneck(in_channels, out_channels, stride, downsample, reduction))
        
        in_channels = out_channels * Bottleneck.expansion
        for _ in range(1, blocks):
            layers.append(Bottleneck(in_channels, out_channels, reduction=reduction))
        
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)  # 48x48 -> 24x24
        
        x = self.layer1(x)   # 24x24, 256 channels
        x = self.layer2(x)   # 12x12, 512 channels  
        x = self.layer3(x)   # 6x6, 1024 channels
        x = self.layer4(x)   # 3x3, 2048 channels
        
        x = self.avgpool(x)  # 1x1, 2048 channels
        x = torch.flatten(x, 1)
        x = self.fc(x)
        
        return x
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)


class ResNet50SEModel(ResNet50SE):
    """Wrapper class for compatibility"""
    def __init__(self, num_classes=15, **kwargs):
        super(ResNet50SEModel, self).__init__(num_classes=num_classes, **kwargs)