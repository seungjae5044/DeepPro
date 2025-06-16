import torch
import torch.nn as nn

class SEResNeXtModel(nn.Module):
    def __init__(self, num_classes=15, cardinality=8, base_width=64, reduction=16, 
                 dropout=0.5, fc_dropout=0.7, channel_multiplier=2):
        super(SEResNeXtModel, self).__init__()
        
        self.cardinality = cardinality
        self.base_width = base_width
        self.reduction = reduction
        self.channel_multiplier = channel_multiplier  # 채널 수 증가 배수
        
        # Stem block
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # Progressive stages with increased channels
        # 채널 수를 기존보다 크게 설정
        self.stage1 = self._make_stage(128, 256 * channel_multiplier, 2, stride=1)
        self.stage2 = self._make_stage(256 * channel_multiplier, 512 * channel_multiplier, 3, stride=2)
        self.stage3 = self._make_stage(512 * channel_multiplier, 1024 * channel_multiplier, 3, stride=2)
        
        # Classification head
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(1024 * channel_multiplier, 512)
        self.fc1_relu = nn.ReLU(inplace=True)
        self.fc_dropout = nn.Dropout(fc_dropout)
        self.fc2 = nn.Linear(512, num_classes)
        
        self._initialize_weights()
    
    def _make_stage(self, in_channels, out_channels, num_blocks, stride=1):
        layers = []
        
        # First block handles stride and channel change
        layers.append(SEResNeXtBlock(
            in_channels, out_channels, self.cardinality, self.base_width,
            self.reduction, stride=stride,
            downsample=(stride != 1 or in_channels != out_channels)
        ))
        
        # Remaining blocks
        for _ in range(1, num_blocks):
            layers.append(SEResNeXtBlock(
                out_channels, out_channels, self.cardinality, self.base_width,
                self.reduction
            ))
        
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.stem(x)
        
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        
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
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


class SEResNeXtBlock(nn.Module):
    """SE-ResNeXt block combining grouped convolutions with squeeze-and-excitation"""
    
    def __init__(self, in_channels, out_channels, cardinality, base_width, reduction=16, 
                 stride=1, downsample=False):
        super(SEResNeXtBlock, self).__init__()
        
        # Calculate group width for ResNeXt
        group_width = int(out_channels / 4) * base_width // 64
        
        # Pre-activation structure
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.relu1 = nn.ReLU(inplace=True)
        
        # ResNeXt grouped convolution layers
        # 1x1 conv to reduce channels
        self.conv1 = nn.Conv2d(in_channels, group_width * cardinality, 
                              kernel_size=1, bias=False)
        
        self.bn2 = nn.BatchNorm2d(group_width * cardinality)
        self.relu2 = nn.ReLU(inplace=True)
        
        # 3x3 grouped conv (핵심 ResNeXt 특징)
        self.conv2 = nn.Conv2d(group_width * cardinality, group_width * cardinality,
                              kernel_size=3, stride=stride, padding=1,
                              groups=cardinality, bias=False)
        
        self.bn3 = nn.BatchNorm2d(group_width * cardinality)
        self.relu3 = nn.ReLU(inplace=True)
        
        # 1x1 conv to expand channels
        self.conv3 = nn.Conv2d(group_width * cardinality, out_channels,
                              kernel_size=1, bias=False)
        
        # SE block (핵심 SENet 특징)
        self.se_block = SEBlock(out_channels, reduction)
        
        # Shortcut connection
        self.shortcut = nn.Sequential()
        if downsample:
            self.shortcut = nn.Sequential(
                nn.BatchNorm2d(in_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels, out_channels, kernel_size=1, 
                         stride=stride, bias=False)
            )
    
    def forward(self, x):
        # Store input for residual connection
        residual = x
        
        # Pre-activation and ResNeXt grouped convolutions
        out = self.bn1(x)
        out = self.relu1(out)
        out = self.conv1(out)
        
        out = self.bn2(out)
        out = self.relu2(out)
        out = self.conv2(out)  # Grouped convolution
        
        out = self.bn3(out)
        out = self.relu3(out)
        out = self.conv3(out)
        
        # Apply SE block (채널 어텐션)
        out = self.se_block(out)
        
        # Add shortcut
        residual = self.shortcut(residual)
        out += residual
        
        return out


class SEBlock(nn.Module):
    """Squeeze-and-Excitation block"""
    
    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()
        
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        batch_size, channels, _, _ = x.size()
        
        # Squeeze: Global average pooling
        squeeze = self.global_avg_pool(x).view(batch_size, channels)
        
        # Excitation: FC layers
        excitation = self.fc(squeeze).view(batch_size, channels, 1, 1)
        
        # Scale: Element-wise multiplication
        return x * excitation