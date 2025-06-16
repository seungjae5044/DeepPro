import torch
import torch.nn as nn
import torch.nn.functional as F

class ResNeXtShuffleNetModel(nn.Module):
    def __init__(self, num_classes=15, groups=8, dropout=0.5, fc_dropout=0.7):
        super(ResNeXtShuffleNetModel, self).__init__()
        
        self.groups = groups
        
        # Stem block
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # Hybrid ResNeXt-ShuffleNet stages
        self.stage1 = self._make_stage(64, 128, 2, stride=1)
        self.stage2 = self._make_stage(128, 256, 3, stride=2)
        self.stage3 = self._make_stage(256, 512, 3, stride=2)
        
        # Global average pooling and classifier
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(512, num_classes)
        
        self._initialize_weights()
    
    def _make_stage(self, in_channels, out_channels, num_blocks, stride=1):
        layers = []
        
        # First block handles stride and channel change
        layers.append(ResNeXtShuffleBlock(
            in_channels, out_channels, self.groups, stride=stride,
            downsample=(stride != 1 or in_channels != out_channels)
        ))
        
        # Remaining blocks
        for _ in range(1, num_blocks):
            layers.append(ResNeXtShuffleBlock(
                out_channels, out_channels, self.groups
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
        x = self.fc(x)
        
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


class ResNeXtShuffleBlock(nn.Module):
    """Hybrid block combining ResNeXt grouped convolution with ShuffleNet channel shuffle"""
    
    def __init__(self, in_channels, out_channels, groups, stride=1, downsample=False):
        super(ResNeXtShuffleBlock, self).__init__()
        
        self.groups = groups
        self.stride = stride
        
        # Pre-activation
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.relu1 = nn.ReLU(inplace=True)
        
        # ResNeXt-style grouped 1x1 conv (bottleneck)
        mid_channels = out_channels // 2
        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=1, 
                              groups=groups, bias=False)
        
        # Channel shuffle (ShuffleNet key innovation)
        self.shuffle_channels = mid_channels
        
        # Pre-activation for middle layer
        self.bn2 = nn.BatchNorm2d(mid_channels)
        self.relu2 = nn.ReLU(inplace=True)
        
        # Depthwise 3x3 conv (ShuffleNet style)
        self.conv2 = nn.Conv2d(mid_channels, mid_channels, kernel_size=3, 
                              stride=stride, padding=1, groups=mid_channels, bias=False)
        
        # Pre-activation for final layer
        self.bn3 = nn.BatchNorm2d(mid_channels)
        self.relu3 = nn.ReLU(inplace=True)
        
        # Final 1x1 grouped conv
        self.conv3 = nn.Conv2d(mid_channels, out_channels, kernel_size=1, 
                              groups=groups, bias=False)
        
        # Shortcut connection
        self.shortcut = nn.Sequential()
        if downsample:
            self.shortcut = nn.Sequential(
                nn.BatchNorm2d(in_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels, out_channels, kernel_size=1, 
                         stride=stride, bias=False)
            )
    
    def channel_shuffle(self, x):
        """ShuffleNet channel shuffle operation"""
        batch_size, channels, height, width = x.size()
        channels_per_group = channels // self.groups
        
        # Reshape to (batch_size, groups, channels_per_group, height, width)
        x = x.view(batch_size, self.groups, channels_per_group, height, width)
        
        # Transpose to (batch_size, channels_per_group, groups, height, width)
        x = x.transpose(1, 2).contiguous()
        
        # Flatten back to (batch_size, channels, height, width)
        x = x.view(batch_size, channels, height, width)
        
        return x
    
    def forward(self, x):
        # Store input for residual connection
        residual = x
        
        # Pre-activation and first grouped conv
        out = self.bn1(x)
        out = self.relu1(out)
        out = self.conv1(out)
        
        # Channel shuffle (key ShuffleNet innovation)
        out = self.channel_shuffle(out)
        
        # Depthwise convolution
        out = self.bn2(out)
        out = self.relu2(out)
        out = self.conv2(out)
        
        # Final grouped convolution
        out = self.bn3(out)
        out = self.relu3(out)
        out = self.conv3(out)
        
        # Add shortcut
        residual = self.shortcut(residual)
        out += residual
        
        return out