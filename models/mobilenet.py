import torch
import torch.nn as nn


class DepthwiseSeparableConv(nn.Module):
    """Depthwise Separable Convolution with Preactivation"""
    def __init__(self, in_channels, out_channels, stride=1):
        super(DepthwiseSeparableConv, self).__init__()
        
        # Preactivation: BN -> ReLU -> Conv
        self.preact_bn = nn.BatchNorm2d(in_channels)
        self.preact_relu = nn.ReLU(inplace=True)
        
        # Depthwise convolution
        self.depthwise = nn.Conv2d(
            in_channels, in_channels, kernel_size=3, stride=stride, 
            padding=1, groups=in_channels, bias=False
        )
        
        # Pointwise convolution (1x1)
        self.pointwise = nn.Conv2d(
            in_channels, out_channels, kernel_size=1, stride=1, 
            padding=0, bias=False
        )
        
        # Batch normalization after pointwise
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        # Preactivation
        x = self.preact_bn(x)
        x = self.preact_relu(x)
        
        # Depthwise separable convolution
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        x = self.relu(x)
        
        return x


class MobileNet(nn.Module):
    """Pure MobileNet with Preactivation and Weight Initialization"""
    def __init__(self, num_classes=15, width_multiplier=1.0, init_weights=True):
        super(MobileNet, self).__init__()
        
        # First layer: standard convolution
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, int(32 * width_multiplier), kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(int(32 * width_multiplier)),
            nn.ReLU(inplace=True)
        )
        
        # MobileNet layers configuration
        # (in_channels, out_channels, stride)
        layers_config = [
            (32, 64, 1),    # 48x48
            (64, 128, 2),   # 24x24
            (128, 128, 1),  # 24x24
            (128, 256, 2),  # 12x12
            (256, 256, 1),  # 12x12
            (256, 512, 2),  # 6x6
            (512, 512, 1),  # 6x6
            (512, 512, 1),  # 6x6
            (512, 512, 1),  # 6x6
            (512, 512, 1),  # 6x6
            (512, 1024, 2), # 3x3
            (1024, 1024, 1) # 3x3
        ]
        
        # Build depthwise separable convolution layers
        self.layers = nn.ModuleList()
        for in_ch, out_ch, stride in layers_config:
            in_ch = int(in_ch * width_multiplier)
            out_ch = int(out_ch * width_multiplier)
            self.layers.append(DepthwiseSeparableConv(in_ch, out_ch, stride))
        
        # Global average pooling and classifier
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(int(1024 * width_multiplier), num_classes)
        
        if init_weights:
            self._initialize_weights()
    
    def forward(self, x):
        x = self.conv1(x)
        
        for layer in self.layers:
            x = layer(x)
        
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        
        return x
    
    def _initialize_weights(self):
        """Initialize weights using He initialization for ReLU"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)


class MobileNetModel(MobileNet):
    """Wrapper class for compatibility with existing code"""
    def __init__(self, num_classes=15, **kwargs):
        super(MobileNetModel, self).__init__(num_classes=num_classes, **kwargs)