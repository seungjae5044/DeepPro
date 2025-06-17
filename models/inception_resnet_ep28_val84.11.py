import torch
import torch.nn as nn



class SmallInceptionResNetA(nn.Module):
    def __init__(self, in_channels=64, output_ch = 128, scale=0.3):
        super().__init__()
        self.scale = scale
        mid_ch = in_channels//4

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, mid_ch, 1),
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels, mid_ch, 1),
            nn.Conv2d(mid_ch, mid_ch, 3, padding=1),
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels, mid_ch, 1),
            nn.Conv2d(mid_ch, mid_ch//2 * 3, 3, padding=1),
            nn.Conv2d(mid_ch//2 * 3, mid_ch * 2, 3, padding=1),
        )
        self.downsample = nn.Conv2d(in_channels, output_ch, 1)
        self.conv_linear = nn.Conv2d(in_channels, output_ch, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        b1 = self.conv1(x)
        b2 = self.conv2(x)
        b3 = self.conv3(x)
        out = torch.cat([b1, b2, b3], dim=1)
        out = self.conv_linear(out)
        x = self.downsample(x)
        return self.relu(x + self.scale * out)


class InceptionResNetModel(nn.Module):
    def __init__(self, num_classes=15, init_weights=True, dropout=0.3, fc_dropout=0.5):
        super(InceptionResNetModel, self).__init__()
        
        # Stem block
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            #nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            #nn.BatchNorm2d(64),
            #nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # Inception-ResNet blocks
        #self.inception1 = self._make_inception_resnet_block(64, [16, 24, 32, 16])
        #self.inception2 = self._make_inception_resnet_block(88, [32, 48, 64, 32])
        self.inception1 = SmallInceptionResNetA(64, 88)
        self.inception2 = SmallInceptionResNetA(88, 176)
        
        # Transition layers with Depthwise Separable Convolution
        self.transition1 = self._make_depthwise_separable_conv(176, 128, kernel_size=3, stride=2, padding=1)
        
        #self.inception3 = self._make_inception_resnet_block(128, [48, 64, 96, 48])
        #self.inception4 = self._make_inception_resnet_block(256, [64, 96, 128, 64])
        self.inception3 = SmallInceptionResNetA(128, 256)
        self.inception4 = SmallInceptionResNetA(256, 352)
        
        self.transition2 = self._make_depthwise_separable_conv(352, 256, kernel_size=3, stride=2, padding=1)
        
        # Classification head
        self.final_bn = nn.BatchNorm2d(256)
        self.final_relu = nn.ReLU()
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.dropout1 = nn.Dropout(dropout)
        self.fc1 = nn.Linear(256, 128)
        self.relu_fc = nn.ReLU()
        self.dropout2 = nn.Dropout(fc_dropout)
        self.fc2 = nn.Linear(128, num_classes)
        
        if init_weights:
            self._initialize_weights()
    
    def _make_depthwise_separable_conv(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        """Depthwise Separable Convolution with Pre-Activation"""
        return nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            # Depthwise convolution
            nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, 
                     stride=stride, padding=padding, groups=in_channels),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            # Pointwise convolution
            nn.Conv2d(in_channels, out_channels, kernel_size=1)
        )
    
    def _make_inception_resnet_block(self, in_channels, channels):
        """Inception-ResNet block with residual connection"""
        # Inception branches
        branch1 = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels, channels[0], kernel_size=1)
        )
        
        branch2 = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels, channels[1], kernel_size=1),
            nn.BatchNorm2d(channels[1]),
            nn.ReLU(),
            nn.Conv2d(channels[1], channels[1], kernel_size=3, padding=1)
        )
        
        branch3 = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels, channels[2], kernel_size=1),
            nn.BatchNorm2d(channels[2]),
            nn.ReLU(),
            nn.Conv2d(channels[2], channels[2], kernel_size=3, padding=1),
            nn.BatchNorm2d(channels[2]),
            nn.ReLU(),
            nn.Conv2d(channels[2], channels[2], kernel_size=3, padding=1)
        )
        
        branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels, channels[3], kernel_size=1)
        )
        

        # Output channels after concatenation
        total_channels = sum(channels)
        
        # 1x1 conv to match input channels for residual connection
        channel_adjust = None
        if total_channels != in_channels:
            channel_adjust = nn.Conv2d(in_channels, total_channels, kernel_size=1)
        
        return nn.ModuleDict({
            'branches': nn.ModuleList([branch1, branch2, branch3, branch4]),
            'channel_adjust': channel_adjust
        })
    
    def _forward_inception_resnet(self, x, inception_resnet_block):
        """Forward pass with residual connection"""
        # Get branches and channel adjustment layer
        branches = inception_resnet_block['branches']
        channel_adjust = inception_resnet_block['channel_adjust']
        
        # Forward through inception branches
        branch_outputs = []
        for branch in branches:
            branch_outputs.append(branch(x))
        
        # Concatenate branch outputs
        inception_out = torch.cat(branch_outputs, dim=1)
        
        # Prepare residual connection
        residual = x
        if channel_adjust is not None:
            residual = channel_adjust(residual)
        
        # Add residual connection
        out = inception_out + residual
        
        return out
    
    def forward(self, x):
        x = self.stem(x)
        
        #x = self._forward_inception_resnet(x, self.inception1)
        #x = self._forward_inception_resnet(x, self.inception2)
        x = self.inception1(x)       
        x = self.inception2(x)
        x = self.transition1(x)
        
        #x = self._forward_inception_resnet(x, self.inception3)
        #x = self._forward_inception_resnet(x, self.inception4)
        x = self.inception3(x)
        x = self.inception4(x)
        x = self.transition2(x)
        
        # Final activation for pre-activation structure
        x = self.final_bn(x)
        x = self.final_relu(x)
        
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout1(x)
        x = self.fc1(x)
        x = self.relu_fc(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        
        return x
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)