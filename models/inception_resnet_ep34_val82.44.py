import torch
import torch.nn as nn
import torch.nn.functional as F

class SmallInceptionResNetA(nn.Module):
    def __init__(self, in_channels, output_ch, scale=0.1):
        super().__init__()
        self.scale = scale
        mid_ch = in_channels // 4

        self.conv1 = nn.Conv2d(in_channels, mid_ch, kernel_size=1)

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels, mid_ch, kernel_size=1),
            nn.Conv2d(mid_ch, mid_ch, kernel_size=3, padding=1)
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels, mid_ch, kernel_size=1),
            nn.Conv2d(mid_ch, mid_ch * 3 // 2, kernel_size=3, padding=1),
            nn.Conv2d(mid_ch * 3 // 2, mid_ch * 2, kernel_size=3, padding=1)
        )

        self.conv_linear = nn.Conv2d(mid_ch * 4, output_ch, kernel_size=1)
        self.shortcut = nn.Conv2d(in_channels, output_ch, kernel_size=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        b1 = self.conv1(x)
        b2 = self.conv2(x)
        b3 = self.conv3(x)
        out = torch.cat([b1, b2, b3], dim=1)
        out = self.conv_linear(out)
        residual = self.shortcut(x)
        return self.relu(residual + self.scale * out)

class InceptionResNetModel(nn.Module):
    def __init__(self, num_classes=15, init_weights=True, dropout=0.3, fc_dropout=0.5):
        super().__init__()

        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            #nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.block1 = SmallInceptionResNetA(64, 128)
        self.down1 = nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1)

        self.block2 = SmallInceptionResNetA(128, 256)
        self.down2 = nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1)

        self.block3 = SmallInceptionResNetA(256, 384)
        self.down3 = nn.Conv2d(384, 384, kernel_size=3, stride=2, padding=1)

        self.block4 = SmallInceptionResNetA(384, 512)

        # 1x1 conv to normalize channel dims
        self.proj1 = nn.Conv2d(128, 128, kernel_size=1)
        self.proj2 = nn.Conv2d(256, 128, kernel_size=1)
        self.proj3 = nn.Conv2d(384, 128, kernel_size=1)
        self.proj4 = nn.Conv2d(512, 128, kernel_size=1)

        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Sequential(
            nn.Linear(128 * 4 * 1 * 1, 128),
            nn.ReLU(),
            nn.Dropout(fc_dropout),
            nn.Linear(128, num_classes)
        )
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.stem(x)          # (B, 64, H/2, W/2)
        f1 = self.block1(x)       # (B, 128, H/2, W/2)
        x = self.down1(f1)        # (B, 128, H/4, W/4)

        f2 = self.block2(x)       # (B, 256, H/4, W/4)
        x = self.down2(f2)        # (B, 256, H/8, W/8)

        f3 = self.block3(x)       # (B, 384, H/8, W/8)
        x = self.down3(f3)        # (B, 384, H/16, W/16)

        f4 = self.block4(x)       # (B, 512, H/16, W/16)

        # 1x1 conv + adaptive pooling to match shape of f4 (B, 128, H/16, W/16)
        out1 = F.adaptive_avg_pool2d(self.proj1(f1), f4.shape[-2:])
        out2 = F.adaptive_avg_pool2d(self.proj2(f2), f4.shape[-2:])
        out3 = F.adaptive_avg_pool2d(self.proj3(f3), f4.shape[-2:])
        out4 = self.proj4(f4)  # already same resolution

        concat = torch.cat([out1, out2, out3, out4], dim=1)  # (B, 512, H/16, W/16)
        pooled = F.adaptive_avg_pool2d(concat, 1)           # (B, 512, 1, 1)
        flat = pooled.view(pooled.size(0), -1)              # (B, 512)

        return self.classifier(flat)

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
