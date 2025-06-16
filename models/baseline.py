import torch
import torch.nn as nn

class BaselineModel(nn.Module):
    def __init__(self, num_classes=15, init_weights=True):
        super(BaselineModel, self).__init__()

        # 48x48
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=1, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        # 24x24
        self.down_sample_layer1 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=1, stride=2),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )

        self.conv2_1 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=1),
            nn.BatchNorm2d(128),
        )
        self.relu = nn.ReLU()

        # 12x12
        self.down_sample_layer2 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=1, stride=2),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )

        self.conv3_1 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 256, kernel_size=1),
            nn.BatchNorm2d(256),
        )
        
        # 6x6
        self.conv3_2 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=1),
            nn.BatchNorm2d(256),
        )

        # 6x6
        self.global_pool = nn.AvgPool2d(kernel_size=6)
        self.fc = nn.Linear(256, num_classes)

        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.conv1(x)

        identity1 = self.down_sample_layer1(x)
        out = self.conv2_1(x)
        out = identity1 + out
        out = self.relu(out)

        identity3 = self.down_sample_layer2(out)
        out = self.conv3_1(out)
        out = identity3 + out
        out = self.relu(out)

        identity4 = out
        out = self.conv3_2(out)
        out = out + identity4
        out = self.relu(out)

        pool = self.global_pool(out)
        fc = pool.view(pool.size(0), -1)
        fc = self.fc(fc)

        return fc

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)