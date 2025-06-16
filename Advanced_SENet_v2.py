print("[예]조의 번호를 입력하시오: 1")
team_idx = int(input(">> 조의 번호를 입력하시오: "))
print(f">> {team_idx}조 입니다.")
team_idx = 'team'+str(team_idx)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torchvision.datasets.utils import download_url, extract_archive
import random
import matplotlib.pyplot as plt
import numpy as np
import os
from torchinfo import summary

URL = "https://github.com/JanghunHyeon/AISW4202-Project/releases/download/v.1.1.0/project_dataset.zip"
ROOT = "./content/data"
ZIP_PATH = os.path.join(ROOT, "project_dataset.zip")
OUT_DIR  = os.path.join(ROOT, "project_dataset")

os.makedirs(ROOT, exist_ok=True)
download_url(URL, root=ROOT, filename="project_dataset.zip")
extract_archive(ZIP_PATH, OUT_DIR)

device = 'cuda' if torch.cuda.is_available() else 'cpu'


# Reproduce를 위한 Seed 고정
seed_id = 777
deterministic = True

random.seed(seed_id)
np.random.seed(seed_id)
torch.manual_seed(seed_id)
if device =='cuda':
    torch.cuda.manual_seed_all(seed_id)
if deterministic:
	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.benchmark = False

##################################--수정 가능한 부분 시작--#########################################

# SE Block 
class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(channels, channels // reduction)
        self.fc2 = nn.Linear(channels // reduction, channels)
        self.relu = nn.ReLU()
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

# SE-ResNet Block 
class SEResNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, reduction=16):
        super(SEResNetBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                              stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, 
                              stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.se = SEBlock(out_channels, reduction)
        
        self.skip_connection = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.skip_connection = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )
        
        self.relu = nn.ReLU()
        
    def forward(self, x):
        identity = self.skip_connection(x)
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.se(out)
        
        out += identity
        out = self.relu(out)
        return out

#1: ResNet + SENet
class ImprovedModel1(nn.Module):
    def __init__(self, num_classes=15, init_weights=True):
        super(ImprovedModel1, self).__init__()
        
        # 媛쒖꽑�� Stem block - 48x48 理쒖쟻��
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # SE-ResNet Blocks
        self.layer1 = nn.Sequential(
            SEResNetBlock(64, 64, stride=1),
            SEResNetBlock(64, 64, stride=1)
        )
        
        self.layer2 = nn.Sequential(
            SEResNetBlock(64, 128, stride=2),
            SEResNetBlock(128, 128, stride=1)
        )
        
        self.layer3 = nn.Sequential(
            SEResNetBlock(128, 256, stride=2),
            SEResNetBlock(256, 256, stride=1)
        )
        
        # Classification head with improved regularization
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.dropout1 = nn.Dropout(0.3)
        self.fc1 = nn.Linear(256, 128)
        self.dropout2 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, num_classes)
        
        if init_weights:
            self._initialize_weights()
    
    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout1(x)
        x = self.fc1(x)
        x = self.relu(x) if hasattr(self, 'relu') else nn.ReLU()(x)
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

# 媛쒖꽑�� 紐⑤뜽 2: Inception-ResNet �섏씠釉뚮━��
class ImprovedModel2(nn.Module):
    def __init__(self, num_classes=15, init_weights=True):
        super(ImprovedModel2, self).__init__()
        
        # Stem block
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # Simplified Inception blocks
        self.inception1 = self._make_inception_block(64, [16, 24, 32, 16])
        self.inception2 = self._make_inception_block(88, [32, 48, 64, 32])
        
        # Transition layers
        self.transition1 = nn.Sequential(
            nn.Conv2d(176, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        
        self.inception3 = self._make_inception_block(128, [48, 64, 96, 48])
        self.inception4 = self._make_inception_block(256, [64, 96, 128, 64])
        
        self.transition2 = nn.Sequential(
            nn.Conv2d(352, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )
        
        # Classification head
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.dropout1 = nn.Dropout(0.3)
        self.fc1 = nn.Linear(256, 128)
        self.dropout2 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, num_classes)
        
        if init_weights:
            self._initialize_weights()
    
    def _make_inception_block(self, in_channels, channels):
        branch1 = nn.Sequential(
            nn.Conv2d(in_channels, channels[0], kernel_size=1),
            nn.BatchNorm2d(channels[0]),
            nn.ReLU()
        )
        
        branch2 = nn.Sequential(
            nn.Conv2d(in_channels, channels[1], kernel_size=1),
            nn.BatchNorm2d(channels[1]),
            nn.ReLU(),
            nn.Conv2d(channels[1], channels[1], kernel_size=3, padding=1),
            nn.BatchNorm2d(channels[1]),
            nn.ReLU()
        )
        
        branch3 = nn.Sequential(
            nn.Conv2d(in_channels, channels[2], kernel_size=1),
            nn.BatchNorm2d(channels[2]),
            nn.ReLU(),
            nn.Conv2d(channels[2], channels[2], kernel_size=3, padding=1),
            nn.BatchNorm2d(channels[2]),
            nn.ReLU(),
            nn.Conv2d(channels[2], channels[2], kernel_size=3, padding=1),
            nn.BatchNorm2d(channels[2]),
            nn.ReLU()
        )
        
        branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels, channels[3], kernel_size=1),
            nn.BatchNorm2d(channels[3]),
            nn.ReLU()
        )
        
        return nn.ModuleList([branch1, branch2, branch3, branch4])
    
    def _forward_inception(self, x, inception_block):
        outputs = []
        for branch in inception_block:
            outputs.append(branch(x))
        return torch.cat(outputs, dim=1)
    
    def forward(self, x):
        x = self.stem(x)
        
        x = self._forward_inception(x, self.inception1)
        x = self._forward_inception(x, self.inception2)
        x = self.transition1(x)
        
        x = self._forward_inception(x, self.inception3)
        x = self._forward_inception(x, self.inception4)
        x = self.transition2(x)
        
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout1(x)
        x = self.fc1(x)
        x = nn.ReLU()(x)
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
##################################--수정 가능한 부분 끝--#########################################
model = ImprovedModel2(num_classes=15).to(device)

inputs = torch.Tensor(1,3,48,48).to(device)
out = model(inputs)
print(out)

transform_train = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=10),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std =[0.229, 0.224, 0.225]
    ),
])

transform_val = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std =[0.229, 0.224, 0.225]
    ),
])

data_dir = os.path.join(OUT_DIR, "data/ProjectDataset")
trainset = ImageFolder(os.path.join(data_dir, "train"), transform=transform_train)
valset = ImageFolder(os.path.join(data_dir, "val"), transform=transform_val)

trainloader = DataLoader(trainset, batch_size=256, shuffle=True, num_workers=6)
valloader = DataLoader(valset, batch_size=64, shuffle=False, num_workers=6)

criterion = nn.CrossEntropyLoss().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr = 0.005)

lr_sche = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.9)

print(len(trainloader))
epochs = 50

for epoch in range(epochs):  # loop over the dataset multiple times
    model.train()
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs
        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.to(device)

        # zero the parameter gradients
        optimizer.zero_grad()
        # forward + backward + optimize
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 50 == 49:    # print every 50 iterations
            print('[%d, %5d] loss: %.3f' %(epoch + 1, i + 1, running_loss / 50))
            running_loss = 0.0

    lr_sche.step()

    correct = 0
    total = 0
    model.eval()
    with torch.no_grad():
        for data in valloader:
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)

            _, predicted = torch.max(outputs.data, 1)

            total += labels.size(0)

            correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the 1,200 validation images: %.3f %%' % (100 * correct / total))


print('Finished Training')

model.eval()
correct = 0
total = 0
with torch.no_grad():
    for data in valloader:
        images, labels = data
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)

        _, predicted = torch.max(outputs.data, 1)

        total += labels.size(0)

        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the 1,200 validation images: %.3f %%' % (100 * correct / total))

fname = f"model_{team_idx}.pt"

model.eval()
example_input = torch.randn(1,3,48,48).to(device)
traced_script = torch.jit.trace(model, example_input)
traced_script.save(fname)