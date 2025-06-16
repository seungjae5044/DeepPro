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

class MyCustomModel(nn.Module):
    def __init__(self, num_classes=10, init_weights=True):
        super().__init__()

        # 48x48
        self.conv1 = nn.Sequential(nn.Conv2d(3, 64, kernel_size=7, stride=1, padding=3),
                                          nn.BatchNorm2d(64),
                                          nn.ReLU(),
                                          nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
                                          )

        # 24x24
        self.down_sample_layer1 = nn.Sequential(nn.Conv2d(64, 128, kernel_size=1, stride=2),
                                                nn.BatchNorm2d(128),
                                                nn.ReLU())

        self.conv2_1 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1),
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
        self.down_sample_layer2 = nn.Sequential(nn.Conv2d(128, 256, kernel_size=1, stride=2),
                                                nn.BatchNorm2d(256),
                                                nn.ReLU())

        self.conv3_1 = nn.Sequential(nn.Conv2d(128, 64, kernel_size=1),
                                     nn.BatchNorm2d(64),
                                     nn.ReLU(),

                                     nn.Conv2d(64, 64, kernel_size=3, padding=1, stride=2),
                                     nn.BatchNorm2d(64),
                                     nn.ReLU(),

                                     nn.Conv2d(64, 256, kernel_size=1),
                                     nn.BatchNorm2d(256),
                                     )
        # 6x6
        self.conv3_2 = nn.Sequential(nn.Conv2d(256, 128, kernel_size=1),
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
        out = out+identity4
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
##################################--수정 가능한 부분 끝--#########################################
model = MyCustomModel(num_classes=15).to(device)

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