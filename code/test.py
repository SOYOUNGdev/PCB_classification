# 라이브러리 임포트
import os
import shutil
import random
import numpy as np
import pandas as pd
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import albumentations as A
import matplotlib.pyplot as plt

# # GPU 사용 설정
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# print(device)
# print(torch.cuda.get_device_name(0))

# # CUDA 사용 가능 여부 확인
# print(f"CUDA available: {torch.cuda.is_available()}")  # True여야 함

# # PyTorch에서 사용하는 CUDA 버전 확인
# print(f"CUDA version in PyTorch: {torch.version.cuda}")


# 하이퍼파라미터 설정
IMAGE_SIZE = 64
BATCH_SIZE = 64
N_EPOCHS = 20

# 이미지 전처리
transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize to [-1, 1]
])

# 데이터세트 디렉토리 설정
original_dir = './PCB_imgs/all'
original_dataset = ImageFolder(root=original_dir, transform=transform)

# 데이터세트 DataLoader로 변환
original_loader = DataLoader(original_dataset, batch_size=BATCH_SIZE, shuffle=True)

# 데이터 분할
file_paths = [img[0] for img in original_dataset.imgs]
targets = original_dataset.targets

train_images, test_images, train_targets, test_targets = train_test_split(
    file_paths, targets, stratify=targets, test_size=0.2, random_state=124
)

train_images, validation_images, train_targets, validation_targets = train_test_split(
    train_images, train_targets, stratify=train_targets, test_size=0.2, random_state=124
)

# DataFrame 생성
train_df = pd.DataFrame({'file_paths': train_images, 'targets': train_targets})
validation_df = pd.DataFrame({'file_paths': validation_images, 'targets': validation_targets})
test_df = pd.DataFrame({'file_paths': test_images, 'targets': test_targets})

# 각 클래스의 이미지 수 확인
print(f'train_ng: {sum(train_df['targets'] == 0)}개')  # 0이 ng 클래스인 경우
print(f'train_ok: {sum(train_df['targets'] == 1)}개')  # 1이 ok 클래스인 경우
print(f'test_ng: {sum(test_df['targets'] == 0)}개') 
print(f'test_ok: {sum(test_df['targets'] == 1)}개') 
print(f'val_ng: {sum(validation_df['targets'] == 0)}개') 
print(f'val_ok: {sum(validation_df['targets'] == 1)}개')  


# 커스텀 데이터세트 정의
class CustomDataset(Dataset):
    def __init__(self, file_paths, targets, aug=None, preprocess=None):
        self.file_paths = file_paths
        self.targets = targets
        self.aug = aug
        self.preprocess = preprocess

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        file_path = self.file_paths[index]
        target = self.targets[index]
        
        image = cv2.imread(file_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE))

        if self.aug is not None:
            image = self.aug(image=image)['image']

        if self.preprocess is not None:
            image = self.preprocess(image)

        image = np.transpose(image, (2, 0, 1))  # (H, W, C) -> (C, H, W)
        image = torch.tensor(image, dtype=torch.float32)
        
        return image, target

# # 데이터 증강 정의
# aug = A.Compose([
#     A.ShiftScaleRotate(p=0.5),
#     A.HorizontalFlip(p=0.5),
#     A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0, p=0.5)
# ])

# 데이터세트 인스턴스 생성
# train_dataset = CustomDataset(train_df['file_paths'].values, train_df['targets'].values, aug=aug)
train_dataset = CustomDataset(train_df['file_paths'].values, train_df['targets'].values)
validation_dataset = CustomDataset(validation_df['file_paths'].values, validation_df['targets'].values)
test_dataset = CustomDataset(test_df['file_paths'].values, test_df['targets'].values)

# DataLoader 설정
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
validation_loader = DataLoader(validation_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# 모델 정의
class CustomModel(nn.Module):
    def __init__(self, model_name='vgg16'):
        super(CustomModel, self).__init__()
        if model_name == 'vgg16':
            self.base_model = torchvision.models.vgg16(weights='IMAGENET1K_V1')
            self.base_model.classifier = nn.Identity()  # 마지막 분류기 제거
        elif model_name == 'resnet50':
            self.base_model = torchvision.models.resnet50(weights='IMAGENET1K_V1')
            self.base_model = nn.Sequential(*list(self.base_model.children())[:-1])  # 마지막 레이어 제거
        elif model_name == 'inception':
            self.base_model = torchvision.models.inception_v3(weights='IMAGENET1K_V1')
            self.base_model.classifier = nn.Identity()  # 마지막 분류기 제거
        elif model_name == 'mobilenet':
            self.base_model = torchvision.models.mobilenet_v2(weights='IMAGENET1K_V1')
            self.base_model.classifier = nn.Identity()  # 마지막 분류기 제거

        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(self._get_features_dim(model_name), 50)
        self.fc2 = nn.Linear(50, 1)  # Sigmoid 출력

    def _get_features_dim(self, model_name):
        if model_name in ['vgg16', 'inception']:
            return 25088  # VGG16의 출력 차원
        elif model_name == 'resnet50':
            return 2048  # ResNet50의 출력 차원
        elif model_name == 'mobilenet':
            return 1280  # MobileNetV2의 출력 차원

    def forward(self, x):
        x = self.base_model(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.dropout(x)
        x = nn.ReLU()(self.fc1(x))
        x = self.dropout(x)
        x = torch.sigmoid(self.fc2(x))
        return x

# 모델 초기화
model = CustomModel(model_name='resnet50').to(device)

# 손실 함수 및 최적화함수 정의
criterion = nn.BCELoss()  # 이진 교차 엔트로피 손실
optimizer = optim.Adam(model.parameters())  # Adam 옵티마이저

# 평가 함수 정의
def evaluate_model(model, data_loader, criterion):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, targets in data_loader:
            images, targets = images.to(device), targets.to(device).float()
            outputs = model(images)
            loss = criterion(outputs.view(-1), targets)
            total_loss += loss.item()
            
            predicted = (outputs.view(-1) > 0.5).float()
            correct += (predicted == targets).sum().item()
            total += targets.size(0)

    avg_loss = total_loss / len(data_loader)
    accuracy = correct / total
    return avg_loss, accuracy

# 모델 훈련
for epoch in range(N_EPOCHS):
    model.train()
    running_loss = 0.0
    
    for images, targets in train_loader:
        images, targets = images.to(device), targets.to(device).float()
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs.view(-1), targets)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()

    # 검증
    val_loss, val_accuracy = evaluate_model(model, validation_loader, criterion)

    print(f'Epoch [{epoch+1}/{N_EPOCHS}], Train Loss: {running_loss / len(train_loader):.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}')

# 테스트 데이터세트로 평가
test_loss, test_accuracy = evaluate_model(model, test_loader, criterion)
print(f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}')

