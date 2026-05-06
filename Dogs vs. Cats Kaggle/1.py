import os
import zipfile
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from PIL import Image
from sklearn.model_selection import train_test_split

# 检查 GPU 是否可用
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 解压训练集
with zipfile.ZipFile('/kaggle/input/competitions/dogs-vs-cats/train.zip', 'r') as zip_ref:
    zip_ref.extractall('/kaggle/working/')

# 查看一下解压后的文件数量
train_dir = '/kaggle/working/train'
filenames = os.listdir(train_dir)
print(f"Total images: {len(filenames)}") # 应该是 25000

data_list = []
for fname in filenames:
    if fname.split('.')[0] == 'dog' :
        label = 1 # dog 为 1
    else :
        label = 0 # cat 为 0
    data_list.append((os.path.join(train_dir, fname), label)) # 作用是？

# 拆分训练集和验证集 (80% 训练, 20% 验证)
train_data, val_data = train_test_split(data_list, test_size=0.2, random_state=42)

print(f"Train size: {len(train_data)}, Val size: {len(val_data)}")

class CatDogDataset(Dataset):
    def __init__(self, data_list, transform=None):
        self.data_list = data_list
        self.transform = transform

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        img_path, label = self.data_list[idx]
        image = Image.open(img_path) # 读取图片
        
        if self.transform:
            image = self.transform(image)
            
        return image, label 

# biuld a DNN and make it useful?






