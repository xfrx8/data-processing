import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter, defaultdict
import plotly.express as px
import torch
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
import random
from model import MyFCNet
x_train=pd.read_excel('data/promoter_w_20799.xlsx',header=None).to_numpy().reshape(-1)
y_train=pd.read_excel('data/fluorescence_w_20799.xlsx',header=None).to_numpy().reshape(-1)

def seq2onehot(seq):     #convert the cRBS sequences to one-hot encoding
    module = np.array([[[1,0,0,0]],
                       [[0,1,0,0]],
                       [[0,0,1,0]],
                       [[0,0,0,1]]])
    i = 0
    cRBS_onehot = []
    for i in seq:
        tmp = []
        for item in i:
            if item == 't' or item == 'T':
                tmp.append(module[0])
            elif item == 'c' or item == 'C':
                tmp.append(module[1])
            elif item == 'g' or item == 'G':
                tmp.append(module[2])
            elif item == 'a' or item == 'A':
                tmp.append(module[3])
            else:
                tmp.append([[0,0,0,0]])
        cRBS_onehot.append(tmp)
    cRBS_onehot=np.array(cRBS_onehot).astype('float32')
    return cRBS_onehot
x_train = seq2onehot(x_train)
class MyDataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return torch.tensor(np.transpose(self.x[idx], (2, 0, 1))), torch.tensor(self.y[idx])

train_dataset = MyDataset(x_train, y_train)

def split_dataset(dataset, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, batch_size=32):
    # 计算数据集的大小
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    # 计算划分的索引
    train_split = int(np.floor(train_ratio * dataset_size))
    val_split = int(np.floor(val_ratio * dataset_size)) + train_split
    test_split = int(np.floor(test_ratio * dataset_size)) + val_split
    # 随机打乱数据集
    random.shuffle(indices)
    # 划分数据集
    train_indices = indices[:train_split]
    val_indices = indices[train_split:val_split]
    test_indices = indices[val_split:test_split]
    # 创建数据加载器
    train_sampler = SubsetRandomSampler(train_indices)
    val_sampler = SubsetRandomSampler(val_indices)
    test_sampler = SubsetRandomSampler(test_indices)
    train_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler)
    val_loader = DataLoader(dataset, batch_size=batch_size, sampler=val_sampler)
    test_loader = DataLoader(dataset, batch_size=batch_size, sampler=test_sampler)
    return train_loader, val_loader, test_loader

train_loader, val_loader, test_loader = split_dataset(train_dataset, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, batch_size=32)

model = MyFCNet()

# 定义损失函数
criterion = torch.nn.MSELoss()
# 定义优化器
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
# 定义学习率衰减策略
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
# 定义训练参数
epochs = 100
# 训练模型
for epoch in range(epochs):
    model.train()
    for i, (x, y) in enumerate(train_loader):
        # 清零梯度
        optimizer.zero_grad()
        # 前向传播
        y_pred = model(x)
        # 计算损失
        loss = criterion(y_pred, y)
        # 反向传播
        loss.backward()
        # 更新参数
        optimizer.step()
    # 更新学习率
    scheduler.step()
    # 计算训练集的损失
    train_loss = 0.0
    for i, (x, y) in enumerate(train_loader):
        y_pred = model(x)
        loss = criterion(y_pred, y)
        train_loss += loss.item()
    train_loss /= len(train_loader)
    # 计算验证集的损失
    val_loss = 0.0
    for i, (x, y) in enumerate(val_loader):
        y_pred = model(x)
        loss = criterion(y_pred, y)
        val_loss += loss.item()
    val_loss /= len(val_loader)
    # 打印损失
    print('Epoch: {}, Train Loss: {:.4f}, Val Loss: {:.4f}'.format(epoch, train_loss, val_loss))

