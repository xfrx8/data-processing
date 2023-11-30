import torch
import torch.nn as nn
import torch.nn.functional as F
class MyFCNet(nn.Module):

    def __init__(self):
        super(MyFCNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=4, out_channels=100, kernel_size=(6, 1))
        self.maxpool1 = nn.MaxPool2d(kernel_size=(2, 1))
        self.conv2 = nn.Conv2d(100, 200, kernel_size=(5, 1))
        self.maxpool2 = nn.MaxPool2d(kernel_size=(2, 1))
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(800, 128)
        self.dropout1 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(128, 64)
        self.dropout2 = nn.Dropout(0.3)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 1)
    
    def forward(self, x, targets=None, epoch=None):
        x = torch.relu(self.conv1(x))
        x = self.maxpool1(x)
        x = torch.relu(self.conv2(x))
        x = self.maxpool2(x)
        x = self.flatten(x)
        x = torch.relu(self.fc1(x))
        x = self.dropout1(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout2(x)
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)
      
        return x

def fcnet1(**kwargs):
    return MyFCNet(**kwargs)