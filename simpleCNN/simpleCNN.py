import torch
import torch.nn as nn
import torch.nn.functional as F

class DigitCNN(nn.Module):
    def __init__(self):
        super(DigitCNN, self).__init__()
        self.channelCount = 8
        # 卷积层
        self.conv1 = nn.Conv2d(1, self.channelCount, kernel_size=3, stride=1, padding=1)
        
        # 全连接层
        self.fc1 = nn.Linear(self.channelCount * 14 * 14, 128)  # 假设经过池化后尺寸为14x14
        self.fc2 = nn.Linear(128, 10)  # 输出10个类别（0-9）
        
        # Dropout层防止过拟合
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x):
        # 卷积 + ReLU + 池化
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)  # 28x28 -> 14x14
        
        # 展平
        x = x.view(-1, self.channelCount * 14 * 14)
        # print(f"x: flatten shape {x.shape} data {x}")
        
        # 全连接层
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        # print(f"x: output {x.shape} data {x}")
        return x