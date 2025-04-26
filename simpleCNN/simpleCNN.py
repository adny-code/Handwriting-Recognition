from time import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from simpleCNN.utils import *

class DigitCNN(nn.Module):
    def __init__(self):
        super(DigitCNN, self).__init__()
        self.channelCount = 32
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


def trainModel(device, model, trainCase, trainTarget):
    startTime = time()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 将模型移动到 GPU
    model.to(device)
    criterion.to(device)

    trainCount = 0
    trueCount = 0
    score = 0
    t = 0
    while score < trainTarget:
        for num in trainCase:
            cases = trainCase[num]
            trueLabel = getTrueLabel(10, num)
            for case in cases:
                data = transBmpToTensor(case).to(device)

                output = model.forward(data)
                index = torch.argmax(output)
                # print(f"type {type(output)} output {output} index {index}")

                if index == num:
                    trueCount += 1
                
                optimizer.zero_grad()
                loss = criterion(output, trueLabel.to(device))
                loss.backward()
                optimizer.step()

                trainCount += 1

            score = trueCount / trainCount
            print(f"t {t} score {score} trueCount {trueCount} trainCount {trainCount}")

        t += 1
    
    endTime = time()
    elapsedTime = endTime - startTime
    print(f"Elapsed time {elapsedTime} seconds")

def testModel(device, model, testCase):
    testCount = 0
    trueCount = 0

    # 将模型移动到 GPU
    model.to(device)

    for num in testCase:
        cases = testCase[num]
        for case in cases:
            data = transBmpToTensor(case).to(device)
            output = model.forward(data)
            index = torch.argmax(output)

            if num == index:
                trueCount += 1
            else:
                print(f"mistaken num {num} res {index} file {case}")
            testCount += 1

    print(f"trueCount {trueCount} testCount {testCount}")