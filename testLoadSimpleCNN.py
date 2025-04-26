from time import time, ctime, sleep
import os
import re
import random

import cv2
from PIL import Image
import torchvision.transforms as transforms
import torch.optim as optim

from simpleCNN import simpleCNN
from simpleCNN.simpleCNN import DigitCNN
from simpleCNN.utils import *

# 设置划分比例，例如 80% 划分到 trainFileGroup，20% 划分到 testFileGroup
trainRatio = 0.8
# trainTime = 100
trainTarget = 0.98

imgWidth = 28
imgHeight = 28


if __name__ == '__main__':
    print('start at:' + ctime())
    prepare(imgWidth, imgHeight)

    trainCase, testCase = getTestcase(processed_dir, 0.9)
    print(testCase)

    modelPath = os.path.join(modelDir, "simpleCNN.pth")

    # 检查是否有可用的 GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    cnn = DigitCNN()
    cnn.load_state_dict(torch.load(modelPath, map_location=device))

    simpleCNN.testModel(device, cnn, testCase)

    print("done")