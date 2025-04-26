from time import time, ctime, sleep
import os
import cv2
from PIL import Image
import torchvision.transforms as transforms
import torch.optim as optim
import torch
import re
import random

abs_work_space = os.getcwd()
origin_dir = os.path.join(abs_work_space, 'rsc/origin')
processed_dir = os.path.join(abs_work_space, 'rsc/processed')
modelDir = os.path.join(abs_work_space, "rsc/model")

def transBmpToTensor(path):
    image = Image.open(path)
    trans = transforms.Compose([
        transforms.ToTensor()  # 将PIL图像转换为Tensor
    ])
    return trans(image)


def createDir(output_dir: str) -> None:
    if not os.path.exists(output_dir):
        try:
            os.makedirs(output_dir)
            print("创建目录: %s", output_dir)
        except OSError as e:
            print("创建目录 %s 失败: %s", output_dir, str(e))
            raise



def clear_old_data():
    """
    清除旧数据, 删除rsc目录下的processed和model文件夹中的内容, 但保留文件夹
    :return: 无异常则返回True
    """
    for directory in [processed_dir]:
        if os.path.exists(directory):
            # 遍历文件夹中的所有内容
            for item in os.listdir(directory):
                item_path = os.path.join(directory, item)
                # 如果是文件，删除文件
                if os.path.isfile(item_path):
                    os.remove(item_path)
                # 如果是文件夹，递归删除文件夹内容后删除文件夹
                elif os.path.isdir(item_path):
                    os.rmdir(item_path)
    return True


def preprocess(dir_name , pre_dir, width, height):
    """
    根据文件夹名, 循环遍历所有图像, 对每一张图像进行预处理
    :param dir_name: 原始文件夹
    :param pre_dir: 预处理后的图像保存路径
    :return: 预处理过程无异常情况则返回True,否则False
    """
    # 1.获得指定文件夹下所有的文件名
    file_name_list = os.listdir(dir_name)
    # 2.针对每一个图像进行预处理操作, 循环遍历文件名列表
    for file_name in file_name_list:
        # 2.1 根据文件名, 读取图像（灰度图）
        img_path = dir_name + "/" + file_name
        # print(file_name)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)    # 灰度图
        # 2.2 对图像进行预处理操作
        dst_img = img_preprocess(img, width, height)
        # 2.3 把预处理后的图像存储
        preimg_path = pre_dir + "/" + file_name
        cv2.imwrite(preimg_path, dst_img)
    return True



def img_preprocess(src_img, width, height):
    """
    针对一张灰度图, 进行预处理,对图像进行简单的剪裁
    :param src_img: 原始图片
    :return: 预处理之后的图片
    """
    image = src_img
    # 获取长宽
    row, col = image.shape
    # 交叉遍历, 顶层从底下往上找0, 右边从左边往右找0
    top = row
    bottom = 0
    left = col
    right = 0
    # i为行, j为列
    for i in range(row):
        for j in range(col):
            # 找0, 也就是找黑色的有字部分
            if image[i, j] == 0:
                # 找到最小的有字部分的行, 也就是顶层
                if i < top:
                    top = i
                # 找到最小的有字部分的列, 也就是最左边
                if j < left:
                    left = j
                # 找到最大的有字部分的行, 也就是底层
                if i > bottom:
                    bottom = i
                # 找到最大的有字部分的列, 也就是最右边
                if j > right:
                    right = j
    # 剪裁图像
    dst_img = image[int(top):int(bottom), int(left):int(right)]
    # 统一预处理后的图像大小, 8 * 8像素
    dst_img = cv2.resize(dst_img, (width, height))
    return dst_img

def prepare(width, height):
    """
    预处理图像, 提取特征值, 并存储到文件中
    """
    createDir(processed_dir)
    clear_old_data()
    preprocess(origin_dir, processed_dir, width, height)
    print(f"origin {origin_dir} processed {processed_dir}")

def getTestcase(dir, trainRatio):
    trainFileGroup = {}
    testFileGroup = {}
    
    totalCount = 0

    fileList = os.listdir(dir)
    for file in fileList:
        match = re.match(r'(\d+)_', file)
        if match:
            number = int(match.group(1))

            if random.random() < trainRatio:
                if number not in trainFileGroup:
                    trainFileGroup[number] = []

                trainFileGroup[number].append(os.path.join(dir, file))
            else:
                if number not in testFileGroup:
                    testFileGroup[number] = []

                testFileGroup[number].append(os.path.join(dir, file))              

            totalCount += 1  
        else:
            pass

    print(f"load testcase count {totalCount}")
    return (trainFileGroup, testFileGroup)

def getTrueLabel(length: int, value: int) -> torch.Tensor:
    assert value < length, "value must be smaller that length"
    trueLabel = torch.zeros(1, length)
    trueLabel[0][value] = 1.0
    return trueLabel
