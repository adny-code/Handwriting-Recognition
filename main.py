import os
import multiprocessing
from xtquant import xtdata
from time import time, ctime, sleep
from atexit import register
import sys
import time
import cv2
import numpy as np
from hwrImage import image_handle
from hwrModel import knn_model


# 设置编码格式为utf-8
sys.stdout.reconfigure(encoding='utf-8')

# 设置最大线程数为逻辑核心数
logical_cores = multiprocessing.cpu_count()

# TODO: 这里设置的线程数是逻辑核心数-1, 设置所有核心, joblib库有个warning
os.environ["LOKY_MAX_CPU_COUNT"] = str(logical_cores - 1)
print(f"Logical cores: {logical_cores}")


@register 
def atexit():
     print('all done at:' + ctime())


def recognition(img_path, clf):
    # 读取单张图片进行识别
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # 灰度图
    # 对单张图片进行预处理
    dst_img = image_handle.img_preprocess(img)
    # 对单张图片提取特征
    img_feat = image_handle.get_feature(dst_img)
    # 对单张图片进行预测
    result = clf.predict(img_feat)
    return result


def get_feature():
    """
    预处理图像, 提取特征值, 并存储到文件中
    """
    image_handle.clear_old_data()
    image_handle.preprocess(image_handle.origin_dir, image_handle.processed_dir)
    image_handle.create_feature_file(image_handle.processed_dir, image_handle.feature_file)


def get_clf(split: float = 0.3):
    """
    训练模型, 返回训练好的模型和测试集
    :param split: 测试集比例
    :return: 训练好的模型和测试集
    """
    x_train, x_test, y_train, y_test = knn_model.load_data(image_handle.feature_file, split)
    clf = knn_model.train_model(x_train, y_train)
    return clf, x_test, y_test




if __name__ == '__main__':
    print('start at:' + ctime())
    
    # 预处理图像, 提取特征值, 并存储到文件中
    get_feature()

    # 训练模型
    clf, x_test, y_test = get_clf(0.2)

    # 测试模型准确率
    knn_model.test_model(clf, x_test, y_test)

    # 识别单张图片
    test_pattern_dir = image_handle.test_pattern_dir
    first: int = recognition(os.path.join(test_pattern_dir, '0_49.bmp'), clf)
    second: int = recognition(os.path.join(test_pattern_dir, '3_11.bmp'), clf)
    print(f"第一个图片识别为: {first}")
    print(f"第二个图片识别为: {second}")
    