import numpy as np
from sklearn import neighbors, datasets
from sklearn.model_selection import train_test_split



def load_data(file_name, test_s):
    """
    读取数据集，分割训练集和测试集
    :param file_name: 数据集文件名
    :param test_s: 测试集比例
    :return: 训练集和测试集
    """
    # 读取数据集，数据集为csv格式，第一列为目标值，其他列为特征值
    # usecols参数意思是选取文件的列，usecols=tuple(range(64))为前64列（0-63）
    X = np.loadtxt(file_name, usecols=tuple(range(64)))
    # usecols=(64,)意思是选取第64列，也就是目标值
    Y = np.loadtxt(file_name, usecols=(64,))
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=test_s)
    return x_train, x_test, y_train, y_test



def train_model(x_train, y_train):
    """
    训练模型, 使用KNN算法
    :param x_train: 训练集特征值
    :param y_train: 训练集目标值
    :return: 训练好的模型
    """
    # 调用KNN算法进行训练clf参数
    clf = neighbors.KNeighborsClassifier(n_neighbors=4, algorithm='auto', n_jobs=-1)
    # 用划分好的x_train, y_train进行训练
    clf.fit(x_train, y_train)
    return clf



def test_model(clf, x_test, y_test):
    # 用clf自带的函数进行测试参数训练结果
    result = clf.score(x_test, y_test)
    print("测试成功率为："+ str(result))