import numpy as np

class Perceptron(object):
    """
    Perceptron:感知器
        感知器收敛的前提是：两个类别必须是线性可分的，且学习速率必须足够小，否则感知器算法会永远不停的更新权值
    """

    def __init__(self, eta=0.01, n_iter=10):
        """
        初始化感知器对象
        :param eta: float 学习速率
        :param n_iter: int 在训练集进行迭代的次数
        """
        self.eta = eta
        self.n_iter = n_iter

    def net_input(self, x):
        """
        计算净输入
        :param x: list[np.array] 一维数组数据集
        :return: 计算向量的点积
            向量点积的概念：
                {1，2，3} * {4，5，6} = 1*4+2*5+3*6 = 32

        description:
            sum(i*j for i, j in zip(x, self.w_[1:])) python计算点积
        """
        print(x, end=" ")
        print(self.w_[:], end=" ")
        x_dot = np.dot(x, self.w_[1:]) + self.w_[0]
        print("的点积是：%d" % x_dot, end="  ")
        return x_dot

    """ 计算类标 """
    def predict(self, x):
        """
        预测方法
        :param x: list[np.array] 一维数组数据集
        :return:
        """
        target_pred = np.where(self.net_input(x) >= 0.0, 1, -1)
        print("预测值：%d" % target_pred, end="; ")
        return target_pred

    def fit(self, x, y):
        """
        学习、训练方法
        :param x: list[np.array] 一维数组数据集
        :param y: 被训练的数据集的实际结果
        :return:
          权值，初始化为一个零向量R的(m+1)次方，m代表数据集中纬度（特征）的数量
          x.shape[1] = (100,2) 一百行2列：表示数据集中的列数即特征数

          np.zeros(count) 将指定数量count初始化成元素均为0的数组 self.w_ = [ 0.  0.  0.]
        """

        """
        按照python开发惯例，对于那些并非在初始化对象时创建但是又被对象中其他方法调用的属性，可以在后面添加一个下划线
        """
        self.w_ = np.zeros(1 + x.shape[1])
        # 收集每轮迭代过程中错误分类样本的数量，以便后续对感知器在训练中表现的好坏做出判定
        self.errors_ = []

        for _ in range(self.n_iter):
            errors = 0
            for x_element, target in zip(x, y):
                # 如果预测值（self.predict(x_element)）和实际值(target)一致，则update为0
                update = self.eta * (target - self.predict(x_element))
                print("真实值：%d" % target)
                self.w_[1:] += update * x_element
                self.w_[0] += update
                errors += int(update != 0.0)
            self.errors_.append(errors)
        return self
