import numpy as np

class Perceptron(object):
    """
    Perceptron:感知器
    """
    def __init__(self, eta=0.01, n_iter=10):
        """
        :param eta: 学习速率
        :param n_iter: 在训练集进行迭代的次数
        """
        self.eta = eta
        self.n_iter = n_iter

    def net_input(self, x):
        """
        :param x: 数据集
        :return: 计算向量的点积

        description:
            sum(i*j for i, j in zip(x, self.w_[1:])) python计算点积
        """
        return np.dot(x, self.w_[1:]) + self.w_[0]
    """ 计算类标 """
    def predict(self, x):
        """
        :param x: 数据集
        :return:
        """
        return np.where(self.net_input(x))

    def fit(self, x, y):
        """
        :param x:
        :param y:
        :return:

        """
        # 权值，初始化为一个零向量R的(m+1)次方，m代表数据集中纬度（特征）的数量
        # x.shape[1]表示数据集中的第一行的列数
        self.w_ = np.zeros(1 + x.shape[1])
        # 收集每轮迭代过程中错误分类样本的数量，以便后续对感知器在训练中表现的好坏做出判定
        self.errors_ = []

        for _ in range(self.n_iter):
            errors = 0
            for x_element, target in zip(x, y):
                update = self.eta * (target - self.predict(x_element))
                self.w_[1:] += update * x_element
                self.w_[0] += update
                errors += int(update != 0.0)
            self.errors_.append(errors)
        return self