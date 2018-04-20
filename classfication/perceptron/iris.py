import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import pandas as pd
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
        print(x, end=" ")
        print(self.w_[:], end=" ")
        x_dot = np.dot(x, self.w_[1:]) + self.w_[0]
        print("的点积是：%d" % x_dot, end="  ")
        return x_dot

    """ 计算类标 """
    def predict(self, x):
        """
        :param x: 数据集
        :return:
        """
        target_pred = np.where(self.net_input(x) >= 0.0, 1, -1)
        print("预测值：%d" % target_pred, end="; ")
        return target_pred

    def fit(self, x, y):
        """
        :param x: 被训练的数据集
        :param y: 被训练的数据集的实际结果
        :return:

        """
        """
          权值，初始化为一个零向量R的(m+1)次方，m代表数据集中纬度（特征）的数量
          x.shape[1] = (100,2) 一百行2列：表示数据集中的列数即特征数
          
          np.zeros(count) 将指定数量count初始化成元素均为0的数组 self.w_ = [ 0.  0.  0.]
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


def plot_decision_regions(X, y, classifier, resolution=0.2):
    """
    :parameter
    -----------------------------
    :param self:
    :param X:
    :param y:
    :param classifier:
    :param resolution:
    :return:
    -----------------------------
    """
    markers = ('s', 'x', 'o', '~', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')

    listedColormap = ListedColormap(colors[:len(np.unique(y))])

    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1

    # 将最大值，最小值向量生成二维数组xx1,xx2
    # np.arange(x1_min, x1_max, resolution)  最小值最大值中间，步长为resolution
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution), np.arange(x2_min, x2_max, resolution))

    z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    z = z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, z, alpha=0.4, camp=listedColormap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    for idx, c1 in enumerate(np.unique(y)):
        plt.scatter(x=X[y == c1, 0], y=X[y == c1, 1], alpha=0.8, c=listedColormap(idx), marker=markers[idx], label=c1)

df = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data", header=None)
# 输出最后5行的数据
# print(df.tail())

y = df.iloc[0:100, 4].values
y = np.where(y == "Iris-setosa", -1, 1)
x = df.iloc[0:100, [0, 2]].values

""" 鸢尾花散点图 """
# plt.scatter(x[:50, 0], x[:50, 1], color="red", marker="o", label="setosa")
# plt.scatter(x[50:100, 0], x[50:100, 1], color="blue", marker="x", label="versicolor")
# plt.xlabel("petal length")
# plt.ylabel("sepal length")
# plt.legend(loc="upper left")
# plt.show()

""" 误差数折线图 """
ppn = Perceptron(eta=0.1, n_iter=10)
ppn.fit(x, y)
# plt.plot(range(1, len(ppn.errors_) + 1), ppn.errors_, marker="o")
# plt.xlabel("Epochs")
# plt.ylabel("Number of misclassification")
# plt.show()

plot_decision_regions(x, y, classifier=ppn)
plt.xlabel("sepal length [cm]")
plt.ylabel("petal length [cm]")
plt.legend(loc="uper left")
plt.show()



