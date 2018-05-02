import numpy as np

class AdalineSGD(object):
    """
    AdalineSGD: 自适应线性神经网络-单层神经网络
    """

    def __init__(self, eta=0.01, n_iter=50, shuffle=True, random_state=None):
        """
        初始化感知器对象
        :param eta: float 学习速率
        :param n_iter: int 在训练集进行迭代的次数
        :param shuffle:
        :param random_state:
        """
        self.eta = eta
        self.n_iter = n_iter
        self.shuffle = shuffle
        self.w_initialized = False
        if random_state:
            None

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
        # 按照python开发惯例，对于那些并非在初始化对象时创建但是又被对象中其他方法调用的属性，可以在后面添加一个下划线
        self.w_ = np.zeros(1 + x.shape[1])
        # 存储代价函数的输出值以检查本轮训练后算法是否收敛
        self.cost_ = []

        for i in range(self.n_iter):
            output = self.net_input(x)
            errors = (y - output)
            # 计算第0个位置的权重
            # 计算1到m位置的权重 X.T.dot(errors) 计算特征矩阵与误差向量之间的乘积
            self.w_[1:] += self.eta * x.T.dot(errors)
            self.w_[0] += self.eta * errors.sum()
            cost = (errors**2).sum() / 2.0
            self.cost_.append(cost)
        return self

    def net_input(self, x):
        """
        计算净输入
        :parameter
        ----------
        :param x: list[np.array] 一维数组数据集
        :return: 计算向量的点积
            向量点积的概念：
                {1，2，3} * {4，5，6} = 1*4+2*5+3*6 = 32

        description:
            sum(i*j for i, j in zip(x, self.w_[1:])) python计算点积
        """
        return np.dot(x, self.w_[1:]) + self.w_[0]

    def activation(self, x):
        return self.net_input(x)

    """ 计算类标 """
    def predict(self, x):
        """
        预测方法
        :param x: list[np.array] 一维数组数据集
        :return:
        """
        target_pred = np.where(self.net_input(x) >= 0.0, 1, -1)
        # print("预测值：%d" % target_pred, end="; ")
        return target_pred
