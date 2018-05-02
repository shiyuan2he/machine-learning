import numpy as np


class NativeBayes:
    def N(self, x, mu, sigma):
        p = 1 / sigma / np.sqrt(2 * np.pi) * np.exp(-(x - mu) ** 2 / 2 / sigma ** 2)
        return p

    def fit(self, X, y):
        X1 = X[y == 1]  # 找到所有正样本
        X2 = X[y == 0]  # 找到所有负样本

        # 分别计算负样本的均值和方差:
        self.mu_positive = np.mean(X1, axis=0)
        self.std_positive = np.std(X1, axis=0)
        # 分别计算正样本的均值和方差:
        self.mu_negative = np.mean(X2, axis=0)
        self.std_negative = np.std(X2, axis=0)
        # 计算p(y==1)
        self.py_positive = len(y[y == 1]) / len(y)
        self.py_negative = len(y[y == 0]) / len(y)

    def predict_prob(self, X):
        prb = []

        for a in X:
            # 假设有k个特征:
            p_xk_on_positive = []
            p_xk_on_negative = []
            p_xk = []
            for k, x in enumerate(a):
                # 样本a的第k个特征,值为x
                """
                P(x1, x2) = P(x1)*P(x2) 
                P(x1) = P(x1|y=0)P(y=0) + P(x1|y=1)P(y=1) 全概率公式
                P(x2) = P(x2|y=0)P(y=0) + P(x2|y=1)P(y=1) 全概率公式       
                """
                p_xk_on_positive.append(self.N(x, self.mu_positive[k], self.std_positive[k]))
                p_xk_on_negative.append(self.N(x, self.mu_negative[k], self.std_negative[k]))

                """
                P(y|x1,x2) = P(x1|y)*P(x2|y)*P(y) / P(x1,x2)
                """
                p_xk.append(p_xk_on_negative[k] * self.py_negative + p_xk_on_positive[k] * self.py_positive)
            p_positive_on_x = np.multiply.reduce(p_xk_on_positive) * self.py_positive / np.multiply.reduce(p_xk)
            p_negative_on_x = np.multiply.reduce(p_xk_on_negative) * self.py_negative / np.multiply.reduce(p_xk)
            prb.append([p_negative_on_x, p_positive_on_x])
        return np.array(prb)

    def predict(self, X):
        prob = self.predict_prob(X)
        return np.argmax(prob, axis=1)

# 计算预测精度:
def accurency(y_pred, y_real):
    return 1 - np.sum(abs(y_pred - y_real))/len(y_pred)

from sklearn.datasets import make_moons, make_circles, make_classification
import matplotlib.pyplot as plt

# 获取数据
X, y = make_circles(noise=0.2, factor=0.5, random_state=1)
#X, y = make_moons(noise=0.2, random_state=1)
#X, y = make_classification()
# print(X[:5])
# print(y[:5])

# 模型训练及预测:
clf = NativeBayes()
clf.fit(X, y)
y_pred = clf.predict(X)
print('模型精度为:%0.4f' % accurency(y_pred, y))

new_X = np.array([[0.5, 0.5]])
print(clf.predict(new_X))

import matplotlib as mpl
# 调整图片风格
mpl.style.use('fivethirtyeight')
# 定义xy网格，用于绘制等值线图
x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                     np.arange(y_min, y_max, 0.1))

# 预测可能性
Z = clf.predict_prob(np.c_[xx.ravel(), yy.ravel()])[:, 1]
Z = Z.reshape(xx.shape)
plt.contourf(xx, yy, Z, alpha=.8)
# plt.show()
# 绘制散点图
plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k')
plt.title("GaussianNaiveBayes")
plt.axis("equal")
plt.show()