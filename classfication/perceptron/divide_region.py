import perceptron as pp
import pandas as pd
import matplotlib as mat

from matplotlib.colors import ListedColormap
import numpy as np
import matplotlib.pyplot as plt

def plot_decision_regions(X, y, classifier, resolution=0.2):
    """
    二维数据集决策边界可视化
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
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    # y去重之后的种类
    listedColormap = ListedColormap(colors[0:len(np.unique(y))])

    # 花萼长度最小值-1，最大值+1
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    # 花瓣长度最小值-1，最大值+1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1

    # 将最大值，最小值向量生成二维数组xx1,xx2
    # np.arange(x1_min, x1_max, resolution)  最小值最大值中间，步长为resolution
    new_x1 = np.arange(x1_min, x1_max, resolution)
    new_x2 = np.arange(x2_min, x2_max, resolution)
    xx1, xx2 = np.meshgrid(new_x1, new_x2)

    # classifier.fit(X, y)
    # 预测值
    Z = classifier.predict([xx1, xx2])
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.4, camp=listedColormap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    for idx, c1 in enumerate(np.unique(y)):
        plt.scatter(x=X[y == c1, 0], y=X[y == c1, 1], alpha=0.8, c=listedColormap(idx), marker=markers[idx], label=c1)


df = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data", header=None)
# 输出最后20行的数据，并观察数据结构 萼片长度（sepal length），萼片宽度()，花瓣长度（petal length），花瓣宽度，种类
print(df.tail(n=20))

# 0到100行，第5列
y = df.iloc[0:100, 4].values
# 将target值转数字化 Iris-setosa为-1，否则值为1
y = np.where(y == "Iris-setosa", -1, 1)
# 取出0到100行，第1，第三列的值
x = df.iloc[0:100, [0, 2]].values
ppn = pp.Perceptron(eta=0.1, n_iter=10)
ppn.fit(x, y)
plot_decision_regions(x, y, classifier=ppn)
# 防止中文乱码
zhfont1 = mat.font_manager.FontProperties(fname='C:\Windows\Fonts\simsun.ttc')
plt.title("鸢尾花花瓣、花萼边界分割", fontproperties=zhfont1)
plt.xlabel("花瓣长度 [cm]", fontproperties=zhfont1)
plt.ylabel("花萼长度 [cm]", fontproperties=zhfont1)

plt.legend(loc="uper left")
plt.show()