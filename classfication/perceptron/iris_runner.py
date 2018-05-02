import perceptron as pp
import pandas as pd
import matplotlib as mat

from matplotlib.colors import ListedColormap
import numpy as np
import matplotlib.pyplot as plt

def plot_decision_regions(x, y, classifier, resolution=0.2):
    """
    二维数据集决策边界可视化
    :parameter
    -----------------------------
    :param self: 将鸢尾花花萼长度、花瓣长度进行可视化及分类
    :param x: list 被分类的样本
    :param y: list 样本对应的真实分类
    :param classifier: method  分类器：感知器
    :param resolution:
    :return:
    -----------------------------
    """
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    # y去重之后的种类
    listedColormap = ListedColormap(colors[:len(np.unique(y))])

    # 花萼长度最小值-1，最大值+1
    x1_min, x1_max = x[:, 0].min() - 1, x[:, 0].max() + 1
    # 花瓣长度最小值-1，最大值+1
    x2_min, x2_max = x[:, 1].min() - 1, x[:, 1].max() + 1

    # 将最大值，最小值向量生成二维数组xx1,xx2
    # np.arange(x1_min, x1_max, resolution)  最小值最大值中间，步长为resolution
    new_x1 = np.arange(x1_min, x1_max, resolution)
    new_x2 = np.arange(x2_min, x2_max, resolution)
    xx1, xx2 = np.meshgrid(new_x1, new_x2)

    # 预测值
    # z = classifier.predict([xx1, xx2])
    z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    z = z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, z, alpha=0.4, camp=listedColormap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    for idx, c1 in enumerate(np.unique(y)):
        plt.scatter(x=x[y == c1, 0], y=x[y == c1, 1], alpha=0.8, c=listedColormap(idx), marker=markers[idx], label=c1)


df = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data", header=None)

# 0到100行，第5列
y = df.iloc[0:100, 4].values
# 将target值转数字化 Iris-setosa为-1，否则值为1,相当于激活函数-在此处表现为分段函数
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