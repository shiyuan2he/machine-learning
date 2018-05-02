from sklearn.datasets import make_moons, make_circles, make_classification
# 获取数据
from naive_bayes import NaiveBayes
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

# 获取数据
X, y = make_circles(noise=0.2, factor=0.5, random_state=1)
# X, y = make_moons(noise=0.2, factor=0.5, random_state=1)
# X, y = make_classification(noise=0.2, factor=0.5, random_state=1)

method = NaiveBayes()
method.fit(X, y)

# 调整图片风格
mpl.style.use('fivethirtyeight')
# 定义xy网格，用于绘制等值线图
x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                     np.arange(y_min, y_max, 0.1))
# 预测可能性
Z = method.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
Z = Z.reshape(xx.shape)

plt.contourf(xx, yy, Z, alpha=.8)
# 绘制散点图
plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k')
# 画出一下坐标的点
plt.scatter(0.5, 0.5)
plt.title("GaussianNaiveBayes")
plt.axis("equal")
plt.show()

# 对(0.5, 0.5)处的值进行预测
new_X = np.array([[0.5, 0.5]])
print(method.predict(new_X))