# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import matplotlib as mat
import pandas as pd
import numpy as np

df = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data", header=None)
# 输出最后20行的数据，并观察数据结构 萼片长度（sepal length），萼片宽度()，花瓣长度（petal length），花瓣宽度，种类
print(df.tail(n=20))

# 0到100行，第5列
y = df.iloc[0:100, 4].values
# 将target值转数字化 Iris-setosa为-1，否则值为1
y = np.where(y == "Iris-setosa", -1, 1)
# 取出0到100行，第1，第三列的值
x = df.iloc[0:100, [0, 2]].values

""" 鸢尾花散点图 """

# scatter绘制点图
plt.scatter(x[0:50, 0], x[0:50, 1], color="red", marker="o", label="setosa")
plt.scatter(x[50:100, 0], x[50:100, 1], color="blue", marker="x", label="versicolor")
# 防止中文乱码
zhfont1 = mat.font_manager.FontProperties(fname='C:\Windows\Fonts\simsun.ttc')
plt.title("鸢尾花散点图", fontproperties=zhfont1)
plt.xlabel(u"花瓣长度", fontproperties=zhfont1)
plt.ylabel(u"萼片长度", fontproperties=zhfont1)
plt.legend(loc="upper left")
plt.show()



