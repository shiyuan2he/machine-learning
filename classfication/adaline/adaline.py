import matplotlib.pyplot as plt
from adaline_gd import AdalineGD
import pandas as pd
import numpy as np

class Adaline:

    def init(self):
        df = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data", header=None)
        # 输出最后20行的数据，并观察数据结构 萼片长度（sepal length），萼片宽度()，花瓣长度（petal length），花瓣宽度，种类
        print(df.tail(n=20))

        # 0到100行，第5列
        y = df.iloc[0:100, 4].values
        # 将target值转数字化 Iris-setosa为-1，否则值为1
        y = np.where(y == "Iris-setosa", -1, 1)
        # 取出0到100行，第1，第三列的值
        x = df.iloc[0:100, [0, 2]].values
        fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(8, 4))
        ada1 = AdalineGD(n_iter=10, eta=0.01, ).fit(x, y)

        print(len(ada1.cost))
        ax[0].plot(range(1, len(ada1.cost), marker="o"))
        ax[0].set_xlabel("x")
        ax[0].set_ylabel("y")
        ax[0].set_title("0.01")

        ada2 = AdalineGD(n_iter=10, eta=0.001, ).fit(x, y)
        ax[1].plot(range(1, len(ada2.cost), marker="x"))
        ax[1].set_xlabel("x")
        ax[1].set_ylabel("y")
        ax[1].set_title("0.001")

        plt.show()

if __name__ == '__main__':
    adaline = Adaline()
    adaline.init()
