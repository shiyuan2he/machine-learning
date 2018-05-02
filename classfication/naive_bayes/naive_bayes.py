# -*- coding: utf-8 -*-
import numpy as np
# 朴素贝叶斯算法
class NaiveBayes():

    def N(self, x, mu, std):
        """
        标准正态分布
        """
        par = 1/(np.sqrt(2*np.pi)*std)
        return par*np.exp(-(x-mu)**2/2/std**2)
    def logN(self, x, class_type):
        """
        标准正态分布对数
        """
        if class_type==0:
            return np.log2(self.N(x, self.mu1, self.std1))
        else:
            return np.log2(self.N(x, self.mu2, self.std2))
    def fit(self, X, y):
        """
        训练过程为对于数据的统计
        """
        X1 = X[y==0]
        X2 = X[y==1]
        self.mu1 = np.mean(X1, axis=0)
        self.mu2 = np.mean(X2, axis=0)
        self.std1 = np.std(X1, axis=0)
        self.std2 = np.std(X2, axis=0)
    def predict_proba(self, xx):
        """
        预测过程
        """
        prb = []
        for x in xx:
            prb1_log = np.sum(self.logN(x, 0))
            prb2_log = np.sum(self.logN(x, 1))
            prb1 = 2 ** prb1_log
            prb2 = 2 ** prb2_log
            prb1 = prb1 / (prb1 + prb2)
            prb2 = prb2 / (prb1 + prb2)
            prb.append([prb1, prb2])
        return np.array(prb)

    def predict(self, x):
        """
        预测某一点的结果
        :param x: 具体某一个样本的特征信息
        :return: 预测类别
        """
        prob = self.predict_proba(x)
        return np.argmax(prob, axis=1)