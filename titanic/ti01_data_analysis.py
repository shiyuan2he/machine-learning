import numpy as np
import pandas as pd
import os
print(os.listdir("input"))

df_train = pd.read_csv("input/train.csv")
df_test = pd.read_csv("input/test.csv")

# 合并数据
df_data = df_train.append(df_test)

# 观察数据行和列数
print(df_train.shape)
print("-------------")
print(df_test.shape)

# 看列名
"""
    'PassengerId':   乘客id
    'Pclass':
    'Name':         乘客名称
    'Sex'：          乘客性别
    'Age'：          乘客年龄
    'SibSp'：
    'Parch'： 
    'Ticket'：
    'Fare'：
    'Cabin'：
    'Embarked'
    'Survived'：
"""
print(df_train.columns)
print(df_test.columns)

# 看数据类型
"""
    PassengerId    891 non-null int64
    Survived       891 non-null int64
    Pclass         891 non-null int64
    Name           891 non-null object
    Sex            891 non-null object
    Age            714 non-null float64
    SibSp          891 non-null int64
    Parch          891 non-null int64
    Ticket         891 non-null object
    Fare           891 non-null float64
    Cabin          204 non-null object
    Embarked       889 non-null object
    
    其中有的类数据类型是object，也就是字符类型数据。
    这种类型数据
    将其转换为数值型数据。
"""
print(df_train.info())

# 查看缺失数据
print(pd.isnull(df_data).sum())
print(pd.isnull(df_train).sum())
print(pd.isnull(df_test).sum())
"""
    Age             263
    Cabin          1014
    Embarked          2
    Fare              1
    Name              0
    Parch             0
    PassengerId       0
    Pclass            0
    Sex               0
    SibSp             0
    Survived        418
    Ticket            0
    dtype: int64
    df_train共177个确实值，df_test 86个缺失值，age对于预测结果很重要，故有补全的必要
    Embarked，Fare分别有2，1个缺失值，对预测结果无关紧要
    Cabin是缺失值比较严重的
"""
print(df_test.head())

