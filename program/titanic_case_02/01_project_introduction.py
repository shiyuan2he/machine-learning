import pandas as pd
import os

print(os.listdir("./../../data/titanic"))
data_train = pd.read_csv("./../../data/titanic/train.csv")
data_test = pd.read_csv("./../../data/titanic/test.csv")
data_full = data_train.append(data_test)

# 观察数据的形状
print(data_train.shape)
print(data_test.shape)
print(data_full.shape)

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
print(data_train.columns)
print(data_test.columns)