import pandas as pd
import os

class ProjectIntroduction():

    def execute(self, data_full):
        print("***************01-数据介绍***************")
        # 观察数据的形状
        # print(data_train.shape)
        # print(data_test.shape)
        # print(data_full.shape)

        # 看列名
        """
            'PassengerId':  乘客id
            'Pclass':       船舱等级
            'Name':         乘客名称
            'Sex'：         乘客性别
            'Age'：         乘客年龄
            'SibSp'：
            'Parch'： 
            'Ticket'：
            'Fare'：
            'Cabin'：       船舱号
            'Embarked':     登陆港口
            'Survived'：
        """
        # print(data_train.columns)
        # print(data_test.columns)
        # print(data_full.columns)

        # 看数据信息
        # print(data_full.info())
        # print(data_full.head())

        print(data_full.describe())
