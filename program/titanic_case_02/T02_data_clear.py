class DataClear(object):

    def execute(self, data_full):
        print("***************02-数据清洗***************")
        """
            1.处理float64缺失值：Age,Fare
            这部分是数值型数据，用均值填充较合理：
        """
        # print(data_full.info())
        data_full['Age'] = data_full['Age'].fillna(data_full['Age'].mean())
        data_full['Fare'] = data_full['Fare'].fillna(data_full['Fare'].mean())
        # print(data_full.info())
 
        """
            处理object缺失：Cabin,Embarked
            这部分是类型数据，其中cabin缺失的较多，达到了77.46%，则此时其众数就是unknown，
            这时直接用U表示unknown较合适；
            而Embarked只有2条缺失，用出现频率最高的，即众数来填充较合理。
            首先获取下Embarked的众数：
            或缺Embarked的众数
            S    914
            C    270
            Q    123
        """

        # print(data_full['Cabin'].value_counts())
        # U==Unknow
        data_full["Cabin"] = data_full["Cabin"].fillna("U")
        #   print(data_full['Embarked'].value_counts())
        data_full["Embarked"] = data_full["Embarked"].fillna("S")
        print(data_full.info())
        """
            以上输出的数据就是比较赶紧规范的数据
        """
        return data_full

