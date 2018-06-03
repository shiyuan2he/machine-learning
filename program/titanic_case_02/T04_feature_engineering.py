import pandas as pd
class FeatureEngineering():

    def execute(self, data_full):
        print("***************04-特征工程***************")
        """
            数据分析、机器学习中，最重要的就是选取合适的数据特征，供模型算法训练。
            若训练的特征相关性高，则能事半功倍，故而特征工程是整个项目的核心所在，
            这一步做得好后面模型的正确率就高。

            那么什么是特征工程？特征工程就是最大限度地从原始数据中提取能表征原始数据的特征，
            以供机器学习算法和模型使用。

            根据前文的info可知，数据可基本分为分类数据、数值数据和序列数据，如下图所示：
            1、数值型数据可以直接使用
            2、时间序列 ==》转换成单独年、月、日
            3、分类数据 ==》用数值代替类别（One-hot编码）

            分类数据：Cabin、Embarked、Parch、SibSp、Pclass、Sex
            数值数据：Fare、Parch、Survived
            字符串数据：Name、Ticket
            序列数据：PassengerId、Age
        """
        # print(data_full.head())

        """
            分类数据主要有Sex，Cabin，Embarked，Pclass，Parch，SibSp，下面逐个清理。
            数据Sex在原数据中填充的是female、male，为了方便后面模型训练，将其转换成数值型的0、1：
        """
        sex_dict = {"male": 0, "female": 1}
        data_full["Sex"] = data_full["Sex"].map(sex_dict)
        # print(data_full["Sex"].head())

        """
            登录港口:Embarked
            接下来处理登录港口Embarked，首先看Embarked数据的前五行，了解情况：
            如下，Embarked显示的是乘客在那个港口登陆，而这又是类别数据，
            这时可用one－hot编码对这一列数据进行降维。
            即：给登陆港口C、S、Q分别建一列，如果是在该港口登陆则值为1，否则为0。
            这样每个乘客，即每一行，只会在三列中的一列为1，其余为0，
            这就实现了类别数据向数值型数据的额转化，且也实现了数据降维。
        
            具体可用pandas的get_dummies方法实现：
        """
        # print(data_full["Embarked"].head())

        df_Embarked = pd.DataFrame()
        embarked_columes = pd.get_dummies(data_full["Embarked"], prefix="Embarked")
        # print(embarked_columes.head())
        # 如上EmbarkedDF就是转换后的Embarked数据，
        # 将其添加到full中，并删除原full中的Embarked列，则Embarked的特征就准备好了，如下。
        data_full = pd.concat([data_full, embarked_columes], axis=1)
        data_full.drop("Embarked", axis=1, inplace=True)
        # print(data_full.head())

        """
            下面处理船舱等级Pclass，官网中介绍Pclass分为高中低，
            数值分别对应为1、2、3，与Embarked数据一致，
            也对其用get_dummies方法实现one－hot编码，如下：
        """
        df_Pclass = pd.DataFrame()
        pclass_columns = pd.get_dummies(data_full["Pclass"], prefix="Pclass")
        data_full = pd.concat([data_full, pclass_columns], axis=1)
        data_full.drop("Pclass", axis=1, inplace=True)
        # print(data_full.head())
        # print(data_full.shape)
        """
            船舱号Cabin
        
            下面处理船舱号Cabin数据，先看Cabin的前5行：
            可见，Cabin数据列并不规整，但也不是全无规律可循，可取每个元素的首字母进行填充，
            然后用新的首字母进行one－hot编码生存特征数据CabinDF，最后更新到full中。
        """
        df_cabin = pd.DataFrame()
        df_cabin = data_full["Cabin"].map(lambda a: a[0])
        cabin_column = pd.get_dummies(df_cabin, axis=1)
        data_full = pd.concat([data_full, cabin_column], axis=1, inplace=True)
        # print(data_full.head())
        return data_full


