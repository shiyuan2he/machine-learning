import os
import pandas as pd
# noinspection PyUnresolvedReferences
from T01_project_introduction import ProjectIntroduction
# noinspection PyUnresolvedReferences
from T02_data_clear import DataClear
# noinspection PyUnresolvedReferences
from T03_data_visualization import DataVisualization
# noinspection PyUnresolvedReferences
from T04_feature_engineering import FeatureEngineering


class Runner(object):
    def __init__(self):
        self.projectIntroductionObj = ProjectIntroduction()
        self.dataClearObj = DataClear()
        self.dataVisualizationObj = DataVisualization()
        self.featureEngineeringObj = FeatureEngineering()

    def execute(self):
        print(os.listdir("./../../data/titanic"))
        data_train = pd.read_csv("./../../data/titanic/train.csv")
        data_test = pd.read_csv("./../../data/titanic/test.csv")
        data_full = data_train.append(data_test, ignore_index=True)
        data_full = self.projectIntroductionObj.execute(data_full)
        data_full = self.dataClearObj.execute(data_full)
        data_full = self.dataVisualizationObj.execute(data_full)
        data_full = self.featureEngineeringObj.execute(data_full)

if __name__ == '__main__':

    runner = Runner()
    runner.execute()
