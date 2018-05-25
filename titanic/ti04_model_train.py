import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.svm import LinearSVC

df_train = pd.read_csv("input/train.csv")
df_test = pd.read_csv("input/test.csv")

# 合并数据
df_data = df_train.append(df_test)

NUMERIC_COLUMNS=['Pclass','Age','SibSp','Parch','Fare']

# create test and training data
data_to_train = df_train[NUMERIC_COLUMNS].fillna(-1000)
y=df_train['Survived']
X=data_to_train
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,random_state=21, stratify=y)


clf = SVC()
clf.fit(X_train, y_train)
linear_svc = LinearSVC()

# Print the accuracy
print("Accuracy: {}".format(clf.score(X_test, y_test)))