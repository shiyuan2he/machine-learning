import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

df_train = pd.read_csv("input/train.csv")
df_test = pd.read_csv("input/test.csv")

# 合并数据
df_data = df_train.append(df_test)

grid = sns.FacetGrid(df_train, col="Pclass", row="Sex", hue="Survived", palette="seismic")
grid = grid.map(plt.scatter, "PassengerId", "Age")
grid.add_legend()
plt.show()
# g = sns.pairplot(df_train[[u'Survived', u'Pclass', u'Sex', u'Age', u'Parch', u'Fare', u'Embarked']],
#                  hue='Survived', palette='seismic', size=4,
#                  diag_kind='kde', diag_kws=dict(shade=True), plot_kws=dict(s=50))
# g.set(xticklabels=[])



# df_train['Sex'].value_counts().plot.pie()
# plt.gca().set_aspect('equal') #most sponsored projects are teacher-led
# plt.title("gender distribution", fontsize=20)
# plt.show()

#sns.violinplot(data = df_train, x = "Sex", y = "Survived")
