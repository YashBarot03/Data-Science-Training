import pandas as pd
#Feature selection 1
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

df=pd.read_csv("C:/Users/CC-072/Desktop/Dataset/IRIS.csv")
print(df)


X=df.drop(['species'],axis=1)
print(X)

Y=df['species']
print(Y)

bestfeatures = SelectKBest(score_func=chi2, k='all')
fit = bestfeatures.fit(X,Y)
dfscores = pd.DataFrame(fit.scores_)
dfcolumns = pd.DataFrame(X.columns)
featuresScores = pd.concat([dfcolumns,dfscores], axis=1)
featuresScores.columns = ['Specs','Score']

print(featuresScores)

# Name: species, Length: 150, dtype: object
#           Specs       Score
# 0  sepal_length   10.817821
# 1   sepal_width    3.594499
# 2  petal_length  116.169847
# 3   petal_width   67.244828
#
# Process finished with exit code 0

# Feature selection using ExtraTreesClassifier
from sklearn.ensemble import ExtraTreesClassifier
import matplotlib.pyplot as plt
model=ExtraTreesClassifier()
model.fit(X,Y)
print(model.feature_importances_)
feat_importance=pd.Series(model.feature_importances_,index=X.columns)
feat_importance.nlargest(4).plot(kind='barh')
plt.show()








