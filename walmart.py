import numpy
import pandas as pd
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA

train = "C:/Users/JAY MATAJI/OneDrive/Desktop/Datascience intern/train walmart.csv"
test = "C:/Users/JAY MATAJI/OneDrive/Desktop/Datascience intern/test walmart.csv"

df = pd.concat([train, test])
print(df)
df['Date'] = pd.to_datetime(df['Date'])
df['days']=df['Date'].dt.day
df['month']=df['Date'].dt.month
df['year']=df['Date'].dt.year
df['WeekOfYear'] = df.Date.dt.isocalendar().week
df.drop('Date',axis=1,inplace=True)

# print(df.isnull().all())

x=df.drop('Weekly_Sales',axis=1)
y=df['Weekly_Sales']

#Feature Extraction
pca = PCA(n_components=3)
fit = pca.fit(x)

#Splitting Data
X_Train, X_Test, Y_Train, Y_Test = train_test_split(x, y, test_size=0.3)
names = ['Linear Regression', "KNN", "Linear_SVM","Gradient_Boosting", "Decision_Tree", "Random_Forest"]
regressors = [
    LinearRegression(),
    KNeighborsRegressor(n_neighbors=3),
    SVR(),
    GradientBoostingRegressor(n_estimators=100),
    DecisionTreeRegressor(max_depth=5),
    RandomForestRegressor(max_depth=5, n_estimators=100)]

scores = []
for name, clf in zip(names, regressors):
    clf.fit(X_Train,Y_Train)
    score = clf.score(X_Test,Y_Test)
    scores.append(score)
scores_df = pd.DataFrame()
scores_df['name'] = names
scores_df['accuracy'] = scores
print(scores_df.sort_values('accuracy', ascending= False))


'''
                name  accuracy
3  Gradient_Boosting  0.901871
4      Decision_Tree  0.694620
5      Random_Forest  0.688970
1                KNN  0.176774
0  Linear Regression  0.114684
2         Linear_SVM -0.013785
'''