import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
import matplotlib.pyplot as plt
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import BernoulliNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import svm
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesRegressor

df=pd.read_json("C:/Users/JAY MATAJI/OneDrive/Desktop/Datascience intern/train cuisine.json")
test=pd.read_json("C:/Users/JAY MATAJI/OneDrive/Desktop/Datascience intern/test cuisine.json")
# df.head()
# df.info()
# df.describe()
df['source']='train'
test['source']='test'
df=pd.concat([df,test])
df=df.dropna()
df = df.sample(n=15000)
print(df)

d_c=['greek','southern_us' ,'filipino' ,'indian' ,'jamaican', 'spanish', 'italian','mexican', 'chinese' ,'british' ,'thai' ,'vietnamese' ,'cajun_creole','brazilian' ,'french' ,'japanese' ,'irish' ,'korean' ,'moroccan' ,'russian']
x=df['ingredients']
y=df['cuisine'].apply(d_c.index)
df['all_ingredients']=df['ingredients'].map(";".join)
cv=CountVectorizer()
x=cv.fit_transform(df['all_ingredients'].values)
selector = ExtraTreesRegressor()
selector.fit(x,y)
X_train,X_test,y_train,y_test=train_test_split(x,y,random_state=0,test_size=0.3)

names = ['Logistic Regression ', "GradientBoostingClasifier", "RandomForestClassifier", "Decision_Tree_Classifier","SVC", "MLPClassifier","MultinomialClassifier"]
regressors = [
LogisticRegression(random_state=45),
GradientBoostingClassifier(n_estimators=12),
RandomForestClassifier(random_state=2),
DecisionTreeClassifier(random_state=42),
svm.SVC(),
MLPClassifier(solver='lbfgs',alpha=1e-5,hidden_layer_sizes=(5,2),random_state=2)
,MultinomialNB()]

scores = []
mean_score=[]
for name, clf in zip(names, regressors):
    clf.fit(X_train,y_train)
    score = accuracy_score(y_test,clf.predict(X_test))
    mse= 1-score
    scores.append(score)
    mean_score.append(mse)

scores_df = pd.DataFrame()
scores_df['name           '] = names
scores_df['accuracy'] = scores
scores_df['Mean_squared_error'] = mean_score
print(scores_df.sort_values('accuracy', ascending= False))

'''    ANS 

             name             accuracy  Mean_squared_error
0       Logistic Regression   0.757111            0.242889
4                        SVC  0.741556            0.258444
2     RandomForestClassifier  0.727778            0.272222
6      MultinomialClassifier  0.708000            0.292000
1  GradientBoostingClasifier  0.636222            0.363778
3   Decision_Tree_Classifier  0.604444            0.395556
5              MLPClassifier  0.582222            0.417778

'''
