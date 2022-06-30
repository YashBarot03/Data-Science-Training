import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

df=pd.read_csv("C:/Users/CC-072/Desktop/Dataset/IRIS.csv")
print(df)
rf=RandomForestClassifier(random_state=1)
lr=LogisticRegression(random_state=0)
gbm=GradientBoostingClassifier(n_estimators=10)
dt=DecisionTreeClassifier(random_state=0)
sv=svm.SVC()
nn=MLPClassifier(solver='lbfgs',alpha=1e-5, hidden_layer_sizes=(5,2),random_state=0)
nb=GaussianNB()

logr=LogisticRegression()
x=df.drop(['species'],axis=1)
print(x)

y=df['species']
print(y)

# rf=RandomForestClassifier
# x=df.drop(['species'],axis=1)
# print(x)
#
# y=df['species']
# print(y)
#
# gbm=GradientBoostingClassifier
# x=df.drop(['species'],axis=1)
# print(x)
#
# y=df['species']
# print(y)
#
# dt=DecisionTreeClassifier
# x=df.drop(['species'],axis=1)
# print(x)
#
# y=df['species']
# print(y)
#
# svc=svm
# x=df.drop(['species'],axis=1)
# print(x)
#
# y=df['species']
# print(y)
#
# nn=MLPClassifier
# x=df.drop(['species'],axis=1)
# print(x)
#
# y=df['species']
# print(y)
#
# nb=GaussianNB
# x=df.drop(['species'],axis=1)
# print(x)
#
# y=df['species']
# print(y)

x_train, x_test, y_train, y_test=train_test_split(x,y,random_state=0,test_size=0.3)
train=logr.fit(x_train,y_train)
y_pred1=logr.predict(x_test)
print(accuracy_score(y_test,y_pred1))

#x_train, x_test, y_train, y_test=train_test_split(x,y,random_state=0,test_size=0.3)rf.fit(x_train,y_train)
rf.fit(x_train,y_train)
y_pred2=rf.predict(x_test)
print(accuracy_score(y_test,y_pred2))

#x_train, x_test, y_train, y_test=train_test_split(x,y,random_state=0,test_size=0.3)
gbm.fit(x_train,y_train)
y_pred3=gbm.predict(x_test)
print(accuracy_score(y_test,y_pred3))

#x_train, x_test, y_train, y_test=train_test_split(x,y,random_state=0,test_size=0.3)
dt.fit(x_train,y_train)
y_pred4=dt.predict(x_test)
print(accuracy_score(y_test,y_pred4))

#x_train, x_test, y_train, y_test=train_test_split(x,y,random_state=0,test_size=0.3)
sv.fit(x_train,y_train)
y_pred5=sv.predict(x_test)
print(accuracy_score(y_test,y_pred5))

#x_train, x_test, y_train, y_test=train_test_split(x,y,random_state=0,test_size=0.3)
nn.fit(x_train,y_train)
y_pred6=nn.predict(x_test)
print(accuracy_score(y_test,y_pred6))

x_train, x_test, y_train, y_test=train_test_split(x,y,random_state=0,test_size=0.3)
nb.fit(x_train,y_train)
y_pred7=nb.predict(x_test)
print(accuracy_score(y_test,y_pred7))



# x_train, x_test, y_train, y_test=train_test_split(x,y,random_state=0,test_size=0.3)
# train=logr.fit(x_train,y_train)
# y_pred=logr.predict(x_test)
# print(accuracy_score(y_test,y_pred))

# 0.9777777777777777
# 0.9777777777777777
# 0.9777777777777777
# 0.9777777777777777
# 0.9777777777777777
# 0.24444444444444444
# 1.0
# 0.9777777777777777







