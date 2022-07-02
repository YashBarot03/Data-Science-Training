# Numerical to Categorical
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import pandas as pd

df=pd.read_csv("C:/Users/JAY MATAJI/OneDrive/Desktop/Datascience intern/IRIS.csv")
print(df)


x=df.drop(['species'],axis=1)
print(x)

y=df['species']
print(y)

rf=RandomForestClassifier()
df['sepal_length']=pd.cut(df['sepal_length'],3,labels=['0','1','2'])
df['sepal_width']=pd.cut(df['sepal_width'],3,labels=['0','1','2'])
df['petal_length']=pd.cut(df['petal_length'],3,labels=['0','1','2'])
df['petal_width']=pd.cut(df['petal_width'],3,labels=['0','1','2'])

le=LabelEncoder()
le.fit(y)
y=le.transform(y)
print(y)

x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=0,test_size=0.3)
rf.fit(x_train,y_train)
y_pred=rf.predict(x_test)
print('Random Forest: ',accuracy_score(y_test,y_pred))

#Categorical to Numerical
le = LabelEncoder()
le.fit(y)
y = le.transform(y)
# "C:\Program Files\Python39\python.exe" "C:/Users/JAY MATAJI/Downloads/Sorting of Mixed Data.py"
#      sepal_length  sepal_width  petal_length  petal_width         species
# 0             5.1          3.5           1.4          0.2     Iris-setosa
# 1             4.9          3.0           1.4          0.2     Iris-setosa
# 2             4.7          3.2           1.3          0.2     Iris-setosa
# 3             4.6          3.1           1.5          0.2     Iris-setosa
# 4             5.0          3.6           1.4          0.2     Iris-setosa
# ..            ...          ...           ...          ...             ...
# 145           6.7          3.0           5.2          2.3  Iris-virginica
# 146           6.3          2.5           5.0          1.9  Iris-virginica
# 147           6.5          3.0           5.2          2.0  Iris-virginica
# 148           6.2          3.4           5.4          2.3  Iris-virginica
# 149           5.9          3.0           5.1          1.8  Iris-virginica
#
# [150 rows x 5 columns]
#      sepal_length  sepal_width  petal_length  petal_width
# 0             5.1          3.5           1.4          0.2
# 1             4.9          3.0           1.4          0.2
# 2             4.7          3.2           1.3          0.2
# 3             4.6          3.1           1.5          0.2
# 4             5.0          3.6           1.4          0.2
# ..            ...          ...           ...          ...
# 145           6.7          3.0           5.2          2.3
# 146           6.3          2.5           5.0          1.9
# 147           6.5          3.0           5.2          2.0
# 148           6.2          3.4           5.4          2.3
# 149           5.9          3.0           5.1          1.8
#
# [150 rows x 4 columns]
# 0         Iris-setosa
# 1         Iris-setosa
# 2         Iris-setosa
# 3         Iris-setosa
# 4         Iris-setosa
#             ...
# 145    Iris-virginica
# 146    Iris-virginica
# 147    Iris-virginica
# 148    Iris-virginica
# 149    Iris-virginica
# Name: species, Length: 150, dtype: object
# [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
#  0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
#  1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 2 2 2 2 2 2 2 2 2 2 2
#  2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2
#  2 2]
# Random Forest:  0.9777777777777777
#
# Process finished with exit code 0
#
