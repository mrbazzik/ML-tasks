import scipy as sp
import pandas as pd
from sklearn.cluster import KMeans
def distance(v1, v2):
    return sp.sum((v1-v2)**2)

df = pd.read_csv("c:\\Users\\Basov_il\\Documents\\GitHub\\ML-tasks\\Titanic\\titanic_train.csv")
df = df.loc[~pd.isnull(df.Embarked),:]

x = df.loc[:,['Pclass','Sex','Age','SibSp','Parch','Fare','Embarked']]
y = df.loc[:,"Survived"]


x.Age = x.Age.fillna(x.Age.mean())

x['EmbarkedN'] = [1]*len(x)
x.loc[x.Embarked=='Q','EmbarkedN']=2
x.loc[x.Embarked=='S','EmbarkedN']=3
x = x.drop("Embarked",1)

x['SexN'] = [0]*len(x)
x.SexN = x.Sex=='male'
x = x.drop("Sex",1)

x = (x-sp.mean(x))/sp.std(x)

km = KMeans(n_clusters=10, init='random')
km.fit(x)
print(km.cluster_centers_)

