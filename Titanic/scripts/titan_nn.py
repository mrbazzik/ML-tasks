import scipy as sp
import pandas as pd
def distance(v1, v2):
    return sp.sum((v1-v2)**2)

import scipy as sp
import pandas as pd
df = pd.read_csv("c:\\Users\\Basov_il\\Documents\\GitHub\\ML-tasks\\Titanic\\titanic_train.csv")
df = df.loc[~pd.isnull(df.Embarked),:]

x = df.loc[:,['Pclass','Sex','Age','SibSp','Parch','Fare','Embarked']]
y = df.loc[:,"Survived"]

x.Age = x.Age.fillna(x.Age.mean())
x['SexN'] = [0]*len(x)
x.SexN = x.Sex=='male'
x = x.drop("Sex",1)

x['EmbarkedN'] = [1]*len(x)
x.loc[x.Embarked=='Q','EmbarkedN']=2
x.loc[x.Embarked=='S','EmbarkedN']=3
x = x.drop("Embarked",1)

x = (x-sp.mean(x))/sp.std(x)
##print(sp.std(x))
##print(x.shape)
##print(x.describe())
##print(x.head())
biganswer=sp.array([])
for n in x.index:
    print(n)
    ch = sp.arange(0,len(x))
    x_train = x.loc[ch!=n,:]
    ##ch = sp.delete(ch,n, 0)
    ##x_train = x.loc[ch,:]
    x_test = x.loc[n,:]
    y_train = y.loc[ch!=n]
    y_test = y.loc[n]
    ##answer=sp.array([])
   ## print(x_train.shape)
   ## print(x_train.index)
    ##for i in x_test.index:
    dists = sp.array([distance(x_test, x_train.loc[k,:]) for k in x_train.index])
    cum = 0
    for j in sp.arange(0,5):
        nearest=dists.argmin()
        cum+=y_train.iloc[nearest]
        dists[nearest] = 9999
    answer = cum/5.0>0.5
##print(answer)
##print(y_test
    biganswer = sp.append(biganswer, answer)
print(sp.mean(biganswer==y))
print(sp.mean(biganswer[y==1]==1))
print(sp.mean(y[biganswer==1]==1))
    
