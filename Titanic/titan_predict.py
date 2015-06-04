import scipy as sp
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.metrics import precision_recall_fscore_support, precision_recall_curve, auc
from sklearn.cross_validation import KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn import grid_search, cross_validation

def cleanData(df):
    df = df.loc[~pd.isnull(df.Embarked),:]
    x = df.loc[:,['Name','Pclass','Sex','Age','SibSp','Parch','Fare','Embarked']]
    if 'Survived' in df.columns:
        y = df.loc[:,"Survived"]
    else:
        y=[]
    x.loc[x.Fare.isnull(),'Fare']=x.Fare.dropna().mean()
    x['SexN'] = [1]*len(x)
    x.loc[x.Sex=='male','SexN']=2
    x = x.drop("Sex",1)

    x['AgeFill'] = df.Age
    for i in range(0,3):
        for j in range(0,2):
            x.loc[(x.AgeFill.isnull()) & (x.Pclass==(i+1)) & (x.SexN==(j+1)),'AgeFill'] = x.loc[(x.Pclass==(i+1)) & (x.SexN==(j+1)),'AgeFill'].dropna().mean()
    x=x.drop('Age',1)
                                                                                                    

    x['EmbarkedN'] = [1]*len(x)
    x.loc[x.Embarked=='Q','EmbarkedN']=2
    x.loc[x.Embarked=='S','EmbarkedN']=3


    x = x.drop("Embarked",1)

    x['Mrs']=[0]*len(x)
    x['Mrs']=[('Mrs.' in x.loc[k,'Name']) for k in x.index]
    x['Mr']=[0]*len(x)
    x['Mr']=[('Mr.' in x.loc[k,'Name']) for k in x.index]
    x['Master']=[0]*len(x)
    x['Master']=[('Master.' in x.loc[k,'Name']) for k in x.index]
    x['Miss']=[0]*len(x)
    x['Miss']=[('Miss.' in x.loc[k,'Name']) for k in x.index]
    x['Dr']=[0]*len(x)
    x['Dr']=[('Dr.' in x.loc[k,'Name']) for k in x.index]
    x['Rev']=[0]*len(x)
    x['Rev']=[('Rev.' in x.loc[k,'Name']) for k in x.index]

    x = x.drop("Name",1)

    x['ClassFare']=x['Pclass']*x['Fare']
    x['SexParch']=x['SexN']*x['Parch']
    x['SexClass']=x['SexN']*x['Pclass']

    x = (x-sp.mean(x))/sp.std(x)
    return (x,y)

df = pd.read_csv("c:\\Users\\Basov_il\\Documents\\GitHub\\ML-tasks\\Titanic\\titanic_train.csv")
dft = pd.read_csv("c:\\Users\\Basov_il\\Documents\\GitHub\\ML-tasks\\Titanic\\test.csv")

x,y = cleanData(df)
xt,yt = cleanData(dft)

clfgs = LogisticRegressionCV()
clfgs.fit(x, y)

yt = clfgs.predict(xt)
dfr = pd.DataFrame({"PassengerId":dft.PassengerId, "Survived":yt.astype(object)})
dfr.to_csv('c:\\Users\\Basov_il\\Documents\\GitHub\\ML-tasks\\Titanic\\titanic_res.csv', index=False, columns=['PassengerId','Survived'])
