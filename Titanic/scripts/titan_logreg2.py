import scipy as sp
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import precision_recall_fscore_support, precision_recall_curve, auc
from sklearn.cross_validation import KFold



df = pd.read_csv("c:\\Users\\Basov_il\\Documents\\GitHub\\ML-tasks\\Titanic\\titanic_train.csv")
df = df.loc[~pd.isnull(df.Embarked),:]
##df = df.loc[~pd.isnull(df.Age),:]


x = df.loc[:,['Name','Pclass','Sex','Age','SibSp','Parch','Fare','Embarked']]
y = df.loc[:,"Survived"]


##x.Age = x.Age.fillna(x.Age.mean())
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
##x = x.drop("Embarked",1)

##x['Q']=x.Embarked.map({'Q':1, 'C':0, 'S':0})
##x['C']=x.Embarked.map({'Q':0, 'C':1, 'S':0})
##x['S']=x.Embarked.map({'Q':0, 'C':0, 'S':1})

x = x.drop("Embarked",1)

x['Call']=[0]*len(x)
x['Call']=[('Mr.' in x.loc[k,'Name'])+1 for k in x.index]
x['Call']=[('Master.' in x.loc[k,'Name'])+2 for k in x.index]
x['Call']=[('Mrs.' in x.loc[k,'Name'])+3 for k in x.index]
x['Call']=[('Miss.' in x.loc[k,'Name'])+4 for k in x.index]


x['Mrs']=[0]*len(x)
x['Mrs']=[('Mrs.' in x.loc[k,'Name']) for k in x.index]

x = x.drop("Name",1)

x['ClassFare']=x['Pclass']*x['Fare']
x['SexParch']=x['SexN']*x['Parch']
x['SexClass']=x['SexN']*x['Pclass']
x['SexCall']=x['SexN']*x['Call']
##x['ClassCall']=x['Pclass']*x['Call']
x['CallParch']=x['Call']*x['Parch']

x['MrsParch']=x['Mrs']*x['Parch']
x['MrsSibSp']=x['Mrs']*x['SibSp']

x['SexCallClass']=x['SexN']*x['Call']*x['Pclass']

##x['AgeClass']=x['Age']*x['Pclass']
x['Family']=x['Parch']+x['SibSp']
##x['SexAge']=x['SexN']*x['Age']
x = (x-sp.mean(x))/sp.std(x)


n_train = 500
x_train = x.iloc[:n_train,:]
y_train = y.iloc[:n_train]
x_test = x.iloc[n_train:,:]
y_test = y.iloc[n_train:]

##x_test = x_test[~pd.isnull(x_test.Age)]
##y_test = y_test[~pd.isnull(x_test.Age)]

cv = KFold(n=len(x), n_folds=10)
clf = LogisticRegressionCV()
scores = []
aucs=[]
for train, test in cv:
    x_train, y_train = x.iloc[train,:], y.iloc[train]
    x_test, y_test = x.iloc[test,:], y.iloc[test]
    clf.fit(x_train, y_train)
    pr = clf.predict_proba(x_test)[:,1]
    scores.append(clf.score(x_test, y_test))
    precision, recall, thres = precision_recall_curve(y_test, clf.predict(x_test))
    aucs.append(auc(recall, precision))
print("Score = %s, Auc = %s"%(sp.mean(scores), sp.mean(aucs)))
