import scipy as sp
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV, LinearRegression
from sklearn.metrics import precision_recall_fscore_support, precision_recall_curve, auc,confusion_matrix
from sklearn.cross_validation import KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn import grid_search, cross_validation

df = pd.read_csv("c:\\Users\\Basov_il\\Documents\\GitHub\\ML-tasks\\Titanic\\titanic_train.csv")
df = df.loc[~pd.isnull(df.Embarked),:]
##df = df.loc[~pd.isnull(df.Age),:]


x = df.loc[:,['Name','Pclass','Sex','Age','SibSp','Parch','Fare','Embarked']]
y = df.loc[:,"Survived"]


##x.Age = x.Age.fillna(x.Age.mean())
x['SexN'] = [1]*len(x)
x.loc[x.Sex=='male','SexN']=2
x = x.drop("Sex",1)

                                                                                         

x['EmbarkedN'] = [1]*len(x)
x.loc[x.Embarked=='Q','EmbarkedN']=2
x.loc[x.Embarked=='S','EmbarkedN']=3


##x['Q']=x.Embarked.map({'Q':1, 'C':0, 'S':0})
##x['C']=x.Embarked.map({'Q':0, 'C':1, 'S':0})
##x['S']=x.Embarked.map({'Q':0, 'C':0, 'S':1})

x = x.drop("Embarked",1)

##x['Call']=[0]*len(x)
##x['Call']=[('Mr.' in x.loc[k,'Name'])+1 for k in x.index]
##x['Call']=[('Master.' in x.loc[k,'Name'])+2 for k in x.index]
##x['Call']=[('Mrs.' in x.loc[k,'Name'])+3 for k in x.index]
##x['Call']=[('Miss.' in x.loc[k,'Name'])+4 for k in x.index]


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

nullAge = x.Age.isnull()
x_Age_train = x.loc[~nullAge,:].drop('Age',1)
y_Age_train = x.Age.loc[~nullAge]
x_predict = x.loc[nullAge,:].drop('Age',1)
clf = LinearRegression()
clf.fit(x_Age_train, y_Age_train)
x['AgeFill'] = x.Age
x.AgeFill.loc[nullAge] = clf.predict(x_predict)
x = x.drop('Age',1)
y = df.loc[:,"Survived"]



females=x.SexN==1
xf = x.loc[females,:].drop(['SexN','SexClass','SexParch','Mr','Master','Rev'],1)
yf = y.loc[females]

males=x.SexN==2
xm = x.loc[males,:].drop(['SexN','SexClass','SexParch','Miss','Mrs'],1)
ym = y.loc[males]

x = (x-sp.mean(x))/sp.std(x)
xf = (xf-sp.mean(xf))/sp.std(xf)
xm = (xm-sp.mean(xm))/sp.std(xm)


clf = LogisticRegressionCV()

print("Females:")
cv = KFold(n=len(xf), n_folds=10)
prec = {'S0':[],'S1':[]}
rec = {'S0':[],'S1':[]}
for train, test in cv:
    x_train, y_train = xf.iloc[train,:], yf.iloc[train]
    x_test, y_test = xf.iloc[test,:], yf.iloc[test]
    clf.fit(x_train, y_train)
    y_p=clf.predict(x_test)
    cm = confusion_matrix(y_test, y_p)
    prec['S0'].append(cm[0,0]/(cm[0,0]+cm[1,0]))
    prec['S1'].append(cm[1,1]/(cm[0,1]+cm[1,1]))
    rec['S0'].append(cm[0,0]/(cm[0,0]+cm[0,1]))
    rec['S1'].append(cm[1,1]/(cm[1,0]+cm[1,1]))
print("For 0:")
print("Prec = %s, Rec = %s"%(sp.mean(prec['S0']), sp.mean(rec['S0'])))
print("For 1:")
print("Prec = %s, Rec = %s"%(sp.mean(prec['S1']), sp.mean(rec['S1'])))


print("Males:")
cv = KFold(n=len(xm), n_folds=10)
prec = {'S0':[],'S1':[]}
rec = {'S0':[],'S1':[]}
for train, test in cv:
    x_train, y_train = xm.iloc[train,:], ym.iloc[train]
    x_test, y_test = xm.iloc[test,:], ym.iloc[test]
    clf.fit(x_train, y_train)
    y_p=clf.predict(x_test)
    cm = confusion_matrix(y_test, y_p)
    prec['S0'].append(cm[0,0]/(cm[0,0]+cm[1,0]))
    prec['S1'].append(cm[1,1]/(cm[0,1]+cm[1,1]))
    rec['S0'].append(cm[0,0]/(cm[0,0]+cm[0,1]))
    rec['S1'].append(cm[1,1]/(cm[1,0]+cm[1,1]))
print("For 0:")
print("Prec = %s, Rec = %s"%(sp.mean(prec['S0']), sp.mean(rec['S0'])))
print("For 1:")
print("Prec = %s, Rec = %s"%(sp.mean(prec['S1']), sp.mean(rec['S1'])))


print("Common:")
cv = KFold(n=len(x), n_folds=10)
prec = {'S0':[],'S1':[]}
rec = {'S0':[],'S1':[]}
for train, test in cv:
    x_train, y_train = x.iloc[train,:], y.iloc[train]
    x_test, y_test = x.iloc[test,:], y.iloc[test]
    clf.fit(x_train, y_train)
    y_p=clf.predict(x_test)
    cm = confusion_matrix(y_test, y_p)
    prec['S0'].append(cm[0,0]/(cm[0,0]+cm[1,0]))
    prec['S1'].append(cm[1,1]/(cm[0,1]+cm[1,1]))
    rec['S0'].append(cm[0,0]/(cm[0,0]+cm[0,1]))
    rec['S1'].append(cm[1,1]/(cm[1,0]+cm[1,1]))

print("For 0:")
print("Prec = %s, Rec = %s"%(sp.mean(prec['S0']), sp.mean(rec['S0'])))
print("For 1:")
print("Prec = %s, Rec = %s"%(sp.mean(prec['S1']), sp.mean(rec['S1'])))

