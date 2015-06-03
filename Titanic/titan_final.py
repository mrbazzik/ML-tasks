import scipy as sp
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.metrics import precision_recall_fscore_support, precision_recall_curve, auc
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

x['AgeFill'] = df.Age
for i in range(0,3):
    for j in range(0,2):
        x.loc[(x.AgeFill.isnull()) & (x.Pclass==(i+1)) & (x.SexN==(j+1)),'AgeFill'] = x.loc[(x.Pclass==(i+1)) & (x.SexN==(j+1)),'AgeFill'].dropna().mean()
x=x.drop('Age',1)
                                                                                                

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
##x['SexCall']=x['SexN']*x['Call']
##x['ClassCall']=x['Pclass']*x['Call']
##x['CallParch']=x['Call']*x['Parch']

##x['MrsParch']=x['Mrs']*x['Parch']
##x['MrsSibSp']=x['Mrs']*x['SibSp']

##x['SexCallClass']=x['SexN']*x['Call']*x['Pclass']

##x['AgeClass']=x['Age']*x['Pclass']
##x['Family']=x['Parch']+x['SibSp']
##x['SexAge']=x['SexN']*x['Age']
x = (x-sp.mean(x))/sp.std(x)


##n_train = 500
##x_train = x.iloc[:n_train,:]
##y_train = y.iloc[:n_train]
##x_test = x.iloc[n_train:,:]
##y_test = y.iloc[n_train:]

##x_test = x_test[~pd.isnull(x_test.Age)]
##y_test = y_test[~pd.isnull(x_test.Age)]

cv = KFold(n=len(x), n_folds=10)
clf = LogisticRegression()
c_range = sp.logspace(-5,4,10)
##clfgs = grid_search.GridSearchCV(clf,{'C':c_range})
clfgs = LogisticRegressionCV()

forest = RandomForestClassifier(n_estimators=100)
scores_reg = []
aucs_reg=[]
scores_for = []
aucs_for=[]
scores = []
aucs=[]
for train, test in cv:
    x_train, y_train = x.iloc[train,:], y.iloc[train]
    x_test, y_test = x.iloc[test,:], y.iloc[test]

    clfgs.fit(x_train, y_train)
    pr = clfgs.predict_proba(x_test)[:,1]
    scores_reg.append(clfgs.score(x_test, y_test))
    precision, recall, thres = precision_recall_curve(y_test, clfgs.predict(x_test))
    aucs_reg.append(auc(recall, precision))

    forest.fit(x_train, y_train)
    pr = forest.predict_proba(x_test)[:,1]
    scores_for.append(forest.score(x_test, y_test))
    precision, recall, thres = precision_recall_curve(y_test, forest.predict(x_test))
    aucs_for.append(auc(recall, precision))

    x_total = sp.vstack((clfgs.predict(x_test),forest.predict(x_test))).transpose()
    y_total = y_test
    cv1 = KFold(n=len(x_total), n_folds=5)
    scoresf = []
    aucsf=[]
    for train1, test1 in cv1:
        x_train_total = x_total[train1,:]
        y_train_total = y_total[train1]
        x_test_total = x_total[test1,:]
        y_test_total = y_total[test1]
        clfgs.fit(x_train_total, y_train_total)
        pr = clfgs.predict_proba(x_test_total)[:,1]
        scoresf.append(clfgs.score(x_test_total, y_test_total))
        precision, recall, thres = precision_recall_curve(y_test_total, clfgs.predict(x_test_total))
        aucsf.append(auc(recall, precision))
    scores.append(sp.mean(scoresf))
    aucs.append(sp.mean(aucsf))

print("Logistic regression")    
print("Score = %s, Auc = %s"%(sp.mean(scores_reg), sp.mean(aucs_reg)))
print("Random forest")    
print("Score = %s, Auc = %s"%(sp.mean(scores_for), sp.mean(aucs_for)))
print("Total")    
print("Score = %s, Auc = %s"%(sp.mean(scores), sp.mean(aucs)))
