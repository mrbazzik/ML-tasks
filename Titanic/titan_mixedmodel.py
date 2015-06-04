import scipy as sp
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV, LinearRegression
from sklearn.metrics import precision_recall_fscore_support, precision_recall_curve, auc,confusion_matrix
from sklearn.cross_validation import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn import grid_search, cross_validation


def cleanData(df):
    df = df.loc[~df.Embarked.isnull(),:]
##    df = df.loc[~df.Fare.isnull(),:]
    df.Fare.loc[df.Fare.isnull()] = df.Fare.loc[~df.Fare.isnull()].mean()
    ##df = df.loc[~pd.isnull(df.Age),:]
    x = df.loc[:,['Name','Pclass','Sex','Age','SibSp','Parch','Fare','Embarked', 'Cabin','Ticket']]
    if 'Survived' in df.columns:
        y = df.loc[:,"Survived"]
    else:
        y = []


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

    x['Call']=[0]*len(x)
    x['Call']=[('Mr.' in x.loc[k,'Name'])+1 for k in x.index]
    x['Call']=[('Master.' in x.loc[k,'Name'])+2 for k in x.index]
    x['Call']=[('Mrs.' in x.loc[k,'Name'])+3 for k in x.index]
    x['Call']=[('Miss.' in x.loc[k,'Name'])+4 for k in x.index]


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

    ##x['CabFull']=x.Cabin.isnull()
    x = x.drop('Cabin',1)

##    x['PC'] = x.Ticket.map(lambda w:'PC' in w)
    x = x.drop('Ticket',1)

    nullAge = x.Age.isnull()
    x_Age_train = x.loc[~nullAge,:].drop('Age',1)
    y_Age_train = x.Age.loc[~nullAge]
    x_predict = x.loc[nullAge,:].drop('Age',1)
    clf = LinearRegression()
    clf.fit(x_Age_train, y_Age_train)
    x['AgeFill'] = x.Age
    x.AgeFill.loc[nullAge] = clf.predict(x_predict)
    x = x.drop('Age',1)

    x['CallAge'] =x['Call']*x['AgeFill']
    x = x.drop('Call',1)
    x = x.drop('Fare',1)
##    x = x.drop('AgeFill',1)
    
      
    return (x,y)

dftr = pd.read_csv("c:\\Users\\Basov_il\\Documents\\GitHub\\ML-tasks\\Titanic\\titanic_train.csv")
dft = pd.read_csv("c:\\Users\\Basov_il\\Documents\\GitHub\\ML-tasks\\Titanic\\test.csv")

x,y = cleanData(dftr)
xt,yt = cleanData(dft)

x = (x-sp.mean(x))/sp.std(x)
xt = (xt-sp.mean(xt))/sp.std(xt)

clf = LogisticRegressionCV()

print("Common:")

cv = StratifiedKFold(y, n_folds=10)
##cv = [(1,1)]

prec = {'S0':[],'S1':[]}
rec = {'S0':[],'S1':[]}
scores = []
min_score = 1
for train, test in cv:
    x_train, y_train = x.iloc[train,:], y.iloc[train]
    x_test, y_test = x.iloc[test,:], y.iloc[test]

##    x_train, y_train = x,y
##    x_test, y_test = xt,yt

    clf.fit(x_train, y_train)
    y_p=clf.predict(x_test)

    females=sp.round_(x_train.SexN,2)==-1.36
    xf = x_train.loc[females,:].drop(['SexN','SexClass','SexParch','Mr','Master','Rev'],1)
    yf = y_train.loc[females]
    clf.fit(xf, yf)
    yf_p =clf.predict(x_test.drop(['SexN','SexClass','SexParch','Mr','Master','Rev'],1))
    
    
    males=sp.round_(x_train.SexN,2)==0.74
    xm = x_train.loc[males,:].drop(['SexN','SexClass','SexParch','Miss','Mrs'],1)
    ym = y_train.loc[males]
    clf.fit(xm, ym)
    ym_p =clf.predict(x_test.drop(['SexN','SexClass','SexParch','Miss','Mrs'],1))

    for i in range(0,len(x_test)):
        if (x_test.iloc[i,:].SexN == 1) & (y_p[i] == 0):
            y_p[i]=yf_p[i]
        else:
            if (x_test.iloc[i,:].SexN == 2) & (y_p[i] == 1):
                y_p[i]=ym_p[i]
    
    cm = confusion_matrix(y_test, y_p)
    prec['S0'].append(cm[0,0]/(cm[0,0]+cm[1,0]))
    prec['S1'].append(cm[1,1]/(cm[0,1]+cm[1,1]))
    rec['S0'].append(cm[0,0]/(cm[0,0]+cm[0,1]))
    rec['S1'].append(cm[1,1]/(cm[1,0]+cm[1,1]))
    score = (cm[0,0]+cm[1,1])/(cm[0,0]+cm[0,1]+cm[1,0]+cm[1,1])
    scores.append(score)
    if score<min_score:
        min_score=score

print("For 0:")
print("Prec = %s, Rec = %s"%(sp.mean(prec['S0']), sp.mean(rec['S0'])))
print("For 1:")
print("Prec = %s, Rec = %s"%(sp.mean(prec['S1']), sp.mean(rec['S1'])))
print("Score = %s, Min score = %s"%(sp.mean(scores), min_score))

##dfr = pd.DataFrame({"PassengerId":dft.PassengerId, "Survived":y_p})
##dfr.to_csv('c:\\Users\\Basov_il\\Documents\\GitHub\\ML-tasks\\Titanic\\titanic_res.csv', index=False, columns=['PassengerId','Survived'])
