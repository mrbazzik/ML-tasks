import scipy as sp
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import precision_recall_curve


df = pd.read_csv("c:\\Users\\Basov_il\\Documents\\GitHub\\ML-tasks\\Titanic\\titanic_train.csv")
df = df.loc[~pd.isnull(df.Embarked),:]
df = df.loc[~pd.isnull(df.Age),:]

x = df.loc[:,['Name','Pclass','Sex','Age','SibSp','Parch','Fare','Embarked']]
y = df.loc[:,"Survived"]

##x.Age = x.Age.fillna(x.Age.mean())
x['SexN'] = [1]*len(x)
x.loc[x.Sex=='male','SexN']=2
x = x.drop("Sex",1)

x['EmbarkedN'] = [1]*len(x)
x.loc[x.Embarked=='Q','EmbarkedN']=2
x.loc[x.Embarked=='S','EmbarkedN']=3
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
##x['SexSibSp']=x['SexN']*x['SibSp']
##x['SexAge']=x['SexN']*x['Age']
x = (x-sp.mean(x))/sp.std(x)


n_train = 500
x_train = x.iloc[:n_train,:]
y_train = y.iloc[:n_train]
x_test = x.iloc[n_train:,:]
y_test = y.iloc[n_train:]

##x_test = x_test[~pd.isnull(x_test.Age)]
##y_test = y_test[~pd.isnull(x_test.Age)]

sizes=[]
e_train = []
e_test = []
for i in range(2,len(x_train)):
    x_t = x_train.iloc[range(0,i),:]
    y_t = y_train.iloc[range(0,i)]
    clf = LogisticRegression(C=20)
    clf.fit(x_t,y_t)
    e_train.append(1-clf.score(x_t,y_t))
    e_test.append(1-clf.score(x_test,y_test))
    sizes.append(i)
##plt.plot(sizes, e_train)
##plt.plot(sizes, e_test)

y_predict_proba = clf.predict_proba(x_test)[:,1]
precision, recall, thresholds = precision_recall_curve(y_test, y_predict_proba)
max_fscore=0
maxth = 0
for i in range(0, len(precision)):
    fscore = precision[i]*recall[i]/(precision[i]+recall[i])
    if fscore > max_fscore:
        max_fscore=fscore
        maxth = thresholds[i]
y_predict = y_predict_proba>=maxth
print("F-score: %s, thres: %s"%(max_fscore*2,maxth))
fscores = precision_recall_fscore_support(y_test, y_predict, average='micro')

plt.plot(recall, precision)
print(fscores)
for i in range(0,len(x.columns)):
    print("%s - %s"%(x.columns[i], clf.coef_[0][i]))
##plt.show()
