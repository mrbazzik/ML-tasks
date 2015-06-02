import scipy as sp
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import neighbors
from sklearn.cross_validation import KFold


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

cv = KFold(n=len(x), n_folds=10, shuffle=True)
scores=[]
for train, test in cv:
    print(train.shape)
    print(test.shape)
    x_test, y_test=x.iloc[test,:], y.iloc[test]
    sizes=[]
    errors_train=[]
    errors_test=[]
    for i in range(2, len(train)):
        x_train, y_train = x.iloc[range(0,i),:], y.iloc[range(0,i)]
        knn = neighbors.KNeighborsClassifier(n_neighbors=2, weights='distance')
        knn.fit(x_train,y_train)
        error_train = 1 - knn.score(x_train,y_train)
        error_test = 1 - knn.score(x_test,y_test)
        sizes.append(i+1)
        errors_train.append(error_train)
        errors_test.append(error_test)
    plt.plot(sizes,errors_train)
    plt.plot(sizes,errors_test)
    plt.show()
    scores.append(1-errors_test[len(errors_test)-1])
print("Mean=%s, Std=%s"%(sp.mean(scores),sp.std(scores)))
##plt.plot(range(0,len(scores)),scores)
##plt.show()
