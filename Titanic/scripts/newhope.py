import pandas as pd
import scipy as sp
from sklearn.ensemble import RandomForestClassifier
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import KFold

def prepareData(df):
    df = df.drop(['Name','Ticket','Cabin'], 1)
    
##  treating nans
    age_mean = df.Age.mean()
    df.Age = df.Age.fillna(age_mean)

    mode_embarked = df.Embarked.mode()[0]
    df.Embarked = df.Embarked.fillna(mode_embarked)

    fare_means = pd.pivot_table(df, values='Fare',index=['Pclass'], aggfunc='mean')
    df.Fare = df[['Fare','Pclass']].apply(lambda x: fare_means[x.Pclass] if pd.isnull(x.Fare) else x.Fare,1)
    
    df['Gender'] = df.Sex.map({'male':1, 'female':0}).astype(int)
##    df['Port'] = df.Embarked.map({'C':1, 'S':2, 'Q':3}).astype(int)
    dum = pd.get_dummies(df.Embarked, prefix='Embarked')
    df = pd.concat([df,dum], 1)
    
    df = df.drop(['Sex','Embarked'], 1)
    cols = df.columns.tolist()
    if cols[1] == 'Survived':
        cols = cols[1:2]+cols[0:1]+cols[2:]
        df = df[cols]
    return df.values

df = pd.read_csv("c:\\Users\\VI\\Documents\\GitHub\\ML-tasks\\Titanic\\titanic_train.csv")
train_data = prepareData(df)

model = RandomForestClassifier(n_estimators=100)
params = {'max_features':[0.5,1], 'max_depth':[5, None]}
gs = GridSearchCV(model, params, cv=5, refit=True)
gs.fit(train_data[:,2:], train_data[:,0])
print('Score = %s'%(gs.best_score_))

##cv = KFold(n=len(train_data),n_folds=5)
##for tr, test in cv:
##    gs.fit(train_data[tr,2:],train_data[tr,0])
##    y_p = gs.predict(train_data[test,2:])
##    print('Score cv = %s'%(sp.mean(y_p==train_data[test,0])))

df_test  = pd.read_csv("c:\\Users\\VI\\Documents\\GitHub\\ML-tasks\\Titanic\\test.csv")
test_data = prepareData(df_test)

output = gs.predict(test_data[:,1:])

result = pd.DataFrame({'PassengerId': test_data[:,0].astype(int), 'Survived': output.astype(int)})
result.to_csv("c:\\Users\\VI\\Documents\\GitHub\\ML-tasks\\Titanic\\result.csv", index=False)
