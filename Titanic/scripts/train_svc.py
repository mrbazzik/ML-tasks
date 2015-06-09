import pandas as pd
import scipy as sp
from sklearn.grid_search import GridSearchCV
import json as js
import pickle as pkl
from sklearn.preprocessing import Imputer
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

def prepareData(df):
    df = df.drop(['Name','Ticket','Cabin'], 1)
    
##  treating nans
    age_mean = df.Age.mean()
##    df.Age = df.Age.fillna(age_mean)
    df.Age = df.Age.fillna(-1)

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
    cols = cols[1:2]+cols[0:1]+cols[2:]
    df = df[cols]
    return df

config = js.loads(open('../SETTINGS.json').read())
train_path = "../"+config["TRAIN_DATA_PATH"]
model_path = "../"+config["MODEL_PATH"]

df = pd.read_csv(train_path+"/train.csv")
df = prepareData(df)
train_data = df.values

imputer = Imputer(strategy='mean', missing_values=-1)

model = SVC(kernel='linear')

pipeline = Pipeline([('imp',imputer),('clf',model)])
params = {'clf__C':[1,10],
          'clf__gamma': [0.1,1]
##          'clf__kernel': ['linear'],
##          'imp__strategy': ['mean','median'],
          }

gs = GridSearchCV(pipeline, params, cv=5, refit=False)
gs.fit(train_data[:,2:], train_data[:,0])
print(gs.best_score_)
print(gs.best_params_)

df.Age = df.Age.map(lambda x: df.Age.mean() if x==-1 else x)
##print('Score = %s'%(gs.best_score_))
train_data = df.values
model = SVC(kernel='linear',C=gs.best_params_['clf__C'],gamma=gs.best_params_['clf__gamma'])
model = model.fit(train_data[:,2:], train_data[:,0])

pkl.dump(model, open(model_path+"/model_svc.pkl",'wb'))
##cv = KFold(n=len(train_data),n_folds=5)
##for tr, test in cv:
##    gs.fit(train_data[tr,2:],train_data[tr,0])
##    y_p = gs.predict(train_data[test,2:])
##    print('Score cv = %s'%(sp.mean(y_p==train_data[test,0])))


