import pandas as pd
import scipy as sp
from sklearn.ensemble import RandomForestClassifier
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import KFold
import json as js
import pickle as pkl
from sklearn.preprocessing import Imputer
from sklearn.pipeline import Pipeline

def prepareData(df):
##    df = df.drop(['Name','Ticket','Cabin'], 1)
    
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

    titles=['Mr.', 'Master.','Mrs.','Miss.','Dr.','Rev.','Major.','Col.','Capt.','Dona.','Lady.','Sir.','Don.','Countess.','Jonkheer.','Mme.','Ms.','Mlle.']
    common=['Mr', 'Master','Mrs','Miss','Dr','Mme','Ms','Mlle']
    arist = ['Dona','Lady','Sir','Don','Countess','Jonkheer','Rev']
    military=['Major','Col','Capt']
    
    def getTitle(x):
        for i in titles:
            if i in x:
                return i[:len(i)-1]

    df['Title'] = df.Name.map(lambda x: getTitle(x))
    
##    for i in titles:
##        title = i[:len(i)-1]
##        df['Title_'+title] = (df.Title==title).astype(int)
        
    df['TitleGroup_Common']=df.Title.map(lambda x: 1 if x in common else 0)
    df['TitleGroup_Arist']=df.Title.map(lambda x: 1 if x in arist else 0)
    df['TitleGroup_Military']=df.Title.map(lambda x: 1 if x in military else 0)
                
 
##    dum = pd.get_dummies(df.Title, prefix='Title')
##    df = pd.concat([df,dum],1)
##    dum = pd.get_dummies(df.TitleGroup, prefix='TitleGroup')
##    df = pd.concat([df,dum],1)
   
    
    df = df.drop(['Name','Ticket','Cabin','Sex','Embarked','Title'], 1)
    cols = df.columns.tolist()
    cols = cols[1:2]+cols[0:1]+cols[2:]
    df = df[cols]
    return df

config = js.loads(open('../SETTINGS.json').read())
train_path = "../"+config["TRAIN_DATA_PATH"]
model_path = "../"+config["MODEL_PATH"]

dft = pd.read_csv(train_path+"/train.csv")
df = prepareData(dft)

train_data = df.values

imputer = Imputer(missing_values=-1)

model = RandomForestClassifier(n_estimators=300)

pipeline = Pipeline([('imp',imputer),('clf',model)])
params = {'clf__max_features':[0.5,1], 'clf__max_depth':[5, None], 'imp__strategy': ['mean','median']}

gs = GridSearchCV(pipeline, params, cv=10)
gs.fit(train_data[:,2:], train_data[:,0])
print(gs.best_score_)
print(gs.best_params_)

df.Age = df.Age.map(lambda x: df.Age.median() if x==-1 else x)
##print('Score = %s'%(gs.best_score_))
train_data = df.values
model = RandomForestClassifier(n_estimators=300,max_features=gs.best_params_['clf__max_features'],max_depth=gs.best_params_['clf__max_depth'])
model = model.fit(train_data[:,2:], train_data[:,0])

pkl.dump(model, open(model_path+"/model_rf.pkl",'wb'))
##cv = KFold(n=len(train_data),n_folds=5)
##for tr, test in cv:
##    gs.fit(train_data[tr,2:],train_data[tr,0])
##    y_p = gs.predict(train_data[test,2:])
##    print('Score cv = %s'%(sp.mean(y_p==train_data[test,0])))


