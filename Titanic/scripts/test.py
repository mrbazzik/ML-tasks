import pandas as pd
import scipy as sp
from sklearn.ensemble import RandomForestClassifier
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import KFold
import pickle as pkl
import json as js

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
    return df

config = js.loads(open('../SETTINGS.json').read())
test_path = "../"+config["TEST_DATA_PATH"]
model_path = "../"+config["MODEL_PATH"]
submit_path = "../"+config["SUBMISSION_PATH"]

dft = pd.read_csv(test_path+"/test.csv")
df = prepareData(dft)
test_data = df.values

gs = pkl.load(open(model_path+"/model_rf.pkl",'rb'))
output = gs.predict(test_data[:,1:])

result = pd.DataFrame({'PassengerId': test_data[:,0].astype(int), 'Survived': output.astype(int)})
result.to_csv(submit_path+"/result_rf.csv", index=False)
