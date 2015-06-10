import pandas as pd
import scipy as sp
from sklearn.ensemble import RandomForestClassifier
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import train_test_split,StratifiedShuffleSplit,StratifiedKFold, cross_val_score
import json as js
import pickle as pkl
from sklearn.preprocessing import Imputer
from sklearn.pipeline import Pipeline
from scipy  import stats
from sklearn import preprocessing

def prepareData(df):

    mode_embarked = df.Embarked.mode()[0]
    df.Embarked = df.Embarked.fillna(mode_embarked)

    df['Gender'] = df.Sex.map({'male':1, 'female':0}).astype(float)

    fare_means = pd.pivot_table(df, values='Fare',index=['Pclass'], aggfunc='median')
    df.Fare = df[['Fare','Pclass']].apply(lambda x: fare_means[x.Pclass] if pd.isnull(x.Fare) else x.Fare,1)

    titles=['Mr.', 'Master.','Mrs.','Miss.','Dr.','Rev.','Major.','Col.','Capt.','Dona.','Lady.','Sir.','Don.','Countess.','Jonkheer.','Mme.','Ms.','Mlle.']
##    common=['Mr', 'Master','Mrs','Miss','Dr','Mme','Ms','Mlle','Rev','Jonkheer','Col']
##    arist = ['Dona','Lady','Sir','Don','Countess']
##    military=['Major','Capt']

    mrlist = ['Mr','Don', 'Major', 'Capt', 'Jonkheer', 'Rev', 'Col','Sir','Dr']
    mrslist = ['Countess', 'Mme','Mrs','Dona','Lady']
    misslist = ['Mlle', 'Ms','Miss']
    
    def getTitle(x):
        for i in titles:
            if i in x:
                return i[:len(i)-1]

    def getTitleGroup(x):
        if x in mrlist:
            return 'Mr'
        else:
            if x in mrslist:
                return 'Mrs'
            else:
                if x in misslist:
                    return 'Miss'
                else:
                    if x=='Master':
                        return 'Master'
    def replace_titles(x):
        title=x['Title']
        if title in ['Mr','Don', 'Major', 'Capt', 'Jonkheer', 'Rev', 'Col', 'Sir']:
            return 'Mr'
        elif title in ['Master']:
            return 'Master'
        elif title in ['Countess', 'Mme','Mrs','Dona','Lady']:
            return 'Mrs'
        elif title in ['Mlle', 'Ms','Miss']:
            return 'Miss'
        elif title =='Dr':
            if x['Sex']=='Male':
                return 'Mr'
            else:
                return 'Mrs'
        elif title =='':
            if x['Sex']=='Male':
                return 'Master'
            else:
                return 'Miss'
        else:
            return title               
        
    df['Title'] = df.Name.map(lambda x: getTitle(x))

    df['Title'] = df.apply(lambda x:replace_titles(x),1)

    le = preprocessing.LabelEncoder()
 
    dum = pd.get_dummies(df.Title, prefix='Title')
    df = pd.concat([df,dum],1)

    age_means = pd.pivot_table(df, values='Age', index=['Title'], aggfunc='mean')
    df.Age = df[['Age','Title']].apply(lambda x: age_means[x.Title] if pd.isnull(x.Age) else x.Age,1)
    
    df['AgeCat'] = df.Age.map(lambda x: 'child' if x<=10 else 'adult' if x<=30 else 'senior' if x<=60 else 'aged')

    df['FamilySize'] = df.SibSp+df.Parch
    df['Family'] = df.SibSp*df.Parch
    df['FarePerPerson']=df.Fare/(df.FamilySize+1)

    df['PclassFare'] = df.Pclass*df.FarePerPerson
    df['PclassAge'] = df.Pclass*df.Age

    df['HighLow']=df['Pclass']
    df.loc[ (df.FarePerPerson<8) ,'HighLow'] = 'Low'
    df.loc[ (df.FarePerPerson>=8) ,'HighLow'] = 'High'

    le.fit(df['Sex'] )
    x_sex=le.transform(df['Sex'])
    df['Sex']=x_sex.astype(float)
    
    le.fit( df['Ticket'])
    x_Ticket=le.transform( df['Ticket'])
    df['Ticket']=x_Ticket.astype(float)

    le.fit(df['Title'])
    x_title=le.transform(df['Title'])
    df['Title'] =x_title.astype(float)

    le.fit(df['HighLow'])
    x_hl=le.transform(df['HighLow'])
    df['HighLow']=x_hl.astype(float)


    le.fit(df['AgeCat'])
    x_age=le.transform(df['AgeCat'])
    df['AgeCat'] =x_age.astype(float)

    le.fit(df['Embarked'])
    x_emb=le.transform(df['Embarked'])
    df['Embarked']=x_emb.astype(float)

    df = df.drop(['PassengerId','Name','Cabin'], 1)
    return df

config = js.loads(open('../SETTINGS.json').read())
train_path = "../"+config["TRAIN_DATA_PATH"]
model_path = "../"+config["MODEL_PATH"]
seed = int(config["SEED"])

dft = pd.read_csv(train_path+"/train.csv")
df = prepareData(dft)

x_train = df.drop("Survived",1)
y_train = df.Survived

x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.2, random_state=seed)

model = RandomForestClassifier(n_estimators=300, random_state=seed)

params = {'max_features':[0.5,1,'auto'], 'max_depth':[5, None]}

gs = GridSearchCV(model, params, cv=StratifiedShuffleSplit(y_train, n_iter=10, test_size=0.2, random_state=seed))
gs.fit(x_train, y_train)
print("Best Grid search results:")
print(gs.best_score_)
print(gs.best_params_)
c = [[df.columns[2+i], gs.best_estimator_.feature_importances_[i]] for i in range(0,len(df.columns)-2)]
d = sorted(c, key=lambda x: x[1],reverse=True)
for i in d:
	print("%s - %s"%(i[0],i[1]))

print("Train accuracy:")
scores = cross_val_score(gs.best_estimator_, x_train, y_train, cv=3, scoring="accuracy")
print(scores.mean(), scores)

print("Test accuracy:")
scores = cross_val_score(gs.best_estimator_, x_test, y_test, cv=3, scoring="accuracy")
print(scores.mean(), scores)

pkl.dump(gs.best_estimator_, open(model_path+"/model_rf.pkl",'wb'))
