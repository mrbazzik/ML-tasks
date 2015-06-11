import pandas as pd
import scipy as sp
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import train_test_split,StratifiedShuffleSplit,StratifiedKFold, cross_val_score
import json as js
import pickle as pkl
from sklearn.preprocessing import Imputer
from sklearn.pipeline import Pipeline
from scipy  import stats
from sklearn import preprocessing
from sklearn.decomposition import PCA
import re

def reFind(str, x, na):
    ar = re.compile(str).findall(x)
    if len(ar) ==0:
        return na
    else:
        return ar[0]

def prepareData(df):
    
    print('cleaning features...')
    scaler = preprocessing.StandardScaler()

    ##-- Cabin --
    df['CabinNA'] = df.Cabin.isnull()
    df.Cabin = df.Cabin.fillna('U0')

    df['CabinFactor'] = pd.factorize(df.Cabin)[0]

    df['CabinLetter'] = df.Cabin.map(lambda x: reFind("([a-zA-z]+)", x, ''))
    dum = pd.get_dummies(df.CabinLetter,prefix='CabinLetter')
    df = pd.concat([df, dum], 1)
    df['CabinLetterFactor'] = df.CabinLetter.map({'A':0,
                                                  'B':1,
                                                  'C':2,
                                                  'D':3,
                                                  'E':4,
                                                  'F':5,
                                                  'G':6,
                                                  'T':7,
                                                  'U':8})
    df['CabinLetterBin'] = df.CabinLetter.map({'A':0,
                                                  'B':0,
                                                  'C':0,
                                                  'D':1,
                                                  'E':1,
                                                  'F':1,
                                                  'G':2,
                                                  'T':2,
                                                  'U':2})
    

    df['CabinNumber'] = df.Cabin.map(lambda x: reFind("([0-9]+)", x, '0')).astype(int)
    ##df['CabinNumberBin'] = pd.qcut(df.CabinNumber,3)
    ##df.CabinNumberBin = pd.factorize(df.CabinNumberBin)[0]

    ##-- Embarked --
    df.Embarked = df.Embarked.fillna(df.Embarked.mode()[0])

    df['EmbarkedFactor'] = pd.factorize(df.Embarked)[0]

    dum = pd.get_dummies(df.Embarked,prefix='Embarked')
    df = pd.concat([df, dum], 1)

    ##-- Fare --
    fare_med = pd.pivot_table(df,values='Fare', index='Pclass', aggfunc='median')
    df.Fare = df[['Fare','Pclass']].apply(lambda x: fare_med[x.Pclass] if pd.isnull(x.Fare) else x.Fare, 1)

##    df['FareScaled'] = scaler.fit_transform(df.Fare)

    df['FareBin'] = pd.qcut(df.Fare,5)
    df.FareBin = pd.factorize(df.FareBin)[0]

    ##-- Age --
    df['AgeNA'] = df.Age.isnull()
    
    knownAge = df.loc[df.Age.notnull(),:]
    unknownAge = df.loc[df.Age.isnull(),:]
    dropCol = ['Age','Survived','Cabin','Embarked','Name','Sex','Ticket','CabinLetter','CabinNumber']
    x_train = knownAge.drop(dropCol,1)
    y_train = knownAge.Age
    x_test = unknownAge.drop(dropCol,1)
    rfr = RandomForestRegressor(n_estimators=2000, n_jobs=-1)
    rfr.fit(x_train,y_train)
    df.loc[unknownAge.index,'Age'] = rfr.predict(x_test)

    
##    df['AgeScaled'] = scaler.fit_transform(df.Age)

    df['AgeBin'] = pd.qcut(df.Age,5)
    df.AgeBin = pd.factorize(df.AgeBin)[0]

    ##-- Parch --
    ##df['ParchBin']=pd.qcut(df.Parch, 3)
    ##df.ParchBin = pd.factorize(df.ParchBin)[0]

    ##-- Sex --
    df.Sex = pd.factorize(df.Sex)[0]

    ##-- SibSp --
    ##df['SibSpBin']=pd.qcut(df.SibSp, 2)
    ##df.SibSpBin = pd.factorize(df.SibSpBin)[0]
    

    ##-- Name --
    df['NameLen'] = df.Name.map(lambda x: len(re.split(' ',x)))

    df['Title'] = df.Name.map(lambda x: re.compile(", (.*?)\.").findall(x)[0])
    titles={'Master': ['Jonkheer'],
            'Miss': ['Ms', 'Mlle'],
            'Mrs': ['Mme'],
            'Sir': ['Capt','Don','Major','Col'],
            'Lady': ['Dona', 'the Countess']}
    df['TitleBin']=df.Title
    for i in titles:
        df.loc[df.Title.isin(titles[i]),'TitleBin'] = i
    dum = pd.get_dummies(df.TitleBin, prefix='Title')
    df = pd.concat([df,dum],1)
    df.TitleBin = pd.factorize(df.TitleBin)[0]

    df['Surname'] = df.Name.map(lambda x: re.split(' ',x)[0])
    

    ##-- Ticket --
    df['TicketPrefix'] = df.Ticket.map(lambda x: reFind("([a-zA-z\.\/]+)", x, ''))
    df['TicketPrefix'] = df.TicketPrefix.map(lambda x: re.sub("[\.\/]",'',x))
    dum = pd.get_dummies(df.TicketPrefix, prefix='TicketPrefix')
    df = pd.concat([df,dum],1)
    df.TicketPrefix = pd.factorize(df.TicketPrefix)[0]

    df['TicketNumber'] = df.Ticket.map(lambda x: reFind("([\d]+$)", x, '0')).astype(int)
    df['TicketNumberLen'] = df.TicketNumber.map(lambda x: len(str(x))).astype(int)
    df['TicketNumberFD'] = df.TicketNumber.map(lambda x: str(x)[0]).astype(int)

    df['TicketNumberBin'] = pd.qcut(df.TicketNumber, 10)
    df.TicketNumberBin = pd.factorize(df.TicketNumberBin)[0]


    ##-- new features --
    print('making new features...')
    df['FamilySize'] = df.Parch+df.SibSp
    
    df['FamilyId'] = df[['FamilySize','Surname']].apply(lambda x: x.Surname+str(x.FamilySize),1)
    df.FamilyId = pd.factorize(df.FamilyId)[0]

    columns = ['Pclass','Sex','SibSp','Parch','CabinFactor', 'CabinLetterFactor','CabinNumber','EmbarkedFactor','Fare','Age','NameLen','TitleBin','TicketPrefix','TicketNumber','TicketNumberLen','FamilySize','FamilyId']
    for i,iel in enumerate(columns):
        for j,jel in enumerate(columns):
            if i < j:
                df[iel+"+"+jel] = df[iel]+df[jel]
            if i <= j:
                df[iel+"*"+jel] = df[iel]*df[jel]
            if i != j:
                df[iel+"/"+jel] = df[iel]/df[jel].astype(float)
                df[iel+"-"+jel] = df[iel]-df[jel]
                

    ##-- check correlated features --
    print('deleting correlated features...')
    df_corr = df.corr(method = 'spearman')
    mask = sp.ones(df_corr.columns.size) - sp.eye(df_corr.columns.size)
    df_corr = df_corr*mask
    dels=[]
    for i in df_corr.columns:
        if i in dels:
            continue
        inds = df_corr.loc[abs(df_corr[i])>=0.98,i].index.tolist()
        for ind in inds:
            if ind not in dels:
                dels.append(ind)
    df = df.drop(dels,1)
    return df

config = js.loads(open('../SETTINGS.json').read())
train_path = "../"+config["TRAIN_DATA_PATH"]
test_path = "../"+config["TEST_DATA_PATH"]
model_path = "../"+config["MODEL_PATH"]
seed = int(config["SEED"])

df_train = pd.read_csv(train_path+"/train.csv")
df_test = pd.read_csv(test_path+"/test.csv")

df_full = pd.concat([df_train,df_test])
df_full = df_full.reset_index()
df_full = df_full.drop('index',1)


df_full = prepareData(df_full)
stop


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
