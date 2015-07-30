setwd("c:\\Users\\Basov_il\\Documents\\GitHub\\ML-tasks\\Titanic\\train_data\\")
df<-read.csv("train.csv")
str(df)
library(dplyr)
by_emb<-group_by(df, Embarked)
summarise(by_emb, Mean=mean(Survived), C=n())

#Q and S - 1, C - 2

by_class<-group_by(df, Pclass)
summarise(by_class, Mean=mean(Survived), C=n())

#3 - 1, 2 - 2, 1 - 3

by_sex<-group_by(df, Sex)
summarise(by_sex, Mean=mean(Survived), C=n())

#male - 1, female - 4

by_age<-group_by(df, Age)
sum_age<-summarise(by_age, Mean=mean(Survived), C=n())
sum_age<-filter(sum_age, C>=2)
barplot(sum_age$Mean, names.arg=sum_age$Age)

by_nage<-group_by(df, is.na(Age))
sum_nage<-summarise(by_nage, Mean=mean(Survived), C=n())

#?16<age<48 - 1, age<16 - 3, age > 48 - 2, noage - 0
#is.na(age) - 1, !isna - 2

by_sibsp<-group_by(df, SibSp)
sum_sibsp<-summarise(by_sibsp, Mean=mean(Survived), C=n())
sum_sibsp

#SibSp>=3 - 1, SibSp=0 - 2, SibSp=1|2 - 3

by_parch<-group_by(df, Parch)
sum_parch<-summarise(by_parch, Mean=mean(Survived), C=n())
sum_parch
#Parch>=4 - 1, SibSp=0 - 2, SibSp=1|2|3 - 3

by_fare<-group_by(df, Fare)
sum_fare<-summarise(by_fare, Mean=mean(Survived), C=n())
sum_fare<-filter(sum_fare, C>=1)
barplot(sum_fare$Mean, names.arg=sum_fare$Fare)

#?Fare

df$Title <- apply(df, 1, function(x) gsub(".*, (.*?\\.) .*","\\1",x['Name']))
by_title<-group_by(df, Title)
sum_title<-summarise(by_title, Mean=mean(Survived), C=n())
sum_title<-filter(sum_title, C>=3)
barplot(sum_title$Mean, names.arg=sum_title$Title)

#Title:
# Mr. - 1, Col.,Dr.,Major.,Master. - 3, Miss. - 4, Mrs.,Lady.,Mlle.,Mme.,Ms.,Sir., the Countess. - 5, others - 0


handleTicketPre<-function(x){
    str<-gsub("([a-zA-Z\\.\\/]+).*","\\1",x['Ticket'])
    if(str==x['Ticket']) ""
    else gsub("[\\.\\/\\d]","",str)
}

df$TicketPre <- apply(df, 1, handleTicketPre)
by_ticketpre<-df %>% group_by(TicketPre) %>% summarise(Mean=mean(Survived), C=n()) %>% filter(C>=1) %>% arrange(Mean)
#sum_title<-filter(sum_title, C>=3)
barplot(by_ticketpre$Mean, names.arg=by_ticketpre$TicketPre,las=2)

#For TicketPre see graph

df$TicketNumber <- apply(df, 1, function(x) as.numeric(gsub(".*?(\\d*)$","\\1", x['Ticket'])))
by_ticnumber <- df %>% group_by(TicketNumber) %>% summarise(Mean=mean(Survived), C=n()) %>% arrange(TicketNumber) %>% filter(C>=1)
barplot(by_ticnumber$Mean, names.arg=by_ticnumber$TicketNumber,las=2)

#TicketNumber - NO

by_cabin <- df %>% group_by(Cabin=="") %>% summarise(Mean=mean(Survived), C=n())

#Cabin=="" - 1, else - 2

df$CabLetter <- apply(df, 1, function(x) gsub("([a-zA-Z]+).*","\\1", x['Cabin']))
by_cablet <- df %>% group_by(CabLetter) %>% summarise(Mean=mean(Survived), C=n()) %>% arrange(Mean) %>% filter(C>=1)
barplot(by_cablet$Mean, names.arg=by_cablet$CabLetter,las=2)

#CabinLetter - see graph!

df$CabNumber <- apply(df, 1, function(x) gsub(".*?(\\d*)$","\\1", x['Cabin']))
 by_cabnumber <- df %>% group_by(CabNumber) %>% summarise(Mean=mean(Survived), C=n()) %>% arrange(Mean) %>% filter(C>=5)
barplot(by_cabnumber$Mean, names.arg=by_cabnumber$CabNumber,las=2)

#CabNumber - NO

##Handling

df$Emb <- apply(df,1, function(x) ifelse(x['Embarked']=='C',2, 1))
df$Class <- apply(df,1,function(x) ifelse(x['Pclass']==1, 3, ifelse(x['Pclass']==2,2,1)))
df$SexN <- apply(df,1, function(x) ifelse(x['Sex']=='male',1, 4))
df$Nage <- apply(df,1, function(x) ifelse(is.na(x['Age']),1, 2))
df$AgeBin <- apply(df,1, function(x) ifelse(is.na(x['Age']),0, 
                                            ifelse(x['Age']>48,2,
                                                   ifelse(x['Age']>16,1,3))))
df$Sib <- apply(df,1, function(x) ifelse(x['SibSp']>=3,1, 
                                            ifelse(x['SibSp']>=1,3,2)))

df$Par <- apply(df,1, function(x) ifelse(x['Parch']>=4,1, 
                                         ifelse(x['Parch']>=1,3,2)))

convertTitle<-function(x){
    # Mr. - 1, Col.,Dr.,Major.,Master. - 3, Miss. - 4, Mrs.,Lady.,Mlle.,Mme.,Ms.,Sir., the Countess. - 5, others - 0
    title<-x['Title']
    if(title=="Mr.") 1
    else if (title %in% c("Col.","Dr.","Major.","Master.")) 3
    else if (title == "Miss.") 4
    else if (title %in% c("Mrs.","Lady.","Mlle.","Mme.","Ms.","Sir.","the Countess.")) 5
    else 0
}

df$TitleC<-apply(df,1,convertTitle)

convertTicketPre<-function(x){
    
    tpre<-x['TicketPre']
    if(tpre=="A") 1
    else if(tpre=="WC") 2
    else if(tpre=="SOTONOQ") 3
    else if(tpre=="SOC") 4
    else if (tpre %in% c("WEP","CA")) 8
    else if (tpre %in% c("","C")) 9
    else if (tpre %in% c("SCPARIS","STONO","SCParis")) 10
    else if (tpre %in% c("PPP")) 11
    else if (tpre %in% c("PC","PP","SCAH")) 15
    else if (tpre %in% c("FCC")) 18
    else if (tpre %in% c("SC","SWPP")) 22
    else 0
}

df$TicketP<-apply(df,1,convertTicketPre)


df$NCabin <- apply(df,1, function(x) ifelse(x['Cabin']=='',1, 2))

convertCabLetter<-function(x){
    
    clet<-x['CabLetter']
    if(clet=="") 1
    else if (clet %in% c("A","G")) 2
    else if (clet %in% c("C","F")) 3
    else if (clet %in% c("B","E","D")) 4
    else 0
}
df$CabinL<-apply(df,1,convertCabLetter)
ids<-df$PassengerId
df<-select(df,-c(PassengerId,Pclass,Name,Sex,Age,SibSp,Parch,Ticket,Cabin,Embarked,Title,TicketPre,CabLetter,CabNumber,TicketNumber))


mylogit<-glm(Survived~.,data=df,family="binomial")
summary(mylogit)

## HCLUST

setwd("c:\\Users\\Basov_il\\Documents\\GitHub\\ML-tasks\\Titanic\\test_data\\")
df_test<-read.csv("test.csv")

df<-select(df,-Survived)
df<-rbind(df,df_test)
df<-select(df,-Cabin)

surv<-df$Survived
df<-select(df,-c(Survived,Cluster))
d<-dist(df)
hc<-hclust(d)
plot(hc)
tr<-cutree(hc, k=7)
plot(tr)

df$Survived<-surv
df$Cluster<-tr

gr<- df %>% group_by(Cluster) %>% summarise(Mean=mean(Survived), C=n())
gr
