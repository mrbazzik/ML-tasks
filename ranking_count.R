n<-500000

#with spouse
ageS <- c(0,5,15,25,35,45,50,55,60,65,70,75,80,85,90,95,100)
f<-ageS
prob<-dnorm(f, mean=mean(f), sd=sd(f))*(1/sum(dnorm(f, mean=mean(f), sd=sd(f))))
ageS<-sample(f,n, replace=TRUE,prob=prob)


eduS <- c(0,28,84,91,112,119,126,140)
f<-eduS
prob<-dnorm(f, mean=mean(f), sd=sd(f))*(1/sum(dnorm(f, mean=mean(f), sd=sd(f))))
eduS<-sample(f,n, replace=TRUE,prob=prob)

folS1 <- c(0,6,8,16,22,29,32)
f<-folS1
prob<-dnorm(f, mean=mean(f), sd=sd(f))*(1/sum(dnorm(f, mean=mean(f), sd=sd(f))))
folS1<-sample(f,n, replace=TRUE,prob=prob)

folS2 <- c(0,6,8,16,22,29,32)
f<-folS2
prob<-dnorm(f, mean=mean(f), sd=sd(f))*(1/sum(dnorm(f, mean=mean(f), sd=sd(f))))
folS2<-sample(f,n, replace=TRUE,prob=prob)

folS3 <- c(0,6,8,16,22,29,32)
f<-folS3
prob<-dnorm(f, mean=mean(f), sd=sd(f))*(1/sum(dnorm(f, mean=mean(f), sd=sd(f))))
folS3<-sample(f,n, replace=TRUE,prob=prob)

folS4 <- c(0,6,8,16,22,29,32)
f<-folS4
prob<-dnorm(f, mean=mean(f), sd=sd(f))*(1/sum(dnorm(f, mean=mean(f), sd=sd(f))))
folS4<-sample(f,n, replace=TRUE,prob=prob)

solS1 <- c(0,1,3,6)
f<-solS1
prob<-dnorm(f, mean=0, sd=sd(f))*(1/sum(dnorm(f, mean=0, sd=sd(f))))
solS1<-sample(f,n, replace=TRUE,prob=prob)

solS2 <- c(0,1,3,6)
f<-solS2
prob<-dnorm(f, mean=0, sd=sd(f))*(1/sum(dnorm(f, mean=0, sd=sd(f))))
solS2<-sample(f,n, replace=TRUE,prob=prob)

solS3 <- c(0,1,3,6)
f<-solS3
prob<-dnorm(f, mean=0, sd=sd(f))*(1/sum(dnorm(f, mean=0, sd=sd(f))))
solS3<-sample(f,n, replace=TRUE,prob=prob)

solS4 <- c(0,1,3,6)
f<-solS4
prob<-dnorm(f, mean=0, sd=sd(f))*(1/sum(dnorm(f, mean=0, sd=sd(f))))
solS4<-sample(f,n, replace=TRUE,prob=prob)

cexpS <- c(0,35,46,56,63,70)
f<-cexpS
prob<-c(75,10,5,5,3,2)
cexpS<-sample(f,n, replace=TRUE,prob=prob)

sedu <- c(0,2,6,7,8,9,10)
f<-sedu
prob<-c(65,2,3,5,10,10,5)
sedu<-sample(f,n, replace=TRUE,prob=prob)


sfol1 <- c(0,1,3,5)
f<-sfol1
prob<-c(70,2,10,18)
sfol1<-sample(f,n, replace=TRUE,prob=prob)

sfol2 <- c(0,1,3,5)
f<-sfol2
prob<-c(70,2,10,18)
sfol2<-sample(f,n, replace=TRUE,prob=prob)

sfol3 <- c(0,1,3,5)
f<-sfol3
prob<-c(70,2,10,18)
sfol3<-sample(f,n, replace=TRUE,prob=prob)

sfol4 <- c(0,1,3,5)
f<-sfol4
prob<-c(70,2,10,18)
sfol4<-sample(f,n, replace=TRUE,prob=prob)

scexp <- c(0,5,7,8,9,10)
f<-scexp
prob<-c(70,10,10,5,3,2)
scexp<-sample(f,n, replace=TRUE,prob=prob)

#without spouse
age <- c(0,6,17,28,39,50,55,61,66,72,77,83,88,94,99,105,110)
f<-age
prob<-dnorm(f, mean=mean(f), sd=sd(f))*(1/sum(dnorm(f, mean=mean(f), sd=sd(f))))
age<-sample(f,n, replace=TRUE,prob=prob)

edu <- c(0,30,90,98,120,128,135,150)
f<-edu
prob<-dnorm(f, mean=mean(f), sd=sd(f))*(1/sum(dnorm(f, mean=mean(f), sd=sd(f))))
edu<-sample(f,n, replace=TRUE,prob=prob)

fol1 <- c(0,6,9,17,23,31,34)
f<-fol1
prob<-dnorm(f, mean=mean(f), sd=sd(f))*(1/sum(dnorm(f, mean=mean(f), sd=sd(f))))
fol1<-sample(f,n, replace=TRUE,prob=prob)

fol2 <- c(0,6,9,17,23,31,34)
f<-fol2
prob<-dnorm(f, mean=mean(f), sd=sd(f))*(1/sum(dnorm(f, mean=mean(f), sd=sd(f))))
fol2<-sample(f,n, replace=TRUE,prob=prob)

fol3 <- c(0,6,9,17,23,31,34)
f<-fol3
prob<-dnorm(f, mean=mean(f), sd=sd(f))*(1/sum(dnorm(f, mean=mean(f), sd=sd(f))))
fol3<-sample(f,n, replace=TRUE,prob=prob)

fol4 <- c(0,6,9,17,23,31,34)
f<-fol4
prob<-dnorm(f, mean=mean(f), sd=sd(f))*(1/sum(dnorm(f, mean=mean(f), sd=sd(f))))
fol4<-sample(f,n, replace=TRUE,prob=prob)

sol1 <- c(0,1,3,6)
f<-sol1
prob<-dnorm(f, mean=0, sd=sd(f))*(1/sum(dnorm(f, mean=0, sd=sd(f))))
sol1<-sample(f,n, replace=TRUE,prob=prob)

sol2 <- c(0,1,3,6)
f<-sol2
prob<-dnorm(f, mean=0, sd=sd(f))*(1/sum(dnorm(f, mean=0, sd=sd(f))))
sol2<-sample(f,n, replace=TRUE,prob=prob)

sol3 <- c(0,1,3,6)
f<-sol3
prob<-dnorm(f, mean=0, sd=sd(f))*(1/sum(dnorm(f, mean=0, sd=sd(f))))
sol3<-sample(f,n, replace=TRUE,prob=prob)

sol4 <- c(0,1,3,6)
f<-sol4
prob<-dnorm(f, mean=0, sd=sd(f))*(1/sum(dnorm(f, mean=0, sd=sd(f))))
sol4<-sample(f,n, replace=TRUE,prob=prob)

cexp <- c(0,40,53,64,72,80)
f<-cexp
prob<-c(75,10,5,5,3,2)
cexp<-sample(f,n, replace=TRUE,prob=prob)


fexp <- c(0,1,2)
f<-fexp
prob<-c(10,20,70)
fexp<-sample(f,n, replace=TRUE,prob=prob)

tradecert <- c(0,1)
f<-tradecert
prob<-c(90,10)
tradecert<-sample(f,n, replace=TRUE,prob=prob)


emp <- c(0,600)
f<-emp
prob<-c(90,10)
emp<-sample(f,n, replace=TRUE,prob=prob)

dfs <- data.frame(Age=ageS,Edu=eduS,FolS1=folS1,FolS2=folS2,FolS3=folS3,FolS4=folS4,SolS1=solS1,SolS2=solS2,SolS3=solS3,SolS4=solS4,Cexp=cexpS,Sedu=sedu,Sfol1=sfol1,Sfol2=sfol2,Sfol3=sfol3,SfolS4=sfol4,Scexp=scexp,Fexp=fexp, Tradecert=tradecert, Emp=emp)
df<- data.frame(Age=age,Edu=edu,FolS1=fol1,FolS2=fol2,FolS3=fol3,FolS4=fol4,SolS1=sol1,SolS2=sol2,SolS3=sol3,SolS4=sol4,Cexp=cexp,Sedu=0,Sfol1=0,Sfol2=0,Sfol3=0,SfolS4=0,Scexp=0,Fexp=fexp, Tradecert=tradecert, Emp=emp)

df$LangEdu<-0
df$CanEdu<-0
df$ForLang<-0
df$ForCan<-0
df$TradeLang<-0

dfs$LangEdu<-0
dfs$CanEdu<-0
dfs$ForLang<-0
dfs$ForCan<-0
dfs$TradeLang<-0

 
addPointsLangEduS <- function(x){
    prod <- x['FolS1']*x['FolS2']*x['FolS3']*x['FolS4']
    if(prod>=29^4 & x['Edu']==84) 25
    else if (prod >= 29^4 & x['Edu']>=91) 50
    else if (prod>=16^4 & prod < 29^4 & x['Edu']==84) 13
    else if (prod>=16^4 & prod < 29^4 & x['Edu']>=91) 25
    else 0
}   
addPointsCanEduS <- function(x){
    if (x['Cexp']==35 & x['Edu']==84) 13
    else if (x['Cexp']>=46 & x['Edu']==84) 25
    else if (x['Cexp']==35 & x['Edu']>=91) 25
    else if (x['Cexp']>=46 & x['Edu']>=91) 50
    else 0
}
    
addPointsForLangS <- function(x){
    prod <- x['FolS1']*x['FolS2']*x['FolS3']*x['FolS4']
    if(prod>=29^4 & x['Fexp']==1) 25
    else if (prod >= 29^4 & x['Fexp']==2) 50
    else if (prod>=16^4 & prod < 29^4 & x['Fexp']==1) 13
    else if (prod>=16^4 & prod < 29^4 & x['Fexp']==2) 25
    else 0
}

addPointsForCanS <- function(x){
    if (x['Cexp']==35 & x['Fexp']==1) 13
    else if (x['Cexp']>=46 & x['Fexp']==1) 25
    else if (x['Cexp']==35 & x['Fexp']==2) 25
    else if (x['Cexp']>=46 & x['Fexp']==2) 50
    else 0
}
addPointsTradeLangS <- function(x){
    prod <- x['FolS1']*x['FolS2']*x['FolS3']*x['FolS4']
    if(prod>=16^4 & x['Tradecert']==1) 50
    else if (prod>=6^4 & prod < 16^4 & x['Tradecert']==1) 25
    else 0
    
}
addPointsLangEdu <- function(x){
    prod <- x['FolS1']*x['FolS2']*x['FolS3']*x['FolS4']
    if(prod>=31^4 & x['Edu']==90) 25
    else if (prod >= 31^4 & x['Edu']>=98) 50
    else if (prod>=17^4 & prod < 31^4 & x['Edu']==90) 13
    else if (prod>=17^4 & prod < 31^4 & x['Edu']>=98) 25
    else 0
}

addPointsCanEdu <- function(x){
    if (x['Cexp']==40 & x['Edu']==90) 13
    else if (x['Cexp']>=53 & x['Edu']==90) 25
    else if (x['Cexp']==40 & x['Edu']>=98) 25
    else if (x['Cexp']>=53 & x['Edu']>=98) 50
    else 0
}

addPointsForLang <- function(x){
    prod <- x['FolS1']*x['FolS2']*x['FolS3']*x['FolS4']
    if(prod>=31^4 & x['Fexp']==1) 25
    else if (prod >= 31^4 & x['Fexp']==2) 50
    else if (prod>=17^4 & prod < 31^4 & x['Fexp']==1) 13
    else if (prod>=17^4 & prod < 31^4 & x['Fexp']==2) 25
    else 0
}

addPointsForCan <- function(x){
    if (x['Cexp']==40 & x['Fexp']==1) 13
    else if (x['Cexp']>=53 & x['Fexp']==1) 25
    else if (x['Cexp']==40 & x['Fexp']==2) 25
    else if (x['Cexp']>=53 & x['Fexp']==2) 50
    else 0
}

addPointsTradeLang <- function(x){
    prod <- x['FolS1']*x['FolS2']*x['FolS3']*x['FolS4']
    if(prod>=17^4 & x['Tradecert']==1) 50
    else if (prod>=6^4 & prod < 17^4 & x['Tradecert']==1) 25
    else 0
}

df$LangEdu<- apply(df,1, addPointsLangEdu)
df$CanEdu<- apply(df,1, addPointsCanEdu)
df$ForLang<- apply(df,1, addPointsForLang)
df$ForCan<- apply(df,1, addPointsForCan)
df$TradeLang<- apply(df,1, addPointsTradeLang)

dfs$LangEdu<- apply(df,1, addPointsLangEduS)
dfs$CanEdu<- apply(df,1, addPointsCanEduS)
dfs$ForLang<- apply(df,1, addPointsForLangS)
dfs$ForCan<- apply(df,1, addPointsForCanS)
dfs$TradeLang<- apply(df,1, addPointsTradeLangS)

dffull<-rbind(df,dfs)

dffull<-dffull[,!(names(dffull) %in% c('Tradecert','Fexp'))]

dffull$Points<- apply(dffull,1,sum)

dffull<-dffull[dffull$Points>=300,]
hist(dffull$Points,breaks=seq(300,1250,10),main='Примерный состав пула из 1000000 человек', xlab='Балл', ylab='Количество человек')

