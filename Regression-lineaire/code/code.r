######PROJET REGRESSION LINEAIRE#####
A=mtcars
F1=as.factor(A[,8])
A[,8]=F1
set.seed(17)
set.seed(17*floor(100*runif(1,0,3)))
set1=sample(1:32,1)
B=A[-set1,]
Y=B[,1]
u=1:11
v=u[-c(1,8,9)]
set2=c(8,sample(v,6,replace=FALSE))
X=B[,set2]

######PARTIE 1 - REGRESSION LINEAIRE SIMPLE######
deter_r=function(){
  n=length(X)
  R=rep(c(0),times=n)
  for (i in 2:n){ #on commence à la 2e colonne car la première correspond à des valeurs qualitatives (or on ne peut faire des regressions linéaires que sur les variables quantitatives)
    x=X[,i]
    R[i]=cor(x,Y) #coefficient linéaire de Pearson
  }
  R=R^2
  R
}

an_chap=(sum(X[,4]*Y) - 31*mean(X[,4])*mean(Y)) / sum((X[,4] - mean(X[,4]))^2)
bn_chap=mean(Y) - an_chap*mean(X[,4])

#On remarque que x=X[,4] correspond au meilleur R
#Dans la suite, on va donc considérer principalement X[,4]


######Graphes######
#Graphes pour x=X[,4]
an_chap=(sum(X[,4]*Y) - 31*mean(X[,4])*mean(Y)) / sum((X[,4] - mean(X[,4]))^2)
bn_chap=mean(Y) - an_chap*mean(X[,4])

plot(X[,4], Y, main="Régression linéaire simple", xlab="Variable explicative : wt ", ylab="Variable réponse", pch=19, col="blue")
abline(a=bn_chap, b=an_chap, col="red", lwd=2)
legend("topright", legend=c("Données", "Droite de régression"),col=c("blue", "red"), pch=c(19, NA), lty=c(NA, 1), lwd=c(NA, 2))

#Graphe pour x=X[,3]
an_chap=(sum(X[,3]*Y) - 31*mean(X[,3])*mean(Y)) / sum((X[,3] - mean(X[,3]))^2)
bn_chap=mean(Y) - an_chap*mean(X[,3])

plot(X[,3], Y, main="Régression linéaire simple", xlab="Variable explicative : wt ", ylab="Variable réponse", pch=19, col="blue")
abline(a=bn_chap, b=an_chap, col="red", lwd=2)
legend("topright", legend=c("Données", "Droite de régression"),col=c("blue", "red"), pch=c(19, NA), lty=c(NA, 1), lwd=c(NA, 2))

#Graphe pour x=X[,6]
an_chap=(sum(X[,6]*Y) - 31*mean(X[,6])*mean(Y)) / sum((X[,6] - mean(X[,6]))^2)
bn_chap=mean(Y) - an_chap*mean(X[,6])

plot(X[,6], Y, main="Régression linéaire simple", xlab="Variable explicative : wt ", ylab="Variable réponse", pch=19, col="blue")
abline(a=bn_chap, b=an_chap, col="red", lwd=2)
legend("topright", legend=c("Données", "Droite de régression"),col=c("blue", "red"), pch=c(19, NA), lty=c(NA, 1), lwd=c(NA, 2))



#On repose :
an_chap=(sum(X[,4]*Y) - 31*mean(X[,4])*mean(Y)) / sum((X[,4] - mean(X[,4]))^2)
bn_chap=mean(Y) - an_chap*mean(X[,4])




######ETUDE DES RESIDUS######
#Residus standardisés
#pour X[,4] = wt
m = lm(Y ~ X[,4])
rst1 = rstandard(m)
qqnorm(rst1, main="Q-Q plot des résidus standardisés") #pour regarder si suit une loi normale standart
qqline(rst1, col="red", lwd=2)

#pour X[,3] = qsec
m2 = lm(Y ~ X[,3])
rst2 = rstandard(m2)
qqnorm(rst2, main="Q-Q plot des résidus standardisés") #pour regarder si suit une loi normale standart
qqline(rst1, col="red", lwd=2)

#pour X[,6] = drat
m3 = lm(Y ~ X[,6])
rst3 = rstandard(m3)
qqnorm(rst3, main="Q-Q plot des résidus standardisés") #pour regarder si suit une loi normale standart
qqline(rst1, col="red", lwd=2)


#Résidus studentisés
m=lm(formula = Y~X[,4])

rst4=rstudent(m)
srst4=sort(rst4)
nt4=length(rst4)
yst4=1/nt4*(1:nt4)

#loi de Student
xt=seq(srst4[1],srst4[nt4],0.01)
yt=pt(xt,length(Y)-3)
plot(srst4,yst4, main="Résidus studentisés",type='s',col='blue',xlim=c(srst4[1],srst4[nt4]),ylim=c(0,1)) ##modifier le titre
lines(xt,yt,col='red')

ks.test(rst4,"pt", length(Y)-3)

#Remarque :
#la p-valeur=0.8667 > 5% donc on peut donc bien procéder aux tests sur a.
#on a fait une double vérification : validité de l'hypothèse de gaussianité du bruit pour les résidus standardisés et pour les résidus studentisés


#Bonus : Les résidus pourraient-ils suivrent plutôt une loi Uniforme plutôt ?
xt=seq(min(srst4), max(srst4), length.out=1000)
yt_unif=punif(xt,min=min(rst4), max(rst4))
plot(srst4, yst4, type="s", col="blue", ylim=c(0,1), main="Résidus studentisés vs Loi Uniforme", xlab="Résidus", ylab="Fonction de répartition")
lines(xt,yt_unif,col="green", lwd=2)
#legend("topleft", legend=c("CDF empirique", "CDF Uniforme"))

ks.test(rst4,"punif", min=min(rst4), max=max(rst4))

#Remarque :
#Ici, la p-valeur du test Uniforme est 0.00743 < 5%. Or la p-valeur du test de Student est > 5%.
#Nous avons donc montrer statistiquement que les résidus suivaient mieux une loi de Student qu'une loi Uniforme.


#On a :
#H0 : a=0
#H1 : a!=0
#D'ou la formule suivante pour ta (on a remplacé an_chap - a par an_chap car a=0)
n=length(X[,4])
Y_chap=an_chap * X[,4] + bn_chap
on2_chap= 1/(n-2) * sum((Y-Y_chap)^2)
on_chap=sqrt(on2_chap)

Ta = an_chap / (on_chap * sqrt( 1 / (sum((X[,4] - mean(X[,4])) ^2 )) ))
Ta=abs(Ta)
PT=pt(Ta, df=n-2)
p_val=1-PT

#on obtient p_val = 1.07838e-10 < 0.05 (5%)
#p_val < alpha donc on décide H1, à savoir : a!=0

######PREDICTION######
#Pour le meilleur R : X[,4]
df = data.frame(x = X[, 4], y = Y)
m = lm(y ~ x, data = df)
plot(df$x, df$y, main = "Régression linéaire avec intervalle de confiance à 95%", xlab = "Variable explicative", ylab = "Variable réponse")
abline(m, col = "red", lwd = 1)
x_seq = seq(min(df$x), max(df$x), length.out = 100)
x_new = data.frame(x = x_seq) 
pred = predict(m, newdata = x_new, interval = "confidence", level = 0.95)
lines(x_seq, pred[, "lwr"], col = "blue", lwd = 1)
lines(x_seq, pred[, "upr"], col = "blue", lwd = 1)
pred = predict(m, newdata = x_new, interval = "prediction", level = 0.95)
lines(x_seq, pred[, "lwr"], col = "green", lwd = 1)
lines(x_seq, pred[, "upr"], col = "green", lwd = 1)

#Pour le moins bon R : X[,3]
df = data.frame(x = X[, 3], y = Y)
m = lm(y ~ x, data = df)
plot(df$x, df$y, main = "Régression linéaire avec intervalle de confiance à 95%", xlab = "Variable explicative", ylab = "Variable réponse")
abline(m, col = "red", lwd = 1)
x_seq = seq(min(df$x), max(df$x), length.out = 100)
x_new = data.frame(x = x_seq) 
pred = predict(m, newdata = x_new, interval = "confidence", level = 0.95)
lines(x_seq, pred[, "lwr"], col = "blue", lwd = 1)
lines(x_seq, pred[, "upr"], col = "blue", lwd = 1)
pred = predict(m, newdata = x_new, interval = "prediction", level = 0.95)
lines(x_seq, pred[, "lwr"], col = "green", lwd = 1)
lines(x_seq, pred[, "upr"], col = "green", lwd = 1)

#Pour un R intermédiaire : X[,6]
df = data.frame(x = X[, 6], y = Y)
m = lm(y ~ x, data = df)
plot(df$x, df$y, main = "Régression linéaire avec intervalle de confiance à 95%", xlab = "Variable explicative", ylab = "Variable réponse")
abline(m, col = "red", lwd = 1)
x_seq = seq(min(df$x), max(df$x), length.out = 100)
x_new = data.frame(x = x_seq) 
pred = predict(m, newdata = x_new, interval = "confidence", level = 0.95)
lines(x_seq, pred[, "lwr"], col = "blue", lwd = 1)
lines(x_seq, pred[, "upr"], col = "blue", lwd = 1)
pred = predict(m, newdata = x_new, interval = "prediction", level = 0.95)
lines(x_seq, pred[, "lwr"], col = "green", lwd = 1)
lines(x_seq, pred[, "upr"], col = "green", lwd = 1)



######PARTIE 2 - REGRESSION LINEAIRE MULTIPLE ET GENERALISATION######
#Régression linéaire Multiple
#X'X inversible ou non
#avec X1 = X[,-1], sans le vs (variable qualitative)
X1 = X[,-1]
df1 = data.frame(X1)
X1mat = model.matrix(~ ., data = df1)
XtX1 = t(X1mat) %*% X1mat
det(XtX1)
# Doit être ≠ 0
rangX1 = qr(X1mat)$rank # Doit être égal à ncol(Xmat)
#Calcul de beta_chapeau
X1tY = t(X1mat) %*% Y
# XᵗY
beta_chap1 = solve(XtX1) %*% X1tY # (XᵗX)^(-1) XᵗY
#Calcul de R²a et R²
n=31
Y_chap1 = X1mat %*% beta_chap1
R1 = 1 - sum((Y_chap1 - Y)^2)/sum((Y - mean(Y))^2) # = 0.854
R1a = 1-((n-1)/(n-rangX1))*(1-R1) # = 0.817

#avec X2 = X1[-6], sans le vs et carb (variables qualitatives)
X2 = X1[-6]
df2 = data.frame(X2)
X2mat = model.matrix(~ ., data = df2)
XtX2 = t(X2mat) %*% X2mat
det(XtX2)
# Doit être ≠ 0
rangX2 = qr(X2mat)$rank # Doit être égal à ncol(X3mat)
#Calcul de beta_chapeau
X2tY = t(X2mat) %*% Y
# XᵗY
beta_chap2 = solve(XtX2) %*% X2tY # (XᵗX)^(-1) XᵗY
#Calcul de R²a et R²
Y_chap2 = X2mat %*% beta_chap2
R2 = 1 - sum((Y_chap2 - Y)^2)/sum((Y - mean(Y))^2) # = 0.853
R2a = 1-((n-1)/(n-rangX2))*(1-R2) # = 0.823
#Validité des hypothèses sur le bruit
#Residus standardisés
#pour X1
m1 = lm(Y ~ X1mat)
rstm1 = rstandard(m1)
qqnorm(rstm1, main="Q-Q plot des résidus standardisés") #pour regarder si suit une loi normale standart
qqline(rstm1, col="red", lwd=2)#pour X2
m2 = lm(Y ~ X2mat)
rstm2 = rstandard(m2)
qqnorm(rstm2, main="Q-Q plot des résidus standardisés") #pour regarder si suit une loi normale standart
qqline(rstm2, col="red", lwd=2)
#Residus studentisés
#pourX1
rstd1=rstudent(m1)
srst1=sort(rstd1)
nt1=length(rstd1)
yst1=1/nt1*(1:nt1)
xt1=seq(srst1[1],srst1[nt1],0.01)
yt1=pt(xt, n - length(Y)-3)
plot(srst1,yst1,type='s',col='blue',xlim=c(srst1[1],srst1[nt1]),ylim=c(0,1))
lines(xt,yt,col='red')
ks.test(rstd1,'pt',length(Y)-3) # p-value = 0.6241 >> 0.05
#pourX2
rstd2=rstudent(m2)
srst2=sort(rstd2)
nt2=length(rstd2)
yst2=1/nt1*(1:nt2)
xt2=seq(srst2[1],srst2[nt2],0.01)
yt2=pt(xt,length(Y)-3)
plot(srst2,yst2,type='s',col='blue',xlim=c(srst2[1],srst2[nt1]),ylim=c(0,1))
lines(xt,yt,col='red')
ks.test(rstd2,'pt',length(Y)-3) # p-value = 0.7273 >> 0.05
#Test H0 et H1
#H0 : beta[1:p]=0
#H1 : ce n'est pas le cas, au moins 1 des coef est différent de 0
#Pour X1
F1 = (norm(Y_chap1 - mean(Y))^2/(rangX1 - 1))/(norm(Y_chap1 - mean(Y))^2/(n-rangX1))
pf1=pf(F1, rangX1-1, n - rangX1)
p_val1 = 1 - pf1 # = 0.00647
#On décide H1
#Pour X2
F2 = (norm(Y_chap2 - mean(Y))^2/(rangX2 - 1))/(norm(Y_chap2 - mean(Y))^2/(n-rangX2))
pf2=pf(F2, rangX2-1, n - rangX2)
p_val2 = 1 - pf2 # = 0.0026
#On décide H1

######PREDICTION######
#1er cas : on enlève la première colonne seulement
x_new=A[set1, set2[-1], drop=FALSE] #ici on retire la première colonne 
x_new_mat = model.matrix(~ ., data = x_new) #c'est la même formule que X1mat
y_pred = x_new_mat %*% beta_chap1

on2_chap= 1/(n-rangX1) * sum((Y-Y_chap1)^2)

facteur_IC = diag(x_new_mat %*% solve(XtX1) %*% t(x_new_mat))
alpha=0.05
t_alpha=qt( (1- (alpha/2)), df=n-rangX1)

IC_lower = y_pred - sqrt(on2_chap) * sqrt(facteur_IC) * t_alpha
IC_upper = y_pred + sqrt(on2_chap) * sqrt(facteur_IC) * t_alpha

facteur_IP = 1 + facteur_IC
IP_lower = y_pred - sqrt(on2_chap) * sqrt(facteur_IP) * t_alpha
IP_upper = y_pred + sqrt(on2_chap) * sqrt(facteur_IP) * t_alpha

cat("la valeur prédite pour x_new est :", y_pred, "\n")
cat("Un intervalle de confiance à 95% est :", round(IC_lower, 3), ",", round(IC_upper, 3), "]\n")
cat("Un intervalle de prédiction à 95% est :", round(IP_lower, 3), ",", round(IP_upper, 3), "]\n")

#2ème cas : on enlève la colonne 1 et la colonne 6
x_new=A[set1, set2[-c(1,6)], drop=FALSE] #ici on retire la première colonne 
x_new_mat = model.matrix(~ ., data = x_new) #c'est la même formule que X1mat
y_pred = x_new_mat %*% beta_chap2

on2_chap= 1/(n-rangX2) * sum((Y-Y_chap2)^2)

facteur_IC = diag(x_new_mat %*% solve(XtX2) %*% t(x_new_mat))
alpha=0.05
t_alpha=qt( (1- (alpha/2)), df=n-rangX2)

IC_lower = y_pred - sqrt(on2_chap) * sqrt(facteur_IC) * t_alpha
IC_upper = y_pred + sqrt(on2_chap) * sqrt(facteur_IC) * t_alpha

facteur_IP = 1 + facteur_IC
IP_lower = y_pred - sqrt(on2_chap) * sqrt(facteur_IP) * t_alpha
IP_upper = y_pred + sqrt(on2_chap) * sqrt(facteur_IP) * t_alpha

cat("la valeur prédite pour x_new est :", y_pred, "\n")
cat("Un intervalle de confiance à 95% est :", round(IC_lower, 3), ",", round(IC_upper, 3), "]\n")
cat("Un intervalle de prédiction à 95% est :", round(IP_lower, 3), ",", round(IP_upper, 3), "]\n")



######SELECTION DE VARIABLE / TROUVER LE R CARRE AJUSTE LE PLUS GRAND POSSIBLE EN ENLEVANT DES COLONNES######

find_Ra=function(X_initial){
  df=data.frame(X_initial)
  Xmat=model.matrix(~ ., data=df)
  XtX=t(Xmat) %*% Xmat
  rangX = qr(Xmat)$rank
  XtY=t(Xmat) %*% Y
  beta_chap=solve(XtX) %*% XtY
  Y_chap=Xmat %*% beta_chap
  n=length(Y)
  R=1-sum((Y_chap - Y)^2) / sum((Y - mean(Y))^2)
  Ra=1-((n-1)/(n-rangX))*(1-R)
  return (Ra)
}


selection = function(){
  X_initial = X[, -c(1, 7)] # on enlève les colonnes qualitatives
  Ra_initial = find_Ra(X_initial)
  long = ncol(X_initial)
  cat("R² ajusté initial :", round(Ra_initial,5), "\n")
  while (long > 1) {
    best_Ra = Ra_initial
    best_i = NA
    for (i in 1:long) {
      X_new = X_initial[, -i, drop = FALSE]
      Ra_new = find_Ra(X_new)
      if (Ra_new > best_Ra) {
        best_Ra = Ra_new
        best_i = i
      }
    }
    # Si aucune amélioration n'a été trouvée, on s'arrête
    if (is.na(best_i)) {
      break
    }
    cat("Suppression de la variable :", names(X_initial)[best_i], " -> Ra=", round(best_Ra,5), "\n")
    # Mettre à jour X_initial et Ra_initial
    X_initial = X_initial[, -best_i, drop = FALSE]
    Ra_initial = best_Ra
    long = ncol(X_initial)
  }
  cat("Variables finales sélectionnées :", names(X_initial), "\n")
  cat("R² ajusté final :",round(best_Ra,5), "\n")
}

#Lorsque l'on appelle la fonction selection(), on obtient les résultats suivants :
#"
#R² ajusté initial : 0.8239 
#Suppression de la variable : hp  -> Ra= 0.827 
#Suppression de la variable : disp  -> Ra= 0.83004 
#Variables finales sélectionnées : qsec wt drat 
#R² ajusté final : 0.83004 
#"
