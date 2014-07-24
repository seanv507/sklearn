require(glmnet)
require(doMC)


setwd("~/projects/sklearn/source")
X<-read.table('threshold_pca_image.txt',sep=' ',header=FALSE)
X=data.matrix(X)
y<-read.table('../data/labels.txt')
cv = cv.glmnet(X[,], y$V1,family='multinomial')
# took about a day to run?
plot(cv)
y_pred<-predict(cv,X)
y_pred1=drop(y_pred)
y_pred1[0:10]
mean(max.col(y_pred1)-1==y$V1)
