# -*- coding: utf-8 -*-
"""
Created on Fri Apr 11 22:04:49 2014

@author: sean
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.grid_search import GridSearchCV

from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import cross_val_score
from sklearn.cross_validation import KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.lda import LDA
from sklearn.qda import QDA
import lr




# load data set
X_train,y_train=lr.gen_data(1000)
X_test,y_test=lr.gen_data(10000)


# classifiers
names = ["Nearest Neighbors", "Linear SVM", "RBF SVM", "Decision Tree",
         "Random Forest", "AdaBoost", "Naive Bayes", "LDA", "QDA"]
classifiers = [
    KNeighborsClassifier(3),
    SVC(kernel="linear", C=0.025),
    SVC(gamma=2, C=1),
    DecisionTreeClassifier(max_depth=5),
    RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
    AdaBoostClassifier(),
    GaussianNB(),
    LDA(),
    QDA()]
    
    
# set random num generator
rng = np.random.RandomState(2)

# split training /test set
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.4, random_state=rng)




# sklearn.cross_validation.StratifiedKFold(y, n_folds=3, indices=True, k=None)
cv=KFold(len(X_train), n_folds=5,random_state=rng)
# cross validation
scores_list=[]
for name,clf in zip(names,classifiers):
    scores=cross_val_score(clf,X_train,y_train,cv=cv)
    clf.fit(X_train,y_train)
    sc=clf.score(X_test,y_test)
    np.append(scores,sc)
    scores_list.append(scores)
    
df_scores=pd.DataFrame(scores_list[0],columns=[names[0]])
for i,n in enumerate(names[1:]):
    df_scores[n]=scores_list[i]
# parameter iteration

#param_grid = [
#  {'C': [1, 10, 100, 1000], 'kernel': ['linear']},
#  {'C': [1, 10, 100, 1000], 'gamma': [0.001, 0.0001], 'kernel': ['rbf']},
# ]
#clf = GridSearchCV(estimator=svc, param_grid=dict(gamma=gammas),
#...                    n_jobs=-1)
# clf_new=GridSearchCV(clf, param_grid, 
#                                 scoring=None, loss_func=None, score_func=None, 
#                                 fit_params=None, n_jobs=1, iid=True, refit=True, 
#                                cv=None, verbose=0, pre_dispatch='2*n_jobs')


# train model


# train model