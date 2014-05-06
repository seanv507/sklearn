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






# classifiers

classifiers = {
    "Nearest Neighbors": (KNeighborsClassifier(),{'n_neighbors':[3,5]}),
    "Linear SVM":  (SVC(kernel="linear"),{'C':[0.01,0.1,1]}),
    "RBF SVC": (SVC(),[   {'gamma':[2], 'C':[0.01,0.1]},   {'gamma':[4], 'C':[0.01,0.1]}]),
    "Decision Tree": (DecisionTreeClassifier(),{'max_depth':[5]}),
    "Random Forest": (RandomForestClassifier(),{'max_depth':[5], 'n_estimators':[10], 'max_features':[1]}),
    "LDA":(LDA(),[{}]),
    "QDA":(QDA(),[{}])
    }
    

    
    #An empty dict signifies default parameters.
    
# load data set
X_train,y_train=lr.gen_data(1000)
X_test,y_test=lr.gen_data(10000)


    
# set random num generator
rng = np.random.RandomState(2)

# split training /test set
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.4, random_state=rng)
# check repeatable?

# sklearn.cross_validation.StratifiedKFold(y, n_folds=3, indices=True, k=None)
# why doesn't it allow random number generator
n_folds=5
cv=KFold(len(X_train), n_folds=n_folds,random_state=rng,shuffle=True)
# cross validation
scores_list=[]
best_list=[]
for name,(clf,param_grid) in classifiers.iteritems():
    gs_cv=GridSearchCV(clf, param_grid, cv=cv,refit=True)
    gs_cv.fit(X_train,y_train)
    for sim in gs_cv.grid_scores_:
        df=pd.DataFrame(sim.cv_validation_scores,columns=['score'])
        df['clf']=name
        df['params']=str(sim.parameters)
        df['fold']=range(n_folds)
        scores_list.append(df)
    
    sc=gs_cv.score(X_test,y_test)
    best_list.append({'clf':name,'params':str(gs_cv.best_params_),'fold':-1,'score':sc})
    
    
scores_list.append(pd.DataFrame(best_list))
#df_scores all contains
df_scores_all=pd.concat(scores_list)