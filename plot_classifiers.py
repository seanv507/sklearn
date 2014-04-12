# -*- coding: utf-8 -*-
"""
Created on Sat Apr 12 12:47:47 2014

@author: sv507
"""

names= classifiers.keys()
df_scores=pd.DataFrame(scores_list[0],columns=[names[0]])
for i,n in enumerate(names[1:]):
    df_scores[n]=scores_list[i]
#clf = GridSearchCV(estimator=svc, param_grid=dict(gamma=gammas),
#...                    n_jobs=-1)
# clf_new=GridSearchCV(clf, param_grid, 
#                                 scoring=None, loss_func=None, score_func=None, 
#                                 fit_params=None, n_jobs=1, iid=True, refit=True, 
#                                cv=None, verbose=0, pre_dispatch='2*n_jobs')


# train model

df_scores.sort(axis=1,inplace=True)

plt.clf()
ax=plt.subplot('111')
ax.plot(df_scores.iloc[n_folds,:].values,'ro',label='test')

plt.xticks(range(len(names)),df_scores.columns,rotation='vertical')

ax.bar(range(len(names)),df_scores[:n_folds].mean().values,yerr=df_scores[:n_folds].std().values/sqrt(n_folds-1),align='center',label='x-val')
plt.legend()
