# -*- coding: utf-8 -*-
"""
Created on Wed Aug 02 10:31:55 2017

@author: sw
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from xgboost.sklearn import XGBClassifier
from sklearn import cross_validation, metrics   #Additional     scklearn functions
from sklearn.grid_search import GridSearchCV   #Perforing grid search
from matplotlib.pylab import rcParams
import matplotlib.pylab as plt


trainData = pd.read_csv('trainData1.csv')
#testData = pd.read_csv('testData1.csv')
target = 'label'
predictors = [#第一批数据 179
              'xCoorMin', 'xCoorVar',
              'timeMax', 'timeMin', 'timeLen', 'timeMean',
              'speedDiffLen', 'speedMax', 'speedMin', 'speedVar', 
              'xSpeedMax', 'xSpeedMin', 'xSpeedDiffLen', 'xSpeedDiffVar', 'xSpeedMode',
              'xOneThirdSpeedMax', 'xOneThirdSpeedVar', 'xOneThirdSpeedLen',
              #第二批数据加上 18380 88
              'xOneFouthSpeedVar', 'xOneFouthSpeedLen', 'xOneFouthSpeedRatio',
              'xOneFifthSpeedVar', 'xOneFifthSpeedLen', 'xOneFifthSpeedRatio',
              ]

#1,确定学习速率和较好n_estimators
def modelfit(alg, dtrain, predictors,useTrainCV=True, cv_folds=5, early_stopping_rounds=50):
    if useTrainCV:
        xgb_param = alg.get_xgb_params()
        xgtrain = xgb.DMatrix(dtrain[predictors].values, label=dtrain[target].values)
        cvresult = xgb.cv(xgb_param, xgtrain, num_boost_round=alg.get_params()['n_estimators'], nfold=cv_folds, show_progress=True)
        alg.set_params(n_estimators=cvresult.shape[0])
    
    #Fit the algorithm on the data
    alg.fit(dtrain[predictors], dtrain['label'],eval_metric='auc')
    
    #Predict training set:
    dtrain_predictions = alg.predict(dtrain[predictors])
    dtrain_predprob = alg.predict_proba(dtrain[predictors])[:,1]
    
    #Print model report:
    print "\nModel Report"
    print "Accuracy : %.4g" % metrics.accuracy_score(dtrain['label'].values, dtrain_predictions)
    print "AUC Score (Train): %f" % metrics.roc_auc_score(dtrain['label'], dtrain_predprob)
    
    feat_imp = pd.Series(alg.booster().get_fscore()).sort_values(ascending=False)
    feat_imp.plot(kind='bar', title='Feature Importances')
    plt.ylabel('Feature Importance Score')


#2,搜索max_depth 和 min_weight
param_test1 = {
 'max_depth':range(3,10,2),
 'min_child_weight':range(1,8,2)
}
gsearch1 = GridSearchCV(estimator = XGBClassifier(         learning_rate =0.1, n_estimators=50, max_depth=5,
min_child_weight=1, gamma=0, subsample=0.8,             colsample_bytree=0.8,
 objective= 'binary:logistic', nthread=4,     scale_pos_weight=1, seed=0), 
 param_grid = param_test1,     scoring='roc_auc',n_jobs=4,iid=False, cv=5)
gsearch1.fit(trainData[predictors],trainData[target])
gsearch1.grid_scores_, gsearch1.best_params_,     gsearch1.best_score_


#

xgb1 = XGBClassifier(
 learning_rate =0.1,
 n_estimators=100,
 max_depth=5,
 min_child_weight=1,
 gamma=0,
 subsample=0.8,
 colsample_bytree=0.8,
 objective= 'binary:logistic',
 nthread=4,
# scale_pos_weight=1,
 seed=0)
modelfit(xgb1, trainData, predictors)



#dtrain = xgb.DMatrix(trainData[predictors], label=trainData[target])
#dtest = xgb.DMatrix(testData[predictors])
#param = {'booster':'gbtree', 
#         'objective':'binary:logistic', 
#         'eta':0.25,
#         'gamma':0.00000001, 
#         'min_child_weight':10, 
#         'max_depth':6,
#         'subsample':0.5, 
#         'colsample_bytree':0.8, 
#         'eval_metric':"logloss",
#         'eval_metric':"auc", 
#         'scale_pos_weight':1, 
#         'seed':0
#         }
##param = {}
#numRound = 88
#clf = xgb.train(param, dtrain, numRound) 
#y = clf.predict(dtest)

instanceIDs = testData.id
res = instanceIDs.to_frame()
res['prob'] = y
res['id'] = res['id'].astype(int)
res = res.sort_values(by='prob')  
res.iloc[0:20000].id.to_csv('./BDC20160715.txt', header=None, index=False)

