#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 22 11:44:38 2018

@author: angela serra
"""


#%%

from WilliamsPlot import williams_plot
import numpy
from sklearn import cross_validation
from sklearn import linear_model
from sklearn.linear_model import ARDRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn import svm
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.svm import SVR
from sklearn.linear_model import Ridge
import scipy
from math import sqrt

from mlxtend.regressor import StackingRegressor

# Description: Fitness function for regression model with one objective
# Input:
  # individual is the binary chromosome to be evaluated
  # objective_function is the objective_function to be evaluated
  # data is the training dataset
  # y are the labels
  # nf is the number of folds for the cross
  # nb is the number of repetition of the cross validation
  # REG is the regression model
# Output: A tuple containing the mean value of the repeated cross validation for the computed fitness
  # If the criterian related to the number of features are satisfied then the cv is performed and the fitness is computed
  # Otherwise penalization values are returned depending on the fitness creterion
def evalRegression(individual, objective_function,data,y, nf = 2, nb = 5,REG = "LR"):
    #In QSAR models, the common accepted maximum number of features is 1/5 of the number of chemicals
    #max_feat = data.shape[0]/5
    max_feat = 15
    if sum(individual) > max_feat or sum(individual)<3:
      return(check_objective_function(0,0,0,10000,10000,0,0,10000,objective_function))

    con_ind = [i for (i, b) in enumerate(individual) if b == 1]
    X = data[:,con_ind]
    r2_scores = []
    q2_scores = []
    q2f3_scores = []
    rmse_train_scores = []
    rmse_test_scores = []
    mean_ad_train = []
    mean_ad_test = []

    for param in range(nb):
        r2_cv = []
        rmse_train_cv = []
        rmse_test_cv = []
        q2_cv = []
        q2f3_cv = []
        ad_train_cv = []
        ad_test_cv = []
        # normal cross-validation
        hist_cv, bin_limits_cv = numpy.histogram(y, 3)
        bins_cv = numpy.digitize(y, bin_limits_cv)
        bins_cv[bins_cv ==4 ] = 3

        kf = cross_validation.StratifiedKFold(bins_cv,nf) #nel ciclo for basta mettere in kf
        for train_index, test_index in kf:#kf.split(X,y):
            # split the training data
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            r2_res,rmse_test,q2_res,rmse_train,q2f3_res,lr = regressors_function(X_train, X_test, y_train, y_test,REG=REG)
            #forse posso provare a restituire 0 in caso di r2 minore di 0.60 che il valore minimo considerato predittivo???

            if r2_res<0:
              r2_cv.append(0)
            else:
              r2_cv.append(r2_res)

            if q2_res<0:
              q2_cv.append(0)
            else:
              q2_cv.append(q2_res)

            # if q2_res>0.55:
            #   q2_cv.append(q2_res)

            q2f3_cv.append(q2f3_res)

            rmse_train_cv.append(rmse_train)
            rmse_test_cv.append(rmse_test)
            wp = williams_plot(X_train, X_test, y_train, y_test, lr,toPrint = False,toPlot= False)
            ad_test_cv.append(wp[0])
            ad_train_cv.append(wp[1])

        r2_scores.append(numpy.mean(r2_cv))
        q2_scores.append(numpy.mean(q2_cv))
        q2f3_scores.append(numpy.mean(q2f3_cv))
        rmse_train_scores.append(numpy.mean(rmse_train_cv))
        rmse_test_scores.append(numpy.mean(rmse_test_cv))
        mean_ad_train.append(numpy.mean(ad_train_cv))
        mean_ad_test.append(numpy.mean(ad_test_cv))

    r2 = numpy.mean(r2_scores)
    q2 = numpy.mean(q2_scores)
    q2f3 = numpy.mean(q2f3_scores)
    rmse_train = numpy.mean(rmse_train_scores)
    rmse_test = numpy.mean(rmse_test_scores)
    ad_train = numpy.mean(mean_ad_train)
    ad_test = numpy.mean(mean_ad_test)

    if("ad" in objective_function):
      if(wp[2]>=1):#r2<0.6 or q2<0.6 or q2f3<0.6 or wp[2]>=1):
      #if wp[0]<100 or wp[1]<100 or wp[2]>=1:
        #return(check_objective_function(0,0,0,10000,10000,0,0,10000,objective_function))
        return(check_objective_function(0,0,0,10000,10000,0,0,10000,objective_function))

    return(check_objective_function(r2,q2,q2f3,rmse_train,rmse_test,ad_train,ad_test,sum(individual),objective_function))



# This function train and test a predefined regressor method and return the r2 and rmse metrics plus the trained model
# Input
  # X_train is the train set matrix
  # X_test is the test set matrix
  # y_train is the vector of train labels
  # y_test is the vector of test labels
  # REG is the name of the regression model
# Output: is a tuple with r2, rmse and trained model
def regressors_function(X_train, X_test, y_train, y_test,REG="LR"):
  if(REG=="LR"): #linear regressor
    #print("Linear Regression")
    estimators = []
   # estimators.append(('standardize', StandardScaler()))
    estimators.append(('lin_reg', linear_model.LinearRegression()))
    lr = Pipeline(estimators)
  if(REG=="SVM"): #svm with RBF
    #print("SVM")
    estimators = []
    estimators.append(('standardize', StandardScaler()))
    estimators.append(('lin_reg', svm.SVR(kernel='rbf')))
    lr = Pipeline(estimators)
  if(REG=="STACKING"):
    #print("STACKING")
    lr2 = linear_model.LinearRegression()
    svr_lin = SVR(kernel='linear')
    ridge = Ridge(random_state=1)
    svr_rbf = SVR(kernel='rbf')
    knn5 = KNeighborsRegressor(n_neighbors=5)

    lr = StackingRegressor(regressors=[svr_lin, lr2, ridge],
                               meta_regressor=svr_rbf)
  if(REG=="kNN"):
    #print("kNN")
    estimators = []
    estimators.append(('standardize', StandardScaler()))
    estimators.append(('lin_reg', KNeighborsRegressor(n_neighbors=3)))
    lr = Pipeline(estimators)
  if(REG=="RF"):
    #print("RF")
    lr = RandomForestRegressor(n_estimators=500,
                                  oob_score = True,
                                  n_jobs=1,
                                  random_state=1)
  lr.fit(X_train, y_train)
  y_pred = lr.predict(X_test)
  r2 = lr.score(X_train, y_train)
  q2 = lr.score(X_test, y_test)
  rmse_train = sqrt(mean_squared_error(y_train, lr.predict(X_train)))
  rmse_test = sqrt(mean_squared_error(y_test, lr.predict(X_test)))
  q2f3_res = q2f3(y_train, y_test, y_pred)

  return r2,rmse_test,q2,rmse_train,q2f3_res,lr

def q2f3(y_train, y_test, pred_test):
  SSRes = numpy.sum(y_test-pred_test)**2
  SSRes = SSRes/len(y_test)
  SStot = numpy.sum((y_train-numpy.mean(y_train))**2)
  SStot = SStot/len(y_train)
  r2 = 1 - (SSRes/SStot)
  return r2

def q2f2(y_train, y_test, pred_test):
  SSRes = numpy.sum(y_test-pred_test)**2
  SStot = numpy.sum((y_test-numpy.mean(y_test))**2)
  r2 = 1 - (SSRes/SStot)
  return r2

def q2f1(y_train, y_test, pred_test):
  SSRes = numpy.sum(y_test-pred_test)**2
  SStot = numpy.sum((y_test-numpy.mean(y_train))**2)
  r2 = 1 - (SSRes/SStot)
  return r2

def ccc(y_train, y_test, pred_test):
  SSRes = numpy.sum((y_test-numpy.mean(y_test))*(pred_test-numpy.mean(pred_test)))*2
  SStot = numpy.sum((y_test-numpy.mean(y_train))**2) + numpy.sum((pred_test-numpy.mean(pred_test))**2) + len(pred_test) * (numpy.mean(y_test)-numpy.mean(pred_test))**2
  r2 = SSRes/SStot
  return r2

# this function takes in input the regression metrics and return the right fitness depending on the objective function

def check_objective_function(r2,q2,q2f3,rmse_train,rmse_test,ad_train,ad_test,nfeat,objective_function):
  # single objective functions
  if objective_function == 'r2':
    return r2,q2,q2f3
  if objective_function == 'rmse':
    return rmse_train, rmse_test
  if objective_function == 'ad':
    return ad_train,ad_test
  if objective_function == 'nfeat':
    return nfeat,

  #double objective functions
  if objective_function == 'r2_rmse':
    return r2,q2,q2f3, rmse_train,rmse_test
  if objective_function == 'r2_ad':
    return r2,q2,q2f3,ad_train,ad_test
  if objective_function == 'r2_nfeat':
    return r2,q2,q2f3,nfeat
  if objective_function == 'rmse_ad':
    return rmse_train, rmse_test, ad_train,ad_test
  if objective_function == 'rmse_nfeat':
    return rmse_train, rmse_test, nfeat
  if objective_function == 'ad_nfeat':
    return ad_train, ad_test,nfeat

  #triple objective function
  if objective_function == 'r2_rmse_nfeat':
    return r2,q2,q2f3,rmse_train, rmse_test,nfeat
  if objective_function == 'r2_rmse_ad':
    return r2,q2,q2f3, rmse_train, rmse_test, ad_train, ad_test
  if objective_function == 'r2_ad_nfeat':
    return r2,q2,q2f3, ad_train, ad_test,nfeat
  if objective_function =='rmse_ad_nfeat':
    return rmse_train, rmse_test, ad_train, ad_test, nfeat

  #quadruple function
  if objective_function == 'r2_ad_rmse_nfeat':
    return r2,q2,q2f3,rmse_train, rmse_test, ad_train, ad_test, nfeat


#
# #%% RF objectivefunction
#
# #individual = numpy.random.randint(2, size=1196)
# # This function evaluate the fitness of the selected features with a random forest for regression.
# # It takes in input the individual (binary chromosome), the dataset in which it select only the feature that are in the chromosome,
# # The number of folds (nf) and the number of time the cross validation is repeated (nb)
# # This is a two-objective fitness function:
#   # - for each set of features it compute the mean r2 values in the repeated cross validation strategy
#   # - it counts how many features are in the solution
# # The optimal solution is the one with high r2 and few number of features.
# # So the first objective is to by maximized and the second one to be minimized
#
# #Linear regression with four objectives
# def evalLinearRegression_2objs(individual, reg_measure,data,y, nf = 2, nb = 5,REG = "LR"):
#     #In QSAR models, the common accepted maximum number of features is 1/5 of the number of chemicals
#     max_feat = data.shape[0]/5
#     #max_feat = 10
#     if sum(individual) > max_feat or sum(individual)<3:
#       if reg_measure == "R2":
#         return 0, 10000
#       if reg_measure == "RMSE":
#         return 10000,10000
#       else: #applicability domain
#         return 0,0,10000
#
#     con_ind = [i for (i, b) in enumerate(individual) if b == 1]
#     X = data[:,con_ind]
#     mean_scores = []
#     mean_ad_train = []
#     mean_ad_test = []
#     for param in range(nb):
#         scores = []
#         ad_train = []
#         ad_test = []
#         # normal cross-validation
#         hist_cv, bin_limits_cv = numpy.histogram(y, 3)
#         bins_cv = numpy.digitize(y, bin_limits_cv)
#        # bins_cv[len(bins_cv)-2] = bins_cv[len(bins_cv)-1]
#         bins_cv[bins_cv ==4 ] = 3
#
#         kf = cross_validation.StratifiedKFold(bins_cv,nf) #nel ciclo for basta mettere in kf
#        # kf = KFold(n_splits=nf,shuffle=True,random_state=None)
#         for train_index, test_index in kf:#kf.split(X,y):
#             # split the training data
#             X_train, X_test = X[train_index], X[test_index]
#             y_train, y_test = y[train_index], y[test_index]
#
#             r2_res, rmse,lr = regressors_function(X_train, X_test, y_train, y_test,REG=REG)
#             #forse posso provare a restituire 0 in caso di r2 minore di 0.60 che il valore minimo considerato predittivo???
#
#             if reg_measure == "R2":
#               #print r2_res
#               if r2_res<0:
#                 scores.append(0)
#               else:
#                 scores.append(r2_res)
#             else:
#               scores.append(rmse)
#
#             wp = williams_plot (X_train, X_test, y_train, y_test, lr,toPrint = False)
#
#             ad_test.append(wp[0])
#             ad_train.append(wp[1])
#
#             #print(r2_res, wp[0],wp[1])
#         #print("End iteration")
#         # calculate mean score for folds
#         mean_scores.append(numpy.mean(scores))
#         mean_ad_train.append(numpy.mean(ad_train))
#         mean_ad_test.append(numpy.mean(ad_test))
#
#         #mean_fscores.append(numpy.mean(fscores))
#     # get maximum score value
#     r2 = numpy.mean(mean_scores)
#     ad_train_cv = numpy.mean(mean_ad_train)
#     ad_test_cv = numpy.mean(mean_ad_test)
#
#     if reg_measure == "R2" or reg_measure == "RMSE":
#       return r2, sum(individual)
#     else:
#       return ad_train_cv,ad_test_cv, sum(individual)
#
# # Description: Fitness function for regression model with one objective
# # Input:
#   # individual is the binary chromosome to be evaluated
#   # reg_measure is the fitness measure to be evaluated (possible values are R2, RMSE, AD or nFeat)
#   # data is the training dataset
#   # y are the labels
#   # nf is the number of folds for the cross
#   # nb is the number of repetition of the cross validation
#   # REG is the regression model
# # Output: A tuple containing the mean value of the repeated cross validation for the computed fitness
#   # If the criterian related to the number of features are satisfied then the cv is performed and the fitness is computed
#   # Otherwise penalization values are returned depending on the fitness creterion
# def evalLinearRegression_1objs(individual, reg_measure,data,y, nf = 2, nb = 5,REG = "LR"):
#     #In QSAR models, the common accepted maximum number of features is 1/5 of the number of chemicals
#     max_feat = data.shape[0]/5
#     #max_feat = 10
#     if sum(individual) > max_feat or sum(individual)<3:
#       if reg_measure == "R2":
#         return 0,
#       if reg_measure == "nFeat":
#         return 10000,
#       if reg_measure == "RMSE" :
#         return 10000,
#       else: #applicability domain
#         return 0,0
#
#     con_ind = [i for (i, b) in enumerate(individual) if b == 1]
#     X = data[:,con_ind]
#     mean_scores = []
#     mean_ad_train = []
#     mean_ad_test = []
#     for param in range(nb):
#         scores = []
#         ad_train = []
#         ad_test = []
#         # normal cross-validation
#         hist_cv, bin_limits_cv = numpy.histogram(y, 3)
#         bins_cv = numpy.digitize(y, bin_limits_cv)
#        # bins_cv[len(bins_cv)-2] = bins_cv[len(bins_cv)-1]
#         bins_cv[bins_cv ==4 ] = 3
#
#         kf = cross_validation.StratifiedKFold(bins_cv,nf) #nel ciclo for basta mettere in kf
#        # kf = KFold(n_splits=nf,shuffle=True,random_state=None)
#         for train_index, test_index in kf:#kf.split(X,y):
#             # split the training data
#             X_train, X_test = X[train_index], X[test_index]
#             y_train, y_test = y[train_index], y[test_index]
#
#             r2_res, rmse,lr = regressors_function(X_train, X_test, y_train, y_test,REG=REG)
#             #forse posso provare a restituire 0 in caso di r2 minore di 0.60 che il valore minimo considerato predittivo???
#
#             if reg_measure == "R2":
#               #print r2_res
#               if r2_res<0:
#                 scores.append(0)
#               else:
#                 scores.append(r2_res)
#             else:
#               scores.append(rmse)
#
#             wp = williams_plot (X_train, X_test, y_train, y_test, lr,toPrint = False)
#
#             ad_test.append(wp[0])
#             ad_train.append(wp[1])
#
#             #print(r2_res, wp[0],wp[1])
#         #print("End iteration")
#         # calculate mean score for folds
#         mean_scores.append(numpy.mean(scores))
#         mean_ad_train.append(numpy.mean(ad_train))
#         mean_ad_test.append(numpy.mean(ad_test))
#
#         #mean_fscores.append(numpy.mean(fscores))
#     # get maximum score value
#     r2 = numpy.mean(mean_scores)
#     ad_train_cv = numpy.mean(mean_ad_train)
#     ad_test_cv = numpy.mean(mean_ad_test)
#
#     if reg_measure == "nFeat":
#       return sum(individual),
#     if reg_measure == "R2":
#       return r2,
#     if reg_measure == "RMSE":
#       return r2,
#     else:
#       return ad_train_cv,ad_test_cv,
#
# #Linear regression with five objectives
# def evalLinearRegression_5_objs(individual, reg_measure,data,y, nf = 2, nb = 5,REG = "LR"):
#     #In QSAR models, the common accepted maximum number of features is 1/5 of the number of chemicals
#     max_feat = data.shape[0]/5
#     #max_feat = 10
#     if sum(individual) > max_feat or sum(individual)<3:
#         return 0,0,0,10000, 10000
#
#     con_ind = [i for (i, b) in enumerate(individual) if b == 1]
#     X = data[:,con_ind]
#     mean_scores = []
#     mean_ad_train = []
#     mean_ad_test = []
#     mean_rmse=[]
#     for param in range(nb):
#         scores = []
#         ad_train = []
#         ad_test = []
#         rmse_list = []
#         # normal cross-validation
#         hist_cv, bin_limits_cv = numpy.histogram(y, 3)
#         bins_cv = numpy.digitize(y, bin_limits_cv)
#        # bins_cv[len(bins_cv)-2] = bins_cv[len(bins_cv)-1]
#         bins_cv[bins_cv ==4 ] = 3
#
#         kf = cross_validation.StratifiedKFold(bins_cv,nf) #nel ciclo for basta mettere in kf
#        # kf = KFold(n_splits=nf,shuffle=True,random_state=None)
#         for train_index, test_index in kf:#kf.split(X,y):
#             # split the training data
#             X_train, X_test = X[train_index], X[test_index]
#             y_train, y_test = y[train_index], y[test_index]
#
#             r2_res, rmse,lr = regressors_function(X_train, X_test, y_train, y_test,REG=REG)
#
#             #forse posso provare a restituire 0 in caso di r2 minore di 0.60 che il valore minimo considerato predittivo???
#
#             #print r2_res
#             if r2_res<0:
#               scores.append(0)
#             else:
#               scores.append(r2_res)
#
#             rmse_list.append(rmse)
#             wp = williams_plot (X_train, X_test, y_train, y_test, lr,toPrint = False)
#             ad_test.append(wp[0])
#             ad_train.append(wp[1])
#
#         # calculate mean score for folds
#         mean_scores.append(numpy.mean(scores))
#         mean_ad_train.append(numpy.mean(ad_train))
#         mean_ad_test.append(numpy.mean(ad_test))
#         mean_rmse.append(numpy.mean(rmse_list))
#         #mean_fscores.append(numpy.mean(fscores))
#     # get maximum score value
#     r2 = numpy.mean(mean_scores)
#     ad_train_cv = numpy.mean(mean_ad_train)
#     ad_test_cv = numpy.mean(mean_ad_test)
#     rmse = numpy.mean(mean_rmse)
#     return r2,ad_train_cv,ad_test_cv,rmse, sum(individual)
#
#
# #Linear regression with four objectives
# def evalLinearRegression_4_objs(individual, reg_measure,data,y, nf = 2, nb = 5,REG="LR"):
#     #In QSAR models, the common accepted maximum number of features is 1/5 of the number of chemicals
#     max_feat = data.shape[0]/5
#     #max_feat = 10
#     if sum(individual) > max_feat or sum(individual)<3:
#       if reg_measure == "R2":
#         return 0,0,0, 10000
#       else:
#         return 10000,0,0,10000
#
#     con_ind = [i for (i, b) in enumerate(individual) if b == 1]
#     X = data[:,con_ind]
#     mean_scores = []
#     mean_ad_train = []
#     mean_ad_test = []
#     for param in range(nb):
#         scores = []
#         ad_train = []
#         ad_test = []
#         # normal cross-validation
#         hist_cv, bin_limits_cv = numpy.histogram(y, 3)
#         bins_cv = numpy.digitize(y, bin_limits_cv)
#        # bins_cv[len(bins_cv)-2] = bins_cv[len(bins_cv)-1]
#         bins_cv[bins_cv ==4 ] = 3
#
#         kf = cross_validation.StratifiedKFold(bins_cv,nf) #nel ciclo for basta mettere in kf
#        # kf = KFold(n_splits=nf,shuffle=True,random_state=None)
#         for train_index, test_index in kf:#kf.split(X,y):
#             # split the training data
#             X_train, X_test = X[train_index], X[test_index]
#             y_train, y_test = y[train_index], y[test_index]
#
#             r2_res, rmse,lr = regressors_function(X_train, X_test, y_train, y_test,REG=REG)
#
#             if reg_measure == "R2":
#               r2_res = r2_res
#             else:
#               r2_res = rmse
#
#             #forse posso provare a restituire 0 in caso di r2 minore di 0.60 che il valore minimo considerato predittivo???
#
#             #print r2_res
#             if r2_res<0:
#               scores.append(0)
#             else:
#               scores.append(r2_res)
#
#             wp = williams_plot (X_train, X_test, y_train, y_test, lr,toPrint = False)
#
#             ad_test.append(wp[0])
#             ad_train.append(wp[1])
#
#         # calculate mean score for folds
#         mean_scores.append(numpy.mean(scores))
#         mean_ad_train.append(numpy.mean(ad_train))
#         mean_ad_test.append(numpy.mean(ad_test))
#
#     # get maximum score value
#     r2 = numpy.mean(mean_scores)
#     ad_train_cv = numpy.mean(mean_ad_train)
#     ad_test_cv = numpy.mean(mean_ad_test)
#
#     # if r2 < 0.1:
#     #   if reg_measure == "R2":
#     #     return 0,0,0, 10000
#     #   else:
#     #     return 10000,0,0,10000
#
#     return r2,ad_train_cv,ad_test_cv, sum(individual)

# import numpy as np
# from sklearn.model_selection import train_test_split
# X, y = np.arange(10).reshape((5, 2)), range(5)
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
