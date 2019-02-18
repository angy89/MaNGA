#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 22 11:48:12 2018

@author: angela serra
"""
#from objective_functions import evalRFMultiRegression
#from objective_functions import evalRBFsvm
from deap import base
from deap import creator
from deap import tools
from deap import algorithms
import multiprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
from math import sqrt
import numpy
import cPickle as pickle
import numpy,sys,random
from bisect import bisect
from WilliamsPlot import williams_plot
import numpy
from sklearn import cross_validation
from sklearn import linear_model
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn import svm
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn.neighbors import KNeighborsRegressor
from objective_functions import regressors_function, q2f1,q2f2,q2f3,ccc
import csv

#%% Bit Generation

def list_duplicates_of(seq,item):
    start_at = -1
    locs = []
    while True:
        try:
            loc = seq.index(item,start_at+1)
        except ValueError:
            break
        else:
            locs.append(loc)
            start_at = loc
    return locs

# I can generate bit for the cromosomes with a certain probability of having zero and a certain probability of having one
# In this case, if I want to have less ones, I can set the probability of one to 5% or 1% either.
def generateBit():
    return int(numpy.random.choice([0, 1], size=(1,), p=[.99999, .00001])[0])

############################################ Weigthed Random Generator of Int
def weightedChoice(values, weights):
    total = 0
    cum_weights = []
    for w in weights:
        total += w
        cum_weights.append(total)
    x = random.random() * total
    i = bisect(cum_weights, x)
    return values[i]

######################################################## Ranking Functions
def compileProbs(input_array):
    if numpy.unique(input_array).shape[0]==1:
        pass #do thing if the input_array is constant
    else:
        rr = (input_array-numpy.min(input_array))/numpy.ptp(input_array)
        rr *= (0.8-0.1)
        rr += .1
    return rr


# This function exchange the best top k% of the population (deme) from two niches by using the pipein and pipeout connections.
def migPipe(deme, k, pipein, pipeout, selection, replacement=None):
    print("migPipe")
    print(len(deme))
    print(k)
    emigrants = selection(deme, k)
    if replacement is None:
        # If no replacement strategy is selected, replace those who migrate
        immigrants = emigrants
    else:
        # Else select those who will be replaced
        immigrants = replacement(deme, k)
    
    print("Before send emigrants")
    pipeout.put(emigrants)
    
    print("Emigrants sent")
    print("Before receive")
    buf = pipein.get()
    print("After receive")
    
    for place, immigrant in zip(immigrants, buf):
    #    print("in for")
        indx = deme.index(place)
        deme[indx] = immigrant


def jaccard2(st1, st2, feat_names):
    st1 = [feat_names[i] for (i, b) in enumerate(st1) if b == 1]
    st2 = [feat_names[i] for (i, b) in enumerate(st2) if b == 1]
    st1 = set(st1)
    st2 = set(st2)
    union = st1.union(st2)
    inter = st1.intersection(st2)
    if len(inter) == 0: 
        res = 0
    else:
        res = (float(len(inter))/float(len(union)))
    return res

#Compute the jaccard index between the individuals in the population
def getSimScore2(population, feat_names):
    sum_sim_score = 0
    ncomp = 0
    for i in range(len(population)):
        for j in range(len(population)):
            if i > j:
                sum_sim_score = sum_sim_score + jaccard2(population[i],
                                                         population[j],
                                                         feat_names)
                ncomp += 1
    return float(sum_sim_score/ncomp)

# Calculate the number of individual in the population that corresponds to its "percent"
def percentage(percent, whole):
    return (percent * whole) / 100.0
    
# This function compile metrics based on fitness results. As a results it gives the mean values for each objective across the whole population.
# N.B. Chromosomes that were penalized during the fitness evaluation (and do not respect specific criteria) are not taked into account for the metrics evaluation.
# Input:
  # Fits is a list of tuples. The list is long as the number of individuals evaluated. Each tuple has a certain number of elements depending on the objective function
# Output:
  # A tuple that contains the mean value for the objective functions across all the individuals
def compile_metrics(fits, population):
  nobj = len(fits[0])
  
  #extract all fitness information for the population
  objs = []
  for i in range(nobj):
    obj =[a[i] for a in fits]
    objs.append(obj)
  
  # remove individual that were penalized during fitness computation
  idx = []
  for i in range(len(fits)):
    check = True
    for j in range(nobj):
      if objs[j][i]==10000 or objs[j][i]==0:
        check = False
    if check:
        idx.append(i)
        
  idx = map(int, idx)
  if(len(idx)==0): #if no chromosome have an admissible solutions an empty tuple is returned
   # print "len 0"
    res = (0,) * nobj
    res = res + (objs[0],)
    return(res)
  if(len(idx)==1):#if only one chromosome have an admissible solutions a tuple with its fitness values is returned
    #print "len 1"
    res = ()
    for index in range(nobj):
      res = res + ([objs[index][i] for i in idx],)
  
    res = res + (objs[0],)
    return(res)
  else:
    #print "len n"
    for index in range(nobj):
      rt= []
      for i in idx: rt.append(objs[index][i])
      objs[index] = rt

  res = ()
  for index in range(nobj):
    res = res + (sum(objs[index])/len(idx),)
    
  res = res + (objs[0],)
  return(res)

#Functions to fill the loogbook for single objective functions
def fill_loogbook_r2(logbook, procid,gen,compiled_metrics,ssc=0,time=0,min_max_param=0):
      logbook.record(procid=procid,gen=gen,r2=compiled_metrics[0],q2=compiled_metrics[1],q2f3=compiled_metrics[2], ssc=ssc,time=time, max=min_max_param)
      return(logbook)
      
def fill_loogbook_rmse(logbook, procid,gen,compiled_metrics,ssc=0,time=0,min_max_param=0):
      logbook.record(procid=procid,gen=gen,rmse_tr=compiled_metrics[0],rmse_te=compiled_metrics[1], ssc=ssc,time=time, max=min_max_param)
      return(logbook)

def fill_loogbook_ad(logbook, procid,gen,compiled_metrics,ssc=0,time=0,min_max_param=0):
      logbook.record(procid=procid,gen=gen,ad_train=compiled_metrics[0], ad_test=compiled_metrics[0],ssc=ssc,time=time, max=min_max_param)
      return(logbook)
      
def fill_loogbook_nfeat(logbook, procid,gen,compiled_metrics,ssc=0,time=0,min_max_param=0):
      logbook.record(procid=procid,gen=gen,nFeat=compiled_metrics[0], ssc=ssc,time=time, max=min_max_param)
      return(logbook)

#Functions to fill the loogbook for double objective functions
def fill_loogbook_r2_rmse(logbook, procid,gen,compiled_metrics,ssc=0,time=0,min_max_param=0):
      logbook.record(procid=procid,gen=gen,r2=compiled_metrics[0],q2=compiled_metrics[1],q2f3=compiled_metrics[2],rmse_tr = compiled_metrics[3],rmse_te=compiled_metrics[4], ssc=ssc,time=time, max=min_max_param)
      return(logbook)
      
def fill_loogbook_r2_ad(logbook, procid,gen,compiled_metrics,ssc=0,time=0,min_max_param=0):
      logbook.record(procid=procid,gen=gen,r2=compiled_metrics[0],q2=compiled_metrics[1],q2f3=compiled_metrics[2],ad_train = compiled_metrics[3],ad_test= compiled_metrics[4], ssc=ssc,time=time, max=min_max_param)
      return(logbook)
      
def fill_loogbook_r2_nfeat(logbook, procid,gen,compiled_metrics,ssc=0,time=0,min_max_param=0):
      logbook.record(procid=procid,gen=gen,r2=compiled_metrics[0],q2=compiled_metrics[1],q2f3=compiled_metrics[2],nFeat=compiled_metrics[3], ssc=ssc,time=time, max=min_max_param)
      return(logbook)
      
def fill_loogbook_rmse_ad(logbook, procid,gen,compiled_metrics,ssc=0,time=0,min_max_param=0):
      logbook.record(procid=procid,gen=gen,rmse_tr=compiled_metrics[0],rmse_te=compiled_metrics[1], ad_train = compiled_metrics[2], ad_test= compiled_metrics[3], ssc=ssc,time=time, max=min_max_param)
      return(logbook)      
      
def fill_loogbook_rmse_nfeat(logbook,procid, gen,compiled_metrics,ssc=0,time=0,min_max_param=0):
      logbook.record(procid=procid,gen=gen,rmse_tr=compiled_metrics[0],rmse_te=compiled_metrics[1],nFeat=compiled_metrics[2], ssc=ssc,time=time, max=min_max_param)
      return(logbook)

def fill_loogbook_ad_nfeat(logbook, procid,gen,compiled_metrics,ssc=0,time=0,min_max_param=0):
      logbook.record(procid=procid,gen=gen,ad_train = compiled_metrics[0],ad_test = compiled_metrics[1], nFeat=compiled_metrics[2], ssc=ssc,time=time)
      return(logbook)
      
#Functions to fill the loogbook for triple objective functions
def fill_loogbook_r2_rmse_nfeat(logbook, procid,gen,compiled_metrics,ssc=0,time=0, min_max_param=0):
      logbook.record(procid=procid,gen=gen,r2=compiled_metrics[0],q2=compiled_metrics[1],q2f3=compiled_metrics[2],rmse_tr = compiled_metrics[3],rmse_te = compiled_metrics[4],nFeat = compiled_metrics[5], ssc=ssc,time=time, max=min_max_param)
      return(logbook)

def fill_loogbook_r2_rmse_ad(logbook, procid,gen,compiled_metrics,ssc=0,time=0, min_max_param=0):
      logbook.record(procid=procid,gen=gen,r2=compiled_metrics[0],q2=compiled_metrics[1],q2f3=compiled_metrics[2],rmse_tr = compiled_metrics[3],rmse_te = compiled_metrics[4],ad_train = compiled_metrics[5], ad_test=compiled_metrics[6], ssc=ssc,time=time, max=min_max_param)
      return(logbook)

def fill_loogbook_r2_ad_nfeat(logbook, procid,gen,compiled_metrics,ssc=0,time=0, min_max_param=0):
      logbook.record(procid=procid,gen=gen,r2=compiled_metrics[0],q2=compiled_metrics[1],q2f3=compiled_metrics[2],ad_train = compiled_metrics[3],ad_test = compiled_metrics[4], nFeat=compiled_metrics[5], ssc=ssc,time=time, max=min_max_param)
      return(logbook)
     
def fill_loogbook_rmse_ad_nfeat(logbook, procid,gen,compiled_metrics,ssc=0,time=0, min_max_param=0):
      logbook.record(procid=procid,gen=gen,rmse_tr = compiled_metrics[0],rmse_te = compiled_metrics[1],ad_train = compiled_metrics[2],ad_test = compiled_metrics[3], nFeat=compiled_metrics[4], ssc=ssc,time=time, min=min_max_param)
      return(logbook)

#Functions to fill the loogbook for quadruple objective functions

def fill_loogbook_r2_ad_rmse_nfeat(logbook, procid,gen,compiled_metrics,ssc=0,time=0, min_max_param=0):
      #,q2f3=compiled_metrics[2]
      logbook.record(procid=procid,gen=gen,r2=compiled_metrics[0],q2=compiled_metrics[1],q2f3=compiled_metrics[2],
                                            rmse_tr = compiled_metrics[3],rmse_te = compiled_metrics[4],ad_train = compiled_metrics[5],
                                            ad_test = compiled_metrics[6],nFeat=compiled_metrics[7], ssc=ssc,time=time, max=min_max_param)
      return(logbook)
    

# def fill_loogbook_r2_cor_ad_nfeat(logbook, procid,gen,compiled_metrics,ssc=0,time=0, min_max_param=0):
#       print(compiled_metrics)
#       logbook.record(procid=procid,gen=gen,r2=compiled_metrics[0],cor= compiled_metrics[1],ad_train = compiled_metrics[2],ad_test = compiled_metrics[3], nFeat=compiled_metrics[4], ssc=ssc,time=time, max=min_max_param)
#       return(logbook)
def percentage(percent, whole):
    return (percent * whole) / 100.0


def print_logbook_file_console(file,logbook):
  temp = sys.stdout #assign console output to a variable
  sys.stdout = file 
  print(logbook)
  sys.stdout = temp #set stdout back to console output
    
def compute_outside_metrics(X, y, REG,nf=10):
  hist_cv, bin_limits_cv = numpy.histogram(y, 3)
  bins_cv = numpy.digitize(y, bin_limits_cv)
  bins_cv[bins_cv ==4 ] = 3
  CVmse = [] 
  Q2 = [] 
  Q2F1 = [] 
  Q2F2 = [] 
  Q2F3 = [] 
  CCC = [] 

  kf = cross_validation.StratifiedKFold(bins_cv,nf) #nel ciclo for basta mettere in kf
  for train_index, test_index in kf:#kf.split(X,y):
      # split the training data
      X_train, X_test = X[train_index], X[test_index]
      y_train, y_test = y[train_index], y[test_index]
      
      r2_res,rmse_test,q2_res,rmse_train,q2f3_res,lr = regressors_function(X_train, X_test, y_train, y_test,REG=REG)
      y_pred = lr.predict(X_test)
      
      Q2F1.append(q2f1(y_train, y_test, y_pred))
      Q2F2.append(q2f2(y_train, y_test, y_pred))
      Q2F3.append(q2f3(y_train, y_test, y_pred))
      CCC.append(ccc(y_train, y_test, y_pred))
      Q2.append(q2_res)
      CVmse.append(rmse_test)
      
  q2_m = numpy.mean(Q2)
  q2f1_m = numpy.mean(Q2F1)
  q2f2_m = numpy.mean(Q2F2)
  q2f3_m = numpy.mean(Q2F3)
  ccc_m = numpy.mean(CCC)
  CVmse_m = numpy.mean(CVmse)
  return(q2_m, q2f1_m,q2f2_m,q2f3_m,ccc_m,CVmse_m)
      
def analyse_population(procid,population, output_folder,X_train, X_test, y_train, y_test,features,obj_type,REG = 'RF',k=10):
      
  with open(output_folder + "solutions_procid_%s.txt"% procid, "a") as myfile:
    best_indices = tools.selNSGA2(population, k)
    for j in range(k):
     # Picking best individual
      best_ind = best_indices[j]
      if numpy.sum(best_ind)<1:
        print("No features in the solution")
      else:
        indices = [i for i, x in enumerate(best_ind) if x == 1]
        #.write("%s, %s " % (features.iloc[indices,0].tolist(), best_ind.fitness.values))
       # print("Best individual is %s, %s" % (features.iloc[indices,0].tolist(), best_ind.fitness.values))
        opt_feat = features.iloc[indices,0]
        XTrain = X_train[:,indices]
        XTest = X_test[:,indices]
        r2_res,rmse_test,q2_res,rmse_train,q2f3_res,lr = regressors_function(XTrain, XTest, y_train, y_test,REG=REG)
        wp = williams_plot(XTrain, XTest, y_train, y_test, lr,toPrint = True,toPlot = True, path = output_folder,filename = str(j) + "_" + str(procid)) 
        q2_m, q2f1_m,q2f2_m,q2f3_m,ccc_m,CVmse_m = compute_outside_metrics(XTrain, y_train, REG = REG, nf=10)
        
        pred_test = lr.predict(XTest)
        pred_train = lr.predict(XTrain)
        rms_test = sqrt(mean_squared_error(y_test, pred_test))
        rms_train = sqrt(mean_squared_error(y_train,pred_train))
        
        with open(output_folder +'prediction/test_prediction_solution_'+str(j) + "_nicchia_" + str(procid), 'w') as f1:
          writer = csv.writer(f1, delimiter='\t')
          writer.writerows(zip(y_test,pred_test))
        f1.close()
        
        with open(output_folder +'prediction/train_prediction_solution_'+str(j) + "_nicchia_" + str(procid), 'w') as f2:
          writer = csv.writer(f2, delimiter='\t')
          writer.writerows(zip(y_train,pred_train))
        f2.close()
        
        r2_test = lr.score(XTest,y_test)
        r2_train = lr.score(XTrain, y_train)
        myfile.write("%s, %s, %s, %s, %s, %s, %s,%s, %s, %s, %s, %s, %s\n" % (best_ind.fitness.values,r2_test,r2_train, q2_m, q2f1_m,q2f2_m,q2f3_m,ccc_m,wp[0], wp[1],rms_test,rms_train,features.iloc[indices,0].tolist()))

  myfile.close()
  f = open(output_folder + "best_indices_procid_%d"% procid, 'wb')
  pickle.dump(best_indices, f)
  f.close()

# this function takes in input the regression metrics and return the right fitness depending on the objective function
def check_objective_params(objective_function):
  # single objective functions
  if objective_function == 'r2':
    return ((1.0,1.0,1.0),['procid','gen','r2','q2','q2f3', 'ssc','time', 'max'],max,1,fill_loogbook_r2)
  if objective_function == 'rmse':
    return ((-1.0,-1.0),['procid','gen','rmse_tr','rmse_te', 'ssc','time', 'max'],min,1,fill_loogbook_rmse)
  if objective_function == 'ad':
    return ((1.0,1.0) ,['procid','gen','ad_train','ad_test', 'ssc','time', 'max'],max,2,fill_loogbook_ad)
  if objective_function == 'nfeat':
    return ((-1.0,), ['procid','gen','nFeat', 'ssc','time', 'max'],max,1,fill_loogbook_nfeat)

  #double objective functions
  if objective_function == 'r2_rmse':
    return ((1.0,1.0,1.0,-1.0,-1.0),['procid','gen','r2','q2','q2f3','rmse_tr','rmse_te', 'ssc','time', 'max'],max,2,fill_loogbook_r2_rmse)
  if objective_function == 'r2_ad':
    return ((1.0,1.0,1.0,1.0,1.0),['procid','gen','r2','q2','q2f3','ad_train','ad_test', 'ssc','time', 'max'],max,3,fill_loogbook_r2_ad)
  if objective_function == 'r2_nfeat':
    return ((1.0,1.0,1.0,-1.0),['procid','gen','r2','q2','q2f3','nFeat', 'ssc','time', 'max'],max,2,fill_loogbook_r2_nfeat)
  if objective_function == 'rmse_ad':
    return ((-1.0,-1.0, 1.0,1.0),['procid','gen','rmse_tr','rmse_te','ad_train','ad_test', 'ssc','time', 'max'],min,3,fill_loogbook_rmse_ad)
  if objective_function == 'rmse_nfeat':
    return ((-1.0,-1.0 ,-1.0) ,['procid','gen','rmse_tr','rmse_te','nFeat', 'ssc','time', 'max'],min,2,fill_loogbook_rmse_nfeat)
  if objective_function == 'ad_nfeat':
    return ((1.0, 1.0,-1.0),['procid','gen','ad_train','ad_test','nFeat', 'ssc','time', 'max'],max,3,fill_loogbook_ad_nfeat)
    
  #triple objective function
  if objective_function == 'r2_rmse_nfeat':
    return ((1.0,1.0,1.0,-1.0,-1.0,-1.0),['procid','gen','r2','q2','q2f3','rmse_tr','rmse_te','nFeat', 'ssc','time', 'max'],max,3,fill_loogbook_r2_rmse_nfeat)
  if objective_function == 'r2_rmse_ad':
    return ((1.0,1.0,1.0, -1.0,-1.0, 1.0, 1.0),['procid','gen','r2','q2','q2f3','rmse_tr','rmse_te','ad_train','ad_test', 'ssc','time', 'max'],max,4,fill_loogbook_r2_rmse_ad)
  if objective_function == 'r2_ad_nfeat':
    return ((1.0,1.0,1.0, 1.0, 1.0,-1.0),['procid','gen','r2','q2','q2f3','ad_train','ad_test','nFeat', 'ssc','time', 'max'],max,4,fill_loogbook_r2_ad_nfeat)
  if objective_function =='rmse_ad_nfeat':
    return ((-1.0, -1.0,1.0, 1.0, -1.0),['procid','gen','rmse_tr','rmse_te','ad_train','ad_test','nFeat', 'ssc','time', 'max'],min,4,fill_loogbook_rmse_ad_nfeat)
    
  #quadruple function  
  if objective_function == 'r2_ad_rmse_nfeat':
    return ((1.0,1.0, 1.0,-1.0,-1.0, 1.0, 1.0, -1.0),['procid','gen','r2','q2','q2f3','rmse_tr','rmse_te','ad_train','ad_test','nFeat', 'ssc','time', 'max'],max,5,fill_loogbook_r2_ad_rmse_nfeat)

# def compile_metrics_4objs(fits, population):
#       # Compile metrics
#     r2 = [a[0] for a in fits]
#     ad_train = [a[1] for a in fits]
#     ad_test = [a[2] for a in fits]
#     nFeat = [a[3] for a in fits]
#     
#     idx = [i for i, x in enumerate(nFeat) if x != 10000]
#     idx = map(int, idx)
#     if(len(idx)==0):
#       return(0,0,0,0,r2)
#     if(len(idx)==1):
#       return([r2[i] for i in idx],[ad_train[i] for i in idx],[ad_test[i] for i in idx],[nFeat[i] for i in idx],r2)
#     else:
#       r2_t = []
#       for i in idx: r2_t.append(r2[i])
#       r2 = r2_t
#       
#       nFeat_t = []
#       for i in idx: nFeat_t.append(nFeat[i])
#       nFeat = nFeat_t
#       
#       ad_train_t = []
#       for i in idx: ad_train_t.append(ad_train[i])
#       ad_train = ad_train_t
#       ad_test_t = []
#       for i in idx: ad_test_t.append(ad_test[i])
#       ad_test = ad_test_t
# 
#     # Update current mean fitness
#     mean_r2 = sum(r2)/len(idx)
#     mean_ad_train = sum(ad_train)/len(idx)
#     mean_ad_test = sum(ad_test)/len(idx)
#     mean_feature = sum(nFeat)/len(idx)
#     
#     return(mean_r2,mean_ad_train,mean_ad_test,mean_feature,r2)
# 
# def compile_metrics_5objs(fits, population):
#       # Compile metrics
#     r2 = [a[0] for a in fits]
#     ad_train = [a[1] for a in fits]
#     ad_test = [a[2] for a in fits]
#     rmse = [a[3] for a in fits]
#     nFeat = [a[4] for a in fits]
#     
#     idx = [i for i, x in enumerate(nFeat) if x != 10000]
#     idx = map(int, idx)
# 
#     if(len(idx)==0):
#       return(0,0,0,0,0,r2)
#     if(len(idx)==1):
#       return([r2[i] for i in idx],[ad_train[i] for i in idx],[ad_test[i] for i in idx],[rmse[i] for i in idx],[nFeat[i] for i in idx],r2)
#       #return(r2[idx],ad_train[idx],ad_test[idx],rmse[idx],nFeat[idx],r2)
#     else:
#       r2_t = []
#       for i in idx: r2_t.append(r2[i])
#       r2 = r2_t
#       rmse_t = []
#       for i in idx: rmse_t.append(rmse[i])
#       rmse = rmse_t
#       
#       nFeat_t = []
#       for i in idx: nFeat_t.append(nFeat[i])
#       nFeat = nFeat_t
#       
#       ad_train_t = []
#       for i in idx: ad_train_t.append(ad_train[i])
#       ad_train = ad_train_t
#       ad_test_t = []
#       for i in idx: ad_test_t.append(ad_test[i])
#       ad_test = ad_test_t
# 
# 
#       # ad_train = ad_train[idx]
#       # ad_test = ad_test[idx]
#       # nFeat = nFeat[idx]    
#     # Update current mean fitness
#     mean_r2 = sum(r2)/len(idx)
#     mean_rmse = sum(rmse)/len(idx)
#     mean_ad_train = sum(ad_train)/len(idx)
#     mean_ad_test = sum(ad_test)/len(idx)
#     mean_feature = sum(nFeat)/len(idx)
#     
#     return(mean_r2,mean_ad_train,mean_ad_test,mean_rmse,mean_feature,r2)
# 
# #this funciton applie to r2 and rmse
# def compile_metrics_1objs(fits, population):
#     r2 = [a[0] for a in fits]
#     idx = [i for i, x in enumerate(r2) if x != 0]
#     idx = map(int, idx)
# 
#     if(len(idx)==0):
#       return(0,r2)
#     if(len(idx)==1):
#       #return(r2[idx],nFeat[idx],r2)
#       return([r2[i] for i in idx],r2)
#     else:
#       r2_t = []
#       for i in idx: r2_t.append(r2[i])
#       r2 = r2_t
#       mean_r2 = sum(r2)/len(idx)
#       
#     return mean_r2,r2 
# 
# 
# def compile_metrics_2objs(fits, population):
#       # Compile metrics
#     r2 = [a[0] for a in fits]
#     nFeat = [a[1] for a in fits]
# 
#     idx = [i for i, x in enumerate(nFeat) if x != 10000]
#     idx = map(int, idx)
# 
#     if(len(idx)==0):
#       return(0,0,r2)
#     if(len(idx)==1):
#       #return(r2[idx],nFeat[idx],r2)
#       return([r2[i] for i in idx],[nFeat[i] for i in idx],r2)
#     else:
#       r2_t = []
#       for i in idx: r2_t.append(r2[i])
#       r2 = r2_t
#       
#       nFeat_t = []
#       for i in idx: nFeat_t.append(nFeat[i])
#       nFeat = nFeat_t
# 
#       #r2 = r2[idx]
#       #nFeat = nFeat[idx]
# 
#     
#     # Update current mean fitness
#     mean_r2 = sum(r2)/len(idx)
#     mean_feature = sum(nFeat)/len(idx)
#     return(mean_r2,mean_feature,r2)
# 
# def compile_metrics_3objs(fits, population):
#       # Compile metrics
#     ad_train = [a[0] for a in fits]
#     ad_test = [a[1] for a in fits]
#     nFeat = [a[2] for a in fits]
#     
#     idx = [i for i, x in enumerate(nFeat) if x != 10000]
#     idx = map(int, idx)
# 
#     if(len(idx)==0):
#       return(0,0,0,ad_train)
#     if(len(idx)==1):
#       return([ad_train[i] for i in idx],[ad_test[i] for i in idx],[nFeat[i] for i in idx],ad_train)
#       #return(ad_train[idx],ad_test[idx],nFeat[idx],ad_train)
#     else:
#       nFeat_t = []
#       for i in idx: nFeat_t.append(nFeat[i])
#       nFeat = nFeat_t
#       
#       ad_train_t = []
#       for i in idx: ad_train_t.append(ad_train[i])
#       ad_train = ad_train_t
#       ad_test_t = []
#       for i in idx: ad_test_t.append(ad_test[i])
#       ad_test = ad_test_t
#       
#     # Update current mean fitness
#     mean_ad_train = sum(ad_train)/len(idx)
#     mean_ad_test = sum(ad_test)/len(idx)
#     mean_feature = sum(nFeat)/len(idx)
#     
#     return(mean_ad_train,mean_ad_test,mean_feature,ad_train)

# #Desirability functions from https://cran.r-project.org/web/packages/desirability/vignettes/desirability.pdf
# def desirability_max(x,a,b,s=1):
#   if x < a:
#     return 0
#   elif x>b:
#     return 1
#   else:
#     return ((x-a)/(b-a))^1
# 
# def desirability_min(x,a,b,s=1):
#   if x > b:
#     return 0
#   elif x < a:
#     return 1
#   else:
#     return ((x-b)/(a-b))^1
# 
# def merge_desirability(des_vect):
#   return(scipy.stats.mstats.gmean(des_vect))
# 
# def rank_feature(population,K=10):
#   best = tools.selNSGA2(population,K)
#   sum_vect = [0] * len(population[1])
#   
#   for i in range(len(population)) :
#     for j in range(len(sum_vect)):
#       sum_vect[j] = sum_vect[j] + population[i][j]
#       
#   return sum_vect
# 
# # This function compute mutation of a single individual based on how many times the
# # feature is selected in important solutions
# def FlipBit_rankingBased(individual, ranking,indpb,K2=5):
#     """Flip the value of the attributes of the input individual and return the
#     mutant. The *individual* is expected to be a :term:`sequence` and the values of the
#     attributes shall stay valid after the ``not`` operator is called on them.
#     The *indpb* argument is the probability of each attribute to be
#     flipped. This mutation is usually applied on boolean individuals.
#     :param individual: Individual to be mutated.
#     :param indpb: Independent probability for each attribute to be flipped.
#     :returns: A tuple of one individual.
#     This function uses the :func:`~random.random` function from the python base
#     :mod:`random` module.
#     """
#     # for i in xrange(len(individual)):
#     #     if random.random() < indpb:
#     #         individual[i] = type(individual[i])(not individual[i])
#     for i in xrange(len(individual)):
#       if individual[i]==1: #se la feature e' selezionata
#         if ranking[i]<K2: # se la feature non e' ricorrente tra quelle importanti
#           if random.random() < indpb: #allora la muto con una certa prob
#             individual[i] = type(individual[i])(not individual[i])
#       if individual[i] == 0: #se la feature non e' selezionata
#         if ranking[i]>=K2: #se la feature e' importante
#           if random.random() < indpb: #allora la muto con una certa prob
#             individual[i] = type(individual[i])(not individual[i])
# 
#     return individual,
    

