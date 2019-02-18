"""
Created on Wednesday march 21 2018
@author: angela serra
"""
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.linear_model import ElasticNet
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.feature_selection import univariate_selection
from sklearn.feature_selection import f_classif
from collections import deque
from multiprocessing import Event, Pipe, Process, Queue
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from operator import itemgetter
from bisect import bisect
import numpy
import random
import cPickle as pickle
import pandas as panda
import scipy.cluster.hierarchy as sch
import scipy.sparse as sp
import scipy.spatial as spt
import scipy.stats as sps
import itertools
from collections import Counter
import numbers
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import dill
from sklearn import svm
from manga_algorithm import manga_algorithm
from sklearn import cross_validation #, grid_search
from deap import tools
from sklearn import linear_model
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from WilliamsPlot import williams_plot#, williams_plot_to_plot
from sklearn.metrics import mean_squared_error
from math import sqrt
import scipy,getopt,sys

from multiprocessing import Event, Pipe, Process
from collections import deque

from utility_functions import percentage
import os

# How to start the algorithm with an initial ranking of features
# How to select the most frequent used features in the different chromosomes
# How to change dimanically the probability of mutation of a feature

# Aggiungere il coefficiente di correlazione di pearson
# Aggiungere le multi niches
# Aggiungere le funzioni di desirability

if __name__ == "__main__":

  arg = sys.argv

  input_folder = arg[1]
  output_folder = arg[2]
  obj_type = arg[3]
  REG = arg[4]

  if not os.path.exists(output_folder + "wp/"):
    os.makedirs(output_folder + "wp/")
  if not os.path.exists(output_folder + "prediction/"):
    os.makedirs(output_folder + "prediction/")

  #default parameters
  test_perc = 0.1
  ngen = 500 # 50 is used for the simulated dataset, 500 for the real dataset
  npop = 500 # 500 for real data
  NPROC = 1
  nf = 3 #10 for real data
  nb = 3
  n_bins = 5
  gen_bit_cor = False
  numN = 20 #number of niches; 1 is used for the simulated datasets, 20 is used for the real data
  save_rate = 25
  best_k = int(percentage(25, npop))
  mig_rate = 25

  print(obj_type)
  print(REG)
  print(input_folder)
  print("Reading data...")
  X = numpy.loadtxt(input_folder+"X.txt",delimiter = ",")
  features = pd.read_csv(input_folder+"features.txt",delimiter=",",header=None)
  y_labels = numpy.loadtxt(input_folder+"y.txt",delimiter = ",")

  if input_folder == "../qsar_HIA/":
    good = abs(y_labels)>-0.5
    y_labels=y_labels[good]
    X = X[good,:]

  print("Scaling data...")
  # data standardization
  scaler = StandardScaler()
  scaler.fit(X)
  X = scaler.transform(X)
  #X_train, X_test, y_train, y_test = train_test_split(X, y_labels, test_size=0.20, random_state=42)


  hist, bin_limits = numpy.histogram(y_labels, n_bins)
  bins = numpy.digitize(y_labels, bin_limits)
  bins[bins==(n_bins + 1)] = n_bins

  input_list = []
  for index in range(numN):
    print("Creating training test")
    X_train, X_test, y_train, y_test = cross_validation.train_test_split(X,y_labels,test_size=test_perc, stratify=bins)
    data = X_train
    y = y_train

    cor_feat = []
    for i in range(X_train.shape[1]):
      cor_feat.append(abs(scipy.stats.pearsonr(X_train[:,i],y_train)[0]))

    input_list.append([X_train,X_test,y_train,y_test,data,y,cor_feat])

  # pipes = [Pipe(False) for _ in range(numN)]
  # pipes_in = deque(p[0] for p in pipes)
  # pipes_out = deque(p[1] for p in pipes)
  # pipes_in.rotate(1)
  # pipes_out.rotate(-1)

  pipes = [Queue() for _ in range(numN)]
  pipes_in = deque(p for p in pipes)
  pipes_out = deque(p for p in pipes)
  pipes_in.rotate(1)
  pipes_out.rotate(-1)

  e = Event()

  print("Before launching algorithms")

  # procid = 0
  # pipein = None
  # pipeout = None
  # sync = None
  # data =input_list[i][4]
  # y = input_list[i][5]
  # X_test = input_list[i][1]
  # y_test = input_list[i][3]


  #procid,pipein, pipeout, sync,data,y, output_folder,REG = 'LR',obj_type = 'r2_ad_nfeat',nf =5,nb = 5, ngen = 1000, npop = 1000,NPROC = 1, save_rate = 25,gen_bit_cor=False, cor_feat = None
  processes = [Process(target=manga_algorithm, args=(i, ipipe, opipe,e, input_list[i][4],input_list[i][5],input_list[i][1],input_list[i][3],
                                                    features,best_k,output_folder, REG,obj_type,nf,nb,
                                                    ngen, npop,NPROC, save_rate,gen_bit_cor, input_list[i][6],mig_rate))
               for i, (ipipe, opipe) in enumerate(zip(pipes_in, pipes_out))]


  # start the processes
  for proc in processes:
    proc.start()
  # ...
  for proc in processes:
    proc.join()

  print("All processes ended" )

  for i in range(numN):
    numpy.savetxt(output_folder + "X_train_"+ str(i) +".txt",input_list[i][0],delimiter=",")
    numpy.savetxt(output_folder + "X_test_"+ str(i) +".txt",input_list[i][1],delimiter=",")
    numpy.savetxt(output_folder + "y_train_"+ str(i) +".txt",input_list[i][2],delimiter=",")
    numpy.savetxt(output_folder + "y_test_"+ str(i) +".txt",input_list[i][3],delimiter=",")


  print("saved training and test" )

  with open(output_folder + "parameters.txt", "a") as myfile:
    myfile.write("Percentage of test: %s\n" % test_perc)
    myfile.write("Number of generations: %s\n" % ngen)
    myfile.write("Number of individuals: %s\n" % npop)
    myfile.write("Type of regressor: %s\n" % REG)
    myfile.write("NPROC: %s\n" % NPROC)
    myfile.write("Objective type: %s\n" % obj_type)
    myfile.write("N.folds: %s\n" % nf)
    myfile.write("Repetitions: %s\n" % nb)
    myfile.write("N.bins: %s\n" % n_bins)
    myfile.write("Use correlation rank: %s\n" % gen_bit_cor)
    myfile.write("N.niches: %s\n" % numN)
    myfile.write("Save rate: %s\n" % save_rate)
    myfile.write("Individual to migrates: %s\n" % best_k)
    myfile.write("Migration rate: %s\n" % mig_rate)

  print("Saved parameters" )

  print(input_folder)
  print(REG)
  print(obj_type)


  ############

