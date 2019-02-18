#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 22 11:51:58 2018

@author: angela serra
"""

#%%
from time import time

#%% Import my functions
from objective_functions import evalRegression
from utility_functions import generateBit,weightedChoice,compileProbs
from utility_functions import migPipe
from utility_functions import jaccard2,percentage
from utility_functions import getSimScore2,analyse_population
from utility_functions import percentage,print_logbook_file_console,list_duplicates_of
from utility_functions import compile_metrics,check_objective_params#,compile_metrics_4objs,compile_metrics_1objs,compile_metrics_4objs_plus_correlation,compile_metrics_3objs,compile_metrics_2objs,compile_metrics_5objs
from utility_functions import fill_loogbook_r2,fill_loogbook_rmse,fill_loogbook_r2_ad_nfeat,fill_loogbook_rmse_ad_nfeat,fill_loogbook_r2_nfeat,fill_loogbook_rmse_nfeat,fill_loogbook_ad_nfeat,fill_loogbook_r2_ad_rmse_nfeat
import sys
import cPickle as pickle

from deap import base
from deap import creator
from deap import tools
from deap import algorithms
import multiprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import random, numpy

#%%
# This function execute the multi objective genetic algorithm for a single niche

# Input:
    # procid is the id of the niche or the process id
    # pipein
    # pipeout
    # sync
    # data is the matrix with samples on rows and features on columns
    # y is the response vector
    # best_k number of samples to be exchanged between niches
    # REG is a string specifyig the regression model. 
        # NB at the moment the implemented methods are the followings:
            # RF: random forest
            # SVM: svm with RBF kernel
            # LR: lienar regression (default value)
    # nf is the number of folds used to evaluate each individual
    # nb is the number of bootstrap repetition of cross-validation
    # ngen is the number of generations
    # npop is the number of individuals in the population
    # NPROC is the number of parallel processes
    # save_rate: number of individuals in each batch to save in the logbook
    # obj_type: is the typo of objective function we want to use, for example r2_ad_nfeat compute rsquqre, applicability domain and minimize the number of features
    # gen_bit_cor: do the bit in the chromosome must be generate based on the feature correlation with the resposnse variable?
    # cor_feat = vector of correlation between features and response variable; default is None
def manga_algorithm(procid,pipein, pipeout, sync,data,y,X_test, y_test,features, best_k,output_folder,REG = 'LR',obj_type = 'r2_ad_nfeat',nf =5,nb = 5, ngen = 1000, npop = 1000,NPROC = 1, save_rate = 25,gen_bit_cor=False, cor_feat = None,mig_rate = 50):
        
    print("proc %d Structure initializers..." % procid)
    # The create() function takes at least two arguments, a name for the newly created class and a base class. 
    # Any subsequent argument becomes an attribute of the class.
    # The fitness class will be named FitnessMultiMax, it inheriths from the base.Fitness class 
    # it has the attribute weights that identify if the objectives functions has to be maximized (positive weights) or minimized (negative weights)
   
    weigth_type,logheader,min_max_func,index_metrics,fill_function = check_objective_params(obj_type)
    compile_metric_function = compile_metrics

    creator.create("FitnessMultiMax", base.Fitness, weights=weigth_type)
    # The individuals are a list, thet have the attribute fitness. The fitness function is called on each individual
    creator.create("Individual", list, fitness=creator.FitnessMultiMax)
    
    toolbox = base.Toolbox()
    toolbox.register("get_add_pb", compileProbs)

    
    # Init list of pops
    POPs = list() #the list that contains the population
    FITs = list()
    #best_k = int(percentage(25, npop)) #select 25% of individuals for the migration
    #Individual contains a binary vector indicating the selected features.
    weights = toolbox.get_add_pb(cor_feat)

    if gen_bit_cor==False:
      # This is the function that generate the binar chromosomes, it calls the function generateBit 
      # that creates random vectors of 0 and 1 with a certain probability for the zeros and a certain probability for the ones.
      toolbox.register("genbit", generateBit)
      toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.genbit, n = data.shape[1])
    else:
      toolbox.register("attr_feat", weightedChoice,range(data.shape[1]), weights)
      toolbox.register("individual", 
                 tools.initRepeat, 
                 creator.Individual, 
                 toolbox.attr_feat, 
                 data.shape[1])

    toolbox.register("population", tools.initRepeat, list, toolbox.individual, n = npop)
    toolbox.register("evaluate", evalRegression, objective_function = obj_type,data = data, y=y, nf = nf, nb = nb,REG = REG)  
  
    # Single point crossover function
    toolbox.register("mate", tools.cxUniform, indpb=0.1)
    # Mutation function. indpb is the indipendent probability to change the value of a chromosome
    toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
    # The individuals are selected by using the NSGA2 algorithm
    # The implemented NSGA2 algorithm is the one reported in here: 
    # A fast elitist non-dominated sorting genetic algorithm for multi-objective optimization: NSGA-II
    # https://pdfs.semanticscholar.org/59a3/fea1f38c5dd661cc5bfec50add2c2sf881454.pdf   
    toolbox.register("select", tools.selNSGA2) 
    
    toolbox.register("get_ssc", getSimScore2, feat_names = data[2])
    #pool = multiprocessing.Pool(processes=NPROC)
    #toolbox.register("map", pool.map)
    
    population = toolbox.population()
    # This function migrate x% of the population from the current solution to the 
    toolbox.register("migrate", migPipe, k=best_k, pipein=pipein, pipeout=pipeout,
                     selection=tools.selBest, replacement=random.sample)

    print("Population length %s" % len(population))
    #compute the fitness for the population
    
    print("Compute fitness for the population...")
    
    fits = list(toolbox.map(toolbox.evaluate, population))
    
    for fit, ind in zip(fits, population):
        ind.fitness.values = fit
    
    # This is just to assign the crowding distance to the individuals no actual selection is done
    population = toolbox.select(population, len(population))
    
    compiled_metrics = compile_metric_function(fits,population)
    
    print("Init the logbook...")
    logbook = tools.Logbook()
    logbook.header = logheader
    
    # Update the logbook
    logbook = fill_function(logbook, procid,0,compiled_metrics,ssc=0,time=0, min_max_param=0)
    if procid == 0:
        # Synchronization needed to log header on top and only once
        print(logbook.stream)
        sync.set()
    else:
        logbook.log_header = False  # Never output the header
        sync.wait()
        print(logbook.stream)

    # Begin the generational process
    for gen in range(1, ngen + 1):
        t1 = time()
         
        # Part of an evolutionary algorithm applying only the variation part (crossover, mutation or reproduction). It takes the following parameters:
            # population - A list of individuals to vary.
            # toolbox - A Toolbox that contains the evolution operators.
            # lambda_ - The number of children to produce
            # cxpb - The probability of mating two individuals.
            # mutpb - The probability of mutating an individual.
            
        #offspring = algorithms.varOr(population, toolbox, lambda_=100, cxpb=0.5, mutpb=0.01)
        offspring = algorithms.varAnd(population, toolbox, cxpb=0.5, mutpb=0.01)

        fits = list(toolbox.map(toolbox.evaluate, offspring))
        for fit, ind in zip(fits, offspring):
            ind.fitness.values = fit
            
        population = toolbox.select(offspring + population, k = len(population)) #selezionare tutta la popolazione
        compiled_metrics = compile_metric_function(fits,population)
        ssc = toolbox.get_ssc(population)

        # Compile time
        t2 = time()
        dt = t2-t1
#        if(len(str(compiled_metrics[index_metrics]))==1):
        if(len(str(compiled_metrics[len(compiled_metrics)-1]))==1):
            logbook = fill_function(logbook, procid,gen,compiled_metrics,ssc=ssc,time=dt)
        else:
            logbook = fill_function(logbook, procid,gen,compiled_metrics,ssc=ssc,time=dt, min_max_param = min_max_func(compiled_metrics[len(compiled_metrics)-1]))

        print(logbook.stream)
        if gen % mig_rate == 0 and gen > 0:
                # This function migrate x% of the population from the current solution to the 
            toolbox.migrate(population)
            

        if gen % save_rate == 0 and gen > 0:
            POPs.append(list(population))
            FITs.append(list(fits))
            
    print("proc n. %s ended generation " % procid )

    with open(output_folder + "logbook_procid_%s.txt" % procid, "w") as myfile:
      myfile.write("%s" % logbook)
    myfile.close()
    
    print(output_folder)
    
    print("proc n. %s start analysing population " % procid )
    analyse_population(procid,population,output_folder,data, X_test, y, y_test,features,obj_type,REG,k=10)

    f = open(output_folder + "population_procid_%d"% procid, 'wb')
    pickle.dump(population, f)
    f.close()
    
    print("proc n. %s dumped population " % procid )

    f = open(output_folder + "POPs_procid_%d"% procid, 'wb')
    pickle.dump(POPs, f)
    f.close()
  
    f = open(output_folder + "FITs_procid_%d"% procid, 'wb')
    pickle.dump(FITs, f)
    f.close()  
  
    print("proc n. %s dumped POPS " % procid )
    
    f = open(output_folder + "loogbook_procid_%d"% procid, 'wb')
    pickle.dump(logbook, f)
    f.close()
    print("proc n. %s dumped loogbook " % procid )

    print("proc n. %s finish " % procid )

    return (population,logbook,POPs)

