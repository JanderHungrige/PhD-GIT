# -*- coding: utf-8 -*-
"""
Created on Thu Aug 10 22:10:10 2017

@author: 310122653
"""
#----------------------------------------------------------------------------------
# This file uses a brute force feature selection for the fist "Ncombos" combinations, "Ncombos" depending on speed,
# and follows with a greedy forward search, maybe coupled with a greedy backwards search
# This file is based on the "wrapper.py" used for paper two.
#----------------------------------------------------------------------------------
from IPython import get_ipython
get_ipython().magic('reset -sf')

from platform import python_version
print ('Python version: ');print( python_version())	


from Loading_5min_mat_files_cECG import AnnotMatrix_each_patient, FeatureMatrix_each_patient, Class_dict, features_dict, features_indx
from Classifier_routines import Classifier_routine_no_sampelWeight

import itertools
import numpy 
from matplotlib import *
from numpy import *
from pylab import *
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import StratifiedKFold
from sklearn.cross_validation import cross_val_score
from sklearn.cross_validation import KFold
from sklearn.metrics import accuracy_score 
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc
from sklearn import svm, cross_validation
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.linear_model import Perceptron
import sys #to add strings together



#from compute_class_weight import *   


import time
start_time = time.time()

description='_only_ASQSIS_perPatient_nosampleWeight'
consoleinuse='1';

classweight=0 # If classweights should be automatically determined and used for trainnig use: 1 else:0

preGridsearch=0# Search for the optimal c and gamma for all features to start with
finalGridsearch=1 # find the optimal c andgamma for the final subset to increase performance
plotting_grid=0 # plot the gridsearch
c=1.0
gamma=0.2 # both values are overwritten by the gridsearch if gridsearch is chosen

saving=0
#### SELECTING THE LABELS FOR SELECTED BABIES
label=array([1,2,3,4,5,6]) # 1=AS 2=QS 3=Wake 4=Care-taking 5=NA 6= transition
babies =array([0,1,2,3,4,5,6,7,8]) #0-9
lst = array([0,1,2,3,4,5,6,7,8,9,10,11,12,13,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29])
lst=[[x] for x in lst] # To make it look like combs_short, which was a list in a list.

combs=[]
bestAUCs=nan
collected_mean_auc=[]
FeatureSubSet=lst[:]

#### SCALE FEATURES
sc = StandardScaler()
for i in range(len(FeatureMatrix_each_patient)):
    sc.fit(FeatureMatrix_each_patient[i])
    FeatureMatrix_each_patient[i]=sc.transform(FeatureMatrix_each_patient[i])

#### Create Matrices for selected babies
AnnotMatrix_auswahl=[AnnotMatrix_each_patient[k] for k in babies]       # get the annotation values for selected babies
FeatureMatrix_auswahl=[FeatureMatrix_each_patient[k] for k in babies]   # get the feature values for selected babies

#### SELECT FEATURE SET
idx=[in1d(AnnotMatrix_each_patient[sb],label) for sb in babies]#.values()]     # which are the idices for AnnotMatrix_each_patient == label
idx=[nonzero(idx[sb])[0] for sb in range(len(babies))]#.values()]              # get the indices where True
y_each_patient=[val[idx[sb],:] for sb, val in enumerate(AnnotMatrix_auswahl) if sb in range(len(babies))] #get the values for y from idx and label


""" 
#GRID SEARCH FOR C AND GAMMA
"""
if preGridsearch:
    # For pregridsearch take all label
    # for final gridsearch use the labels you are investigating
    [c,gamma]=GridSearch(plotting_grid,array([1,2,3,4,5,6]),lst,AnnotMatrix_each_patient,FeatureMatrix_each_patient,babies)

""" 
#GREEDY FORWARD SEARCH
"""
# To reduce Bias there are two methods.
#1 delete on patient and do the classic leave one out greedy search. AFterwards compare the found subset results with the validation patient. Repeat that over all patients.
# Then take a look which features are represented in all the found subsets and only use them. They have the least bias

#2 again delet a patient . Then find with the remainding 8 sunsets which are search without looping through the patients but everytime for one leftout a whole subst (test on the left out but do not use him to train on). 
# Then out of these subsets find the best one with the validation patient. Loop that over each patient for validation. Find again the commoni features in all resulting best subsets. 



# 0: Search the for the first best feature to create Feature Sub Set 1 (FSS1)
# 1:take FSS1 and add 1 other feature to it.
# 2:Find best AUC/F1/Kappa, -> 3: new combo (FSS2)
# 4:delete the new found feature from lst
# repeate from 2

#lst= the lsit that contains all un-chosen features
#lst2= the list with the chosen features where one by one features are added from lst
for V in range(len(babies)):
    lstcpy=lst[:]
    selected_babies=list(delete(babies,babies[V-1]))# Babies to train and test on ; j-1 as index starts with 0
    selected_validation=selected_babies[V]# Babies to validate on 
    #### Create Matrices for selected babies
    AnnotMatrix_auswahl=[AnnotMatrix_each_patient[k] for k in selected_babies]       # get the annotation values for selected babies
    FeatureMatrix_auswahl=[FeatureMatrix_each_patient[k] for k in selected_babies]   # get the feature values for selected babies
    
    #### SELECT FEATURE SET
    idx=[in1d(AnnotMatrix_each_patient[sb],label) for sb in selected_babies]#.values()]     # which are the idices for AnnotMatrix_each_patient == label
    idx=[nonzero(idx[sb])[0] for sb in range(len(selected_babies))]#.values()]              # get the indices where True
    y_each_patient=[val[idx[sb],:] for sb, val in enumerate(AnnotMatrix_auswahl) if sb in range(len(selected_babies))] #get the values for y from idx and labe
    for i in range(len(lstcpy)):
        Xfeat=[val[:,lstcpy[i]] for sb, val in enumerate(FeatureMatrix_auswahl)] # selecting the features to run
        Xfeat=[val[idx[sb],:] for sb, val in enumerate(Xfeat)]   #selecting the datapoints in label
        #### Run wrapper on first round
        mean_auc=Classifier_routine_no_sampelWeight(Xfeat,y_each_patient,selected_babies,label,classweight,c,gamma)
        collected_mean_auc.append(mean_auc) # This collects the mean AUC of each itteration. As we want to know whcih combination is the best, we collect all mean AUCs and search for the maximum later
    

#    lst=[x for y in lst for x in y]    #Bringing lst back from list in list to list. to be able to work with
    # 2 :Finding the best AUC combination to continue with the greedy forward search    
    m=max(collected_mean_auc)
    bestAUCs=[m]
    idx_m=[i for i, j in enumerate(collected_mean_auc) if j == m]
    selectedF=lstcpy[idx_m[0]]# which features to train and test on
    print('-----')
    print('Subset: ', sep=' ', end='', flush=True); print(selectedF)
    print('max AUC: %.4f' %m)
    print('-----')

    sys.exit('Jan')

    
    for j in range(len(selectedF)):
        lstcpy.remove(selectedF[j]) # Remove the top feature from the list
    
    lst2=selectedF[:] # rewrite the selected features into a new list that we can experimentally add new features
    
    for l in range(len(lstcpy)):
        collected_mean_auc_new=[]
    #    print(lst)
        for k in range(len(lstcpy)): # going through all the leftover features and addign them one by one to the selected set -> test if better
            lst2+=lstcpy[k] # adding next feature for test to the lst
    
            # CREATING THE DATASET WITH x numbers of features WITH SPECIFIC LABELS FOR SELECTED BABIES  
            Xfeat=[val[:,lst2] for sb, val in enumerate(FeatureMatrix_auswahl)] # using the feature values for features in lst2 to run
            Xfeat=[val[idx[sb],:] for sb, val in enumerate(Xfeat)]   #choosing only the values which are taged with the used labels (AS=1 QS=2 ...)
        
            mean_auc=Classifier_routine_no_sampelWeight(Xfeat,y_each_patient,selected_babies,label,classweight,c,gamma)
            if size(mean_auc)>1:
                F1=mean_auc[:]
                mean_auc=mean(mean_auc)
            collected_mean_auc_new.append(mean_auc)
            
            lst2=selectedF[:]  # reset lst2 
            
           
        # Find the best AUC for that run  
        m_new=max(collected_mean_auc_new)        
        if m_new >= m: # if there is an increase in performance with added feature
            bestAUCs.append(m_new)
            idx_m=[i for i, j in enumerate(collected_mean_auc_new) if j == m_new] # find the feature indx which increased performance
            addition=lstcpy[idx_m[0]] # get the feature (=feature idx) to add into the selectedF
            selectedF+=[addition]# add new found feature
            lstcpy.remove(addition) # Remove the top feature from the list
            m=m_new #new maximum value to compare to
            if len(label)>2:
                print('F1|Kappa: %.4f' %m)
            else:
                print('AUC: %.4f' %m)
            print('Subset: '); print(selectedF)
            print('')
           
            
        else:
            maximal_m_so_far=m
            if len(label)>2:      
                print('maximal F1|Kappa: %.4f' %m)
            else:
                print('maximal AUC: %.4f' %m)         
            print('the perfomrance did not increase')
            print('Calculation stopped')
            break
              
        lst2=selectedF[:]    #set lst2 to the new subfeature set
        print('Round: %i of:%i' %(l+1,(len(lstcpy))))    
        
    #Validate with left out patient 
    # Run the classifier with the selected FEature subset in selecteF
    Xfeat=[val[:,selectedF] for sb, val in enumerate(FeatureMatrix_auswahl)] # using the feature values for features in lst2 to run
    Xfeat=[val[idx[sb],:] for sb, val in enumerate(Xfeat)]  
    perf=ValidatedPerformance=Validate_with_classifier(Xfeat,y_each_patient,selected_babies,selected_validation,label,classweight,C,gamma)
        
    Subsets[V]=selectedF # Saving best Subset and performance to be compared with other validated sets 
    Performance[V]=m
    ValidatedPerformance[V]=perf
    BiasInfluence[V]=Performance[V]-ValidatedPerformance[V]
        
        
        

if saving:    
    save('C:/Users/310122653/Dropbox/PHD/python/cECG/Results/selectedF' + description, selectedF)     
    save('C:/Users/310122653/Dropbox/PHD/python/cECG/Results/bestAUCs' + description, bestAUCs)
    save('C:/Users/310122653/Dropbox/PHD/python/cECG/Results/lst' + description, lst)
    import scipy.io as sio
    #sio.savemat('C:/Users/310122653/Dropbox/PHD/python/cECG/Results/', bestAUCs)        

import time
t=time.localtime()
zeit=time.asctime()
print('FINISHED Console ' + consoleinuse)
print("--- %i seconds ---" % (time.time() - start_time))
print("saved at: %s" %zeit)
print("Console 1 : "); print(description)
