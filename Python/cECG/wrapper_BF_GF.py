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

from platform import python_version
print ('Python version: ', sep=' ', end='', flush=True);print( python_version())	


from Loading_5min_mat_files_cECG import AnnotMatrix_each_patient, FeatureMatrix_each_patient, Class_dict, features_dict, features_indx
from Classifier_routines import Classifier_routine_no_sampelWeight

import itertools
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
consoleinuse='4';

classweight=0 # If classweights should be automatically determined and used for trainnig use: 1 else:0
Ncombos=1

#### SELECTING THE LABELS FOR SELECTED BABIES
label=array([1,2]) # 1=AS 2=QS 3=Wake 4=Care-taking 5=NA 6= transition
selected_babies =[0,1,2,3,4,5,6,7,8] #0-9
    
#### CREATE ALL POSSIBLE COMBINATIONS OUT OF 30 FEATURES. STOP AT Ncombos FEATURE SET(DUE TO SVM COMPUTATION TIME)
lst = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29]
combs=[]
bestAUCs=nan
#for i in range(1, len(lst)+1):
        
i=1    
els = [list(x) for x in itertools.combinations(lst, i)]
combs.extend(els)  
combs_short=[combs for combs in combs if len(combs) <= Ncombos] #due to computation time we stop with 5 Features per set.

      
#### SCALE FEATURES
sc = StandardScaler()
for i in range(len(FeatureMatrix_each_patient)):
    sc.fit(FeatureMatrix_each_patient[i])
    FeatureMatrix_each_patient[i]=sc.transform(FeatureMatrix_each_patient[i])

collected_mean_auc=[]
    
#### SELECT FEATURE SET
for Fc in range(len(combs_short)):
#    sys.stdout.write('.'); sys.stdout.flush();
#    print("Working on combination %i. of %i." %(Fc, len(combs_short)) );sys.stdout.flush();

    AnnotMatrix_auswahl=[AnnotMatrix_each_patient[k] for k in selected_babies]              # get the annotation values for selected babies
    FeatureMatrix_auswahl=[FeatureMatrix_each_patient[k] for k in selected_babies]          # get the feature values for selected babies
    
    idx=[in1d(AnnotMatrix_each_patient[sb],label) for sb in selected_babies]#.values()]     # which are the idices for AnnotMatrix_each_patient == label
    idx=[nonzero(idx[sb])[0] for sb in range(len(selected_babies))]#.values()]              # get the indices where True
    y_each_patient=[val[idx[sb],:] for sb, val in enumerate(AnnotMatrix_auswahl) if sb in range(len(selected_babies))] #get the values for y from idx and label

#### CREATING THE DATASET WITH x numbers of features WITH SPECIFIC LABELS FOR SELECTED BABIES  
    Xfeat=[val[:,combs_short[Fc]] for sb, val in enumerate(FeatureMatrix_auswahl)] # selecting the features to run
    Xfeat=[val[idx[sb],:] for sb, val in enumerate(Xfeat)]   #selecting the datapoints in label
    
    #sys.exit('Jan werth')
    mean_auc=Classifier_routine_no_sampelWeight(Xfeat,y_each_patient,selected_babies,label,classweight)
    collected_mean_auc.append(mean_auc) # This collects the mean AUC of each itteration. As we want to know whcih combination is the best, we collect all mean AUCs and search for the maximum later
    print('BF Round: %i of:%i' %(Fc+1,len(combs_short)))    

    
""" 
#GREEDY FORWARD SEARCH
"""
# 1:take best combo and add 1 other feature to it.
# 2:Find best AUC, -> 3: new combo
# 4:delete the new found feature from lst
#repeate

#lst= the lsit that contains all un-chosen features
#lst2= the list with the chosen features where one by one features are added from lst


# 2 :Finding the best AUC combination to continue with the greedy forward search    
m=max(collected_mean_auc)
bestAUCs=[m]
idx_m=[i for i, j in enumerate(collected_mean_auc) if j == m]
selectedF=combs_short[idx_m[0]]# which features to train and test on
print('max AUC: %.4f' %m)
print('Subset: '); print(selectedF)
print('')


for j in range(len(selectedF)):
    lst.remove(selectedF[j]) # Remove the top feature from the list

lst2=selectedF[:] # rewrite the selected features into a new list that we can experimentally add new features
for l in range(len(lst)):
    collected_mean_auc_new=[]
#    print(lst)
    for k in range(len(lst)): # going through all the leftover features and addign them one by one to the selected set -> test if better
        lst2+=[lst[k]] # adding next feature for test to the lst

        # CREATING THE DATASET WITH x numbers of features WITH SPECIFIC LABELS FOR SELECTED BABIES  
        Xfeat=[val[:,lst2] for sb, val in enumerate(FeatureMatrix_auswahl)] # using the feature values for features in lst2 to run
        Xfeat=[val[idx[sb],:] for sb, val in enumerate(Xfeat)]   #choosing only the values which are taged with the used labels (AS=1 QS=2 ...)
    
        mean_auc=Classifier_routine_no_sampelWeight(Xfeat,y_each_patient,selected_babies,label,classweight)
        if size(mean_auc)>1:
            F1=mean_auc[:]
            mean_auc=mean(mean_auc)
        collected_mean_auc_new.append(mean_auc)
        
        lst2=selectedF[:]  # reset lst2 
        
       
    # Find the best AUC for that run  
    m_new=max(collected_mean_auc_new)
    bestAUCs.append(m_new)
    if m_new >= m: # if there is an increase in performance with added feature
        idx_m=[i for i, j in enumerate(collected_mean_auc_new) if j == m_new] # find the feature indx which increased performance
        addition=lst[idx_m[0]] # get the feature (=feature idx) to add into the selectedF
        selectedF+=[addition]# add new found feature
        lst.remove(addition) # Remove the top feature from the list
        m=m_new #new maximum value to compare to
        print('max AUC: %.4f' %m)
        print('Subset: '); print(selectedF)
        print('')
       
        
    else:
        maximal_m_so_far=m
        print('maximal AUC: %.4f' %m)
        print('the perfomrance did not increase')
        print('Calculation stopped')
        break
          
    lst2=selectedF[:]    #set lst2 to the new subfeature set
    print('Round: %i of:%i' %(l+1,(len(lst))))    
      
save('C:/Users/310122653/Dropbox/PHD/python/cECG/Results/selectedF' + description, selectedF)     
save('C:/Users/310122653/Dropbox/PHD/python/cECG/Results/bestAUCs' + description, bestAUCs)
save('C:/Users/310122653/Dropbox/PHD/python/cECG/Results/combs_short' + description, combs_short)
import scipy.io as sio
#sio.savemat('C:/Users/310122653/Dropbox/PHD/python/cECG/Results/', bestAUCs)        

import time
t=time.localtime()
zeit=time.asctime()
print('FINISHED Console ' + consoleinuse)
print("--- %i seconds ---" % (time.time() - start_time))
print("saved at: %s" %zeit)
print("Console 1 : "); print(description)
