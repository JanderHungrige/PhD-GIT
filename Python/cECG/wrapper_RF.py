# -*- coding: utf-8 -*-
"""
Created on Thu Oct 26 13:11:02 2017

@author: 310122653
"""

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
from Classifier_routines import Classifier_random_forest
from GridSearch import *

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
import pdb # use pdb.set_trace() as breakpoint


#from compute_class_weight import *   
import time
start_time = time.time()



"""
**************************************************************************
CHANGE THE DATASET IN Loading_5min_mat_files_cECG.py IF USING ECG OR cECG
**************************************************************************
"""
#_Labels_ECG_Featurelist_Scoring_classweigt_C_gamma

description='_123456_cECG_lst_micro_'
consoleinuse='4'

savepath='/home/310122653/Pyhton_Folder/cECG/Results/'

#### SELECTING THE LABELS FOR SELECTED BABIES
label=array([1,2,4,5,6]) # 1=AS 2=QS 3=Wake 4=Care-taking 5=NA 6= transition
babies =[0,1,2,3,4,5,6,7,8] #0-8
    
#### CREATE ALL POSSIBLE COMBINATIONS OUT OF 30 FEATURES. STOP AT Ncombos FEATURE SET(DUE TO SVM COMPUTATION TIME)
#lst = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29]
lst_old=[3,4,5,6,7,8,9,10,11,13,15,16,17,18,19,20,21,22,23,24,25,26] # From first paper to compare with new features
lst=lst_old
"""
**************************************************************************
CHANGE THE DATASET IN Loading_5min_mat_files_cECG.py IF USING ECG OR cECG
**************************************************************************
"""

classweight=1 # If classweights should be automatically determined and used for trainnig use: 1 else:0
saving=0
Ncombos=1
preGridsearch=0
Used_classifier='TR' #RF=random forest ; ERF= extreme random forest; TR= Decission tree
drawing=1 # draw a the tree structure

Performance=list()
ValidatedPerformance_macro=list()
ValidatedPerformance_K=list()
ValidatedPerformance_micro=list()
ValidatedPerformance_weigth=list()  
ValidatedPerformance_all=zeros(shape=(len(babies),len(label)))
ValidatedFimportance=zeros(shape=(len(babies),len(lst)))
Validatedscoring=list() 
      
#### SCALE FEATURES
sc = StandardScaler()
for i in range(len(FeatureMatrix_each_patient)):
    sc.fit(FeatureMatrix_each_patient[i])
    FeatureMatrix_each_patient[i]=sc.transform(FeatureMatrix_each_patient[i])


for V in range(len(babies)):
    print('**************************')
    print('Validating on patient: %i' %V)  
    
    selected_babies=list(delete(babies,babies[V-1]))# Babies to train and test on ; j-1 as index starts with 0
    selected_test=babies[V-1]# Babies to validate on 
    #### Create Matrices for selected babies
    
#    AnnotMatrix_auswahl=[AnnotMatrix_each_patient[k] for k in selected_babies]              # get the annotation values for selected babies
#    FeatureMatrix_auswahl=[FeatureMatrix_each_patient[k] for k in selected_babies]          # get the feature values for selected babies
#    idx=[in1d(AnnotMatrix_each_patient[sb],label) for sb in selected_babies]#.values()]     # which are the idices for AnnotMatrix_each_patient == label
#    idx=[nonzero(idx[sb])[0] for sb in range(len(selected_babies))]#.values()]              # get the indices where True
#    y_each_patient=[val[idx[sb],:] for sb, val in enumerate(AnnotMatrix_auswahl) if sb in range(len(selected_babies))] #get the values for y from idx and label    
##### CREATING THE DATASET WITH x numbers of features WITH SPECIFIC LABELS FOR SELECTED BABIES  
#    Xfeat=[val[:,lst] for sb, val in enumerate(FeatureMatrix_auswahl)] # selecting the features to run
#    Xfeat=[val[idx[sb],:] for sb, val in enumerate(Xfeat)]   #selecting the datapoints in label
    
    AnnotMatrix_auswahl_valid=[AnnotMatrix_each_patient[k] for k in babies]              # get the annotation values for selected babies
    FeatureMatrix_auswahl_valid=[FeatureMatrix_each_patient[k] for k in babies]
    idx_valid=[in1d(AnnotMatrix_each_patient[sb],label) for sb in babies]#.values()]     # which are the idices for AnnotMatrix_each_patient == label
    idx_valid=[nonzero(idx_valid[sb])[0] for sb in range(len(babies))]#.values()]              # get the indices where True
    y_each_patient_valid=[val[idx_valid[sb],:] for sb, val in enumerate(AnnotMatrix_auswahl_valid) if sb in range(len(babies))] #get the values for y from idx and label
    Xfeat_valid=[val[:,lst] for sb, val in enumerate(FeatureMatrix_auswahl_valid)] # using the feature values for features in lst2 to run
    Xfeat_valid=[val[idx_valid[sb],:] for sb, val in enumerate(Xfeat_valid)]  
    
    """
    VALIDATION
    """        
    #Validate with left out patient 
    # Run the classifier with the selected FEature subset in selecteF
    resultsF1_macro,resultsK,resultsF1_micro,resultsF1_weight,resultsF1_all,Fimportances,scoring \
    =Classifier_random_forest(Xfeat_valid,y_each_patient_valid,selected_babies,selected_test,label,classweight,Used_classifier,drawing)

#    =Classifier_random_forest(Xfeat,y_each_patient,selected_babies,label,classweight)       
#    sys.exit('Jan werth 206')
    ValidatedFimportance[V]=Fimportances

    ValidatedPerformance_macro.append(resultsF1_macro)
    ValidatedPerformance_K.append(resultsK)
    ValidatedPerformance_micro.append(resultsF1_micro)
    ValidatedPerformance_weigth.append(resultsF1_weight)    
    ValidatedPerformance_all[V]=resultsF1_all
    Validatedscoring.append(scoring)
    
"""
ENDING stuff
"""
        
if saving:      
    save(savepath + 'ValidatedFimportance' + description, ValidatedFimportance)     
    
    save(savepath + 'ValidatedPerformance_macro' + description, ValidatedPerformance_macro)     
    save(savepath + 'ValidatedPerformance_K' + description, ValidatedPerformance_K)     
    save(savepath + 'ValidatedPerformance_micro' + description, ValidatedPerformance_micro)
    save(savepath + 'ValidatedPerformance_weigth' + description, ValidatedPerformance_weigth)     
    save(savepath + 'ValidatedPerformance_all' + description, ValidatedPerformance_all)       

import time
t=time.localtime()
zeit=time.asctime()
Minuten=(time.time() - start_time)/60
Stunden=(time.time() - start_time)/3600
print('FINISHED Console ' + consoleinuse)
print("--- %i seconds ---" % (time.time() - start_time))
print("--- %i min ---" % Minuten)
print("--- %i h ---" % Stunden)

if saving:
    print("saved at: %s" %zeit)
print("Console 1 : "); print(description)
