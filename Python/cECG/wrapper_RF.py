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
from platform import python_version
print ('Python version: ', sep=' ', end='', flush=True);print( python_version())	


from Loading_5min_mat_files_cECG import \
babies, AnnotMatrix_each_patient, FeatureMatrix_each_patient_all, Class_dict, features_dict, features_indx, \
FeatureMatrix_each_patient_fromSession, lst
from Classifier_routines import Classifier_random_forest
from GridSearch import *

import itertools
from matplotlib import *
from numpy import *
from pylab import *
from sklearn.preprocessing import MinMaxScaler
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
Klassifier=['RF','ERF','TR','GB']
SampMeth=['NONE','SMOTE','ADASYN']
Whichmix=['perSession', 'all']
"""
**************************************************************************
Loading_5min_mat_files_cECG.py :

       CHANGE THE DATASET IF USING ECG OR cECG

       CHOOSE WHICH BAIES ARE USED
       
       CHOOSE IF FEATURES ARE MERGED OR REPLACED
       
       Choose which FEatures to use

**************************************************************************
"""
#_Labels_ECG_Featurelist_Scoring_classweigt_C_gamma

description='_123456_cECG_lst_micro_'
consoleinuse='4'

savepath='/home/310122653/Pyhton_Folder/cECG/Results/'
"""
**************************************************************************
CHANGE THE DATASET IN Loading_5min_mat_files_cECG.py IF USING ECG OR cECG
**************************************************************************
"""

# SELECTING THE LABELS FOR SELECTED BABIES
label=array([1,2,4]) # 1=AS 2=QS 3=Wake 4=Care-taking 5=NA 6= transition
#--------------------------


# SET INSTRUCTIONS
classweight=1 # If classweights should be automatically ('balanced') determined and used for trainnig use: 0; IF they should be calculated by own function use 1
saving=0
#--------------------------

Used_classifier='RF' #RF=random forest ; ERF= extreme random forest; TR= Decission tree; GB= Gradient boosting
drawing=0 # draw a the tree structure
#--------------------------

plotting=1 # plot annotations over time per patient
compare=0 # additional plot
#---------------------------

#For up and downsampling of data
SamplingMeth='NONE'  # 'NONE' 'SMOTE'  or 'ADASYN'
ChoosenKind=0   # 0-3['regular','borderline1','borderline2','svm'] only when using SMOTE

#---------------------------
probability_threshold=1 # 1 to use different probabilities tan 0.5 to decide on the class. At the moment it is >=0.2 for any other calss then AS
WhichMix='perSession' #perSession or all  # determine how the data was scaled. PEr session or just per patient

# CHOOSING WHICH FEATURE MATRIX IS USED
if WhichMix=='perSession':
       FeatureMatrix_each_patient=FeatureMatrix_each_patient_fromSession       
elif WhichMix=='all':
       FeatureMatrix_each_patient=FeatureMatrix_each_patient_all

"""
CHECKUP
"""
t_a=list()
classpredictions=list()
Performance=list()
ValidatedPerformance_macro=list()
ValidatedPerformance_K=list()
ValidatedPerformance_micro=list()
ValidatedPerformance_weigth=list()  
ValidatedPerformance_all=zeros(shape=(len(babies),len(label)))
ValidatedFimportance=zeros(shape=(len(babies),len(FeatureMatrix_each_patient[0][1])))
Validatedscoring=list() 

####Cheking for miss spelling
if Used_classifier not in Klassifier:
       sys.exit('Misspelling in Used_classifier')    
if SamplingMeth not in SampMeth:
       sys.exit('Misspelling in SamplingMeth')   
if WhichMix not in Whichmix:
       sys.exit('Misspelling in WhichMix')         


       
"""
START
"""       
# scaling is now done directly during loading   


for V in range(len(babies)):
    print('**************************')
    print('Validating on patient: %i' %(V+1) ) 
    
    selected_babies=list(delete(babies,babies[V]))# Babies to train and test on ; j-1 as index starts with 0
    selected_test=babies[V]# Babies to validate on 
    #### Create Matrices for selected babies
    
    AnnotMatrix_auswahl=[AnnotMatrix_each_patient[k] for k in selected_babies]              # get the annotation values for selected babies
    FeatureMatrix_auswahl=[FeatureMatrix_each_patient[k] for k in selected_babies]          # get the feature values for selected babies
    idx=[in1d(AnnotMatrix_each_patient[sb],label) for sb in selected_babies]#.values()]     # which are the idices for AnnotMatrix_each_patient == label
    idx=[nonzero(idx[sb])[0] for sb in range(len(selected_babies))]#.values()]              # get the indices where True
    Xfeat=[val[idx[sb],:] for sb, val in enumerate(FeatureMatrix_auswahl)]   #selecting the datapoints in label
    y_each_patient=[val[idx[sb],:] for sb, val in enumerate(AnnotMatrix_auswahl) if sb in range(len(selected_babies))] #get the values for y from idx and label    
#    Xfeat=[val[:,lst] for sb, val in enumerate(FeatureMatrix_auswahl)] # selecting the features to run
#    Xfeat=[val[idx[sb],:] for sb, val in enumerate(Xfeat)]   #selecting the datapoints in label
 
    #creating another Set where all PAtients are included. In the Random Forest function one is selected for training. otherwise dimension missmatch   
    AnnotMatrix_auswahl_test=[AnnotMatrix_each_patient[k] for k in babies]              # get the annotation values for selected babies
    FeatureMatrix_auswahl_test=[FeatureMatrix_each_patient[k] for k in babies]
    idx_test=[in1d(AnnotMatrix_each_patient[sb],label) for sb in babies]#.values()]     # which are the idices for AnnotMatrix_each_patient == label
    idx_test=[nonzero(idx_test[sb])[0] for sb in range(len(babies))]#.values()]              # get the indices where True
    Xfeat_test=[val[idx_test[sb],:] for sb, val in enumerate(FeatureMatrix_auswahl_test)]  
    y_each_patient_test=[val[idx_test[sb],:] for sb, val in enumerate(AnnotMatrix_auswahl_test) if sb in range(len(babies))] #get the values for y from idx and label
#    Xfeat_test=[val[:,lst] for sb, val in enumerate(FeatureMatrix_auswahl_test)] # using the feature values for features in lst2 to run
#    Xfeat_test=[val[idx_test[sb],:] for sb, val in enumerate(Xfeat_test)]  
     

    #Validate with left out patient 
    # Run the classifier with the selected FEature subset in selecteF
    resultsF1_macro,resultsK,resultsF1_micro,resultsF1_weight,resultsF1_all,Fimportances,scoring,prediction \
    =Classifier_random_forest(Xfeat_test, Xfeat,y_each_patient_test, y_each_patient, selected_babies, \
                              selected_test, label,classweight, Used_classifier, drawing, lst,\
                              ChoosenKind,SamplingMeth,probability_threshold)

#    =Classifier_random_forest(Xfeat,y_each_patient,selected_babies,label,classweight)       
#    sys.exit('Jan werth 206')
    classpredictions.append(prediction)
    ValidatedFimportance[V]=Fimportances

    ValidatedPerformance_macro.append(resultsF1_macro)
    ValidatedPerformance_K.append(resultsK)
    ValidatedPerformance_micro.append(resultsF1_micro)
    ValidatedPerformance_weigth.append(resultsF1_weight)    
    ValidatedPerformance_all[V]=resultsF1_all
    Validatedscoring.append(scoring)

    if plotting:
           t_a.append(np.linspace(0,len(y_each_patient_test[V])*30/60,len(y_each_patient_test[V])))
           if not compare:
                  plt.figure(V) 
                  plt.plot(t_a[V],y_each_patient_test[V])
                  plt.plot(t_a[V],classpredictions[V]+0.04)
                  plt.title([V])    
           if compare:
                  plt.figure(V) 
                  plt.plot(t_a[V],classpredictions[V]+0.07)
                  plt.title([V])
            
"""
ENDING stuff
"""

ValidatedPerformance_K.append(mean(ValidatedPerformance_K)) 
AAAValidatedPerformance_all=array(mean(ValidatedPerformance_all,0))


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
