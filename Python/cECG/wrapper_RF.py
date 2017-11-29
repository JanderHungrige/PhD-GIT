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


#from Loading_5min_mat_files_cECG import \
#babies, AnnotMatrix_each_patient, FeatureMatrix_each_patient_all, Class_dict, features_dict, features_indx, \
#FeatureMatrix_each_patient_fromSession, lst
from Classifier_routines import Classifier_random_forest
from GridSearch import *
from Loading_5min_mat_files_cECG import Loading_data_all,Loading_data_perSession,Feature_names,Loading_Annotations
from LOOCV import leave_one_out_cross_validation

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
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import f1_score
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


#_Labels_ECG_Featurelist_Scoring_classweigt_C_gamma

description='_123456_cECG_lst_micro_'
consoleinuse='4'

savepath='/home/310122653/Pyhton_Folder/cECG/Results/'
"""
**************************************************************************
Loading data declaration
**************************************************************************
"""

dataset='ECG'  # Either ECG or cECG and later maybe MMC or InnerSense
#***************
selectedbabies =[0,1,3,5,6,7] #0-8 ('4','5','6','7','9','10','11','12','13')
#---------------------------
# Feature list
lst = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29]
#lst_old=[3,4,5,6,7,8,9,10,11,14,15,16,17,18,19,20,21,22,23,24,25,26] # From first paper to compare with new features
#lst=lst_old
#---------------------------
ux=0 # if using this on Linux cluster use 1 to change adresses
scaling='Z' # Scaling Z or MM 
#---------------------------
LoosingAnnot5= 0# exchange state 5 if inbetween another state with this state (also only if length <= x)
LoosingAnnot6=0  #Exchange state 6 with the following or previouse state (depending on direction)
LoosingAnnot6_2=0 # as above, but chooses always 2 when 6 was lead into with 1
direction6=0 # if State 6 should be replaced with the state before, use =1; odtherwise with after, use =0. Annotators used before.
Smoothing_short=0 # # short part of any annotation are smoothed out. 
Pack4=0 # State 4 is often split in multible short parts. Merge them together as thebaby does not calm downin 1 min
#---------------------------
Movingwindow=10 # WIndow size for moving average
preaveraging=0
postaveraging=1
exceptNOF=1 #Which Number of Features (NOF) should be used with moving average?  all =oth tzero; only some or all except some defined in FEAT
onlyNOF=0 # [0,1,2,27,28,29]
FEAT=[0,1,2]
#FEAT=[1,2,27,28] # FRO CT
#----------------------------
PolyTrans=0#use polinominal transformation on the Features specified in FEATp
ExpFactor=2# which degree of polinomonal (2)
exceptNOpF= 0#Which Number of Features (NOpF) should be used with polynominal fit?  all =0; only some or all except some defined in FEATp
onlyNOpF=1 # [0,1,2,27,28,29]
#FEATp=[1,2,27,28] # FRO CT
FEATp=[0,3,4,5]

"""
Declaring variables for the wrapper
"""
#=---------------------------
# SELECTING THE LABELS FOR SELECTED BABIES
label=[1,2,4] # 1=AS 2=QS 3=Wake 4=Care-taking 5=NA 6= transition
#--------------------------
# SET INSTRUCTIONS
classweight=1 # If classweights should be automatically ('balanced') determined and used for trainnig use: 0; IF they should be calculated by own function use 1
saving=0
#--------------------------
Used_classifier='RF' #RF=random forest ; ERF= extreme random forest; TR= Decission tree; GB= Gradient boosting
drawing=0 # draw a the tree structure
#--------------------------
plotting=0 # plot annotations over time per patient
compare=0 # additional plot
#---------------------------
#For up and downsampling of data
SamplingMeth='NONE'  # 'NONE' 'SMOTE'  or 'ADASYN'
ChoosenKind=0   # 0-3['regular','borderline1','borderline2','svm'] only when using SMOTE
#---------------------------
probability_threshold=1 # 1 to use different probabilities tan 0.5 to decide on the class. At the moment it is >=0.2 for any other calss then AS
WhichMix='perSession' #perSession or all  # determine how the data was scaled. PEr session or just per patient
#--------------------
N=100 # Estimators for the trees
crit='gini' #gini or entropy method for trees 
msl=5  #min_sample_leafe
deciding_performance_measure='F1_second_label' #Kappa , F1_second_label, F1_third_label, F1_fourth_label
"""
CHECKUP
"""
####Cheking for miss spelling
if Used_classifier not in Klassifier:
       sys.exit('Misspelling in Used_classifier')    
if SamplingMeth not in SampMeth:
       sys.exit('Misspelling in SamplingMeth')   
if WhichMix not in Whichmix:
       sys.exit('Misspelling in WhichMix')         
  
"""
Loading Data
"""
Class_dict, features_dict, features_indx=Feature_names()

# CHOOSING WHICH FEATURE MATRIX IS USED
def loadingdata(whichMix):
       if WhichMix=='perSession':            
              babies, AnnotMatrix_each_patient,FeatureMatrix\
              =Loading_data_perSession(dataset, selectedbabies, lst,ux, scaling,\
                            LoosingAnnot5, LoosingAnnot6, LoosingAnnot6_2, direction6, plotting, Smoothing_short, Pack4,\
                            Movingwindow, preaveraging, postaveraging, exceptNOF, onlyNOF, FEAT,\
                            PolyTrans, ExpFactor, exceptNOpF, onlyNOpF, FEATp)       
              
       elif WhichMix=='all':              
              babies, AnnotMatrix_each_patient, FeatureMatrix\
              =Loading_data_all(dataset,selectedbabies,lst,ux,scaling,\
                            LoosingAnnot5,LoosingAnnot6,LoosingAnnot6_2,direction6,plotting,Smoothing_short,Pack4,\
                            Movingwindow,preaveraging,postaveraging,exceptNOF,onlyNOF,FEAT,\
                            PolyTrans,ExpFactor,exceptNOpF,onlyNOpF,FEATp)
              
       return babies,AnnotMatrix_each_patient,FeatureMatrix



babies,AnnotMatrix_each_patient,FeatureMatrix_each_patient= loadingdata(WhichMix)                  

"""
LOOCV ************************************************************************
"""       
y_each_patient,\
RES_classpredictions_QS,\
RES_Fimportance_QS,\
RES_F1_macro_QS,\
RES_KAPPA_QS,\
RES_F1_micro_QS,\
RES_F1_weigth_QS,\
RES_F1_all_QS,\
RES_scoring_QS,\
RES_Kappa_Performance_QS,\
RES_F1_all_QS\
=leave_one_out_cross_validation(babies,AnnotMatrix_each_patient,FeatureMatrix_each_patient,\
         label,classweight, Used_classifier, drawing, lst,ChoosenKind,SamplingMeth,probability_threshold,plotting,compare,saving,\
         N,crit,msl,deciding_performance_measure)

RES_F1_all_QS_mean=array(mean(RES_F1_all_QS,0))

"""
RUN 2  CT
"""

LoosingAnnot5= 0# exchange state 5 if inbetween another state with this state (also only if length <= x)
LoosingAnnot6=0  #Exchange state 6 with the following or previouse state (depending on direction)
LoosingAnnot6_2=0 # as above, but chooses always 2 when 6 was lead into with 1
direction6=0 # if State 6 should be replaced with the state before, use =1; odtherwise with after, use =0. Annotators used before.
Smoothing_short=0 # # short part of any annotation are smoothed out. 
Pack4=0 # State 4 is often split in multible short parts. Merge them together as thebaby does not calm downin 1 min
#---------------------------

Movingwindow=50 # WIndow size for moving average
preaveraging=0
postaveraging=1
exceptNOF=1 #Which Number of Features (NOF) should be used with moving average?  all =oth tzero; only some or all except some defined in FEAT
onlyNOF=0 # [0,1,2,27,28,29]
#FEAT=[0,1,2]
FEAT=[1,2,27,28] # FRO CT
#----------------------------

PolyTrans=1#use polinominal transformation on the Features specified in FEATp
ExpFactor=2# which degree of polinomonal (2)
exceptNOpF= 0#Which Number of Features (NOpF) should be used with polynominal fit?  all =0; only some or all except some defined in FEATp
onlyNOpF=1 # [0,1,2,27,28,29]
#FEATp=[1,2,27,28] # FRO CT
FEATp=[0,3,4,5]
#--------------------
N=50 # Estimators for the trees
crit='entropy' #gini or entropy
msl=3  #min_sample_leafe
deciding_performance_measure='F1_third_label' #Kappa , F1_second_label, F1_third_label, F1_fourth_label



babies,AnnotMatrix_each_patient,FeatureMatrix_each_patient= loadingdata(WhichMix)                  

"""
LOOCV
""" 
y_each_patient,\
RES_classpredictions_CT,\
RES_Fimportance_CT,\
RES_F1_macro_CT,\
RES_KAPPA_CT,\
RES_F1_micro_CT,\
RES_F1_weigth_CT,\
RES_F1_all_CT,\
RES_scoring_CT,\
RES_Kappa_Performance_CT,\
RES_F1_all_CT\
=leave_one_out_cross_validation(babies,AnnotMatrix_each_patient,FeatureMatrix_each_patient,\
         label,classweight, Used_classifier, drawing, lst, ChoosenKind,SamplingMeth,probability_threshold,plotting,compare,saving,\
         N,crit,msl,deciding_performance_measure)

RES_F1_all_CT_mean=array(mean(RES_F1_all_CT,0))

#Optimize prediction by taking predictions for specific classes from differnt classifiers
#The base predicitions are the one from QS classifier. Over that each 4(care taking) is decided/changed by the classifer results for CT
classpredictions=RES_classpredictions_QS[:]
for o in range(len(RES_classpredictions_CT)):
       for p in range(len(RES_classpredictions_CT[o])):  
#              ind=classpredictionsCT[o]==4
              if RES_classpredictions_CT[o][p]==4: 
                     classpredictions[o][p]=4
              elif classpredictions[o][p]==4 and RES_classpredictions_CT[o][p]!=4: # CT determines if 4 or not
                     classpredictions[o][p]=RES_classpredictions_CT[o][p]
              elif classpredictions[o][p]!=4 and RES_classpredictions_CT[o][p]!=4:
                     classpredictions[o][p]=classpredictions[o][p]
# Kappa over all annotations and predictions merged together
tmp_orig=vstack(y_each_patient)
tmp_pred=hstack(classpredictions)

#Performance of optimized predictions 
RES1_F1_all=zeros(shape=(len(babies),len(label)))
RES1_KAPPA=list()

for K in range(len(babies)):
       RES1_KAPPA.append(cohen_kappa_score(y_each_patient[K].ravel(),classpredictions[K],labels=label)) # Find the threshold where Kapaa gets max
       RES1_F1_all[K]=f1_score(y_each_patient[K].ravel(), classpredictions[K],labels=label, average=None)#, pos_label=None)
       
RES1_KAPPA.append(mean(RES1_KAPPA))
RES1_F1_all_mean=array(mean(RES1_F1_all,0))    
RES1_KAPPA_overall=cohen_kappa_score(tmp_orig.ravel(),tmp_pred.ravel(),labels=label)





"""
END
"""



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
