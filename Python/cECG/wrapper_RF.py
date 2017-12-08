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
import copy

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
class var():
       dataset='cECG'  # Either ECG or cECG and later maybe MMC or InnerSense
       #***************
       selectedbabies =[2,3,5,6,7,8]  #0-8 ('4','5','6','7','9','10','11','12','13')
       #---------------------------
       # Feature list
       lst = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29]
       #lst_old=[3,4,5,6,7,8,9,10,11,14,15,16,17,18,19,20,21,22,23,24,25,26] # From first paper to compare with new features
       #lst=lst_old
       #---------------------------
       label=[1,2,6] # 1=AS 2=QS 3=Wake 4=Care-taking 5=NA 6= transition
       #--------------------------
       classweight=1 # If classweights should be automatically ('balanced') determined and used for trainnig use: 0; IF they should be calculated by own function use 1
       saving=0
       ux=0 # if using this on Linux cluster use 1 to change adresses
       scaling='Z' # Scaling Z or MM 
       #---------------------------
       LoosingAnnot5= 0# exchange state 5 if inbetween another state with this state (also only if length <= x)
       LoosingAnnot6=0  #Exchange state 6 with the following or previouse state (depending on direction)
       LoosingAnnot6_2=0 # as above, but chooses always 2 when 6 was lead into with 1
       direction6=0 # if State 6 should be replaced with the state before, use =1; odtherwise with after, use =0. Annotators used before.
       Smoothing_short=0 # # short part of any annotation are smoothed out. 
       Pack4=0 # State 4 is often split in multible short parts. Merge them together as thebaby does not calm downin 1 min
       merge34=1 # Merging state 3(wake) and 4(CT) togehter. Geoth 4 afterwards
       if merge34 and 3 in label:
              label.remove(3)
       #---------------------------
       Movingwindow=10 # WIndow size for moving average
       preaveraging=0
       postaveraging=1
       exceptNOF=1 #Which Number of Features (NOF) should be used with moving average?  all =oth tzero; only some or all except some defined in FEAT
       onlyNOF=0 # [0,1,2,27,28,29]
       FEAT=[0,1,2]
       #----------------------------
       PolyTrans=0#use polinominal transformation on the Features specified in FEATp
       ExpFactor=2# which degree of polinomonal (2)
       exceptNOpF= 0#Which Number of Features (NOpF) should be used with polynominal fit?  all =0; only some or all except some defined in FEATp
       onlyNOpF=1 # [0,1,2,27,28,29]
       FEATp=[0,3,4,5]
       RBFkernel=1
       #--------------------------
       Used_classifier='GB' #RF=random forest ; ERF= extreme random forest; TR= Decission tree; GB= Gradient boosting
       drawing=0 # draw a the tree structure
       SVMtype='Kernel'  #Kernel or Linear
       Kernel='RBF'#'polynomial'   or ' RBF'  
       strategie='ovr' # or or crammer_singer tochoose for LinearSVM multiclass stragey
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
       deciding_performance_measure='Kappa' #Kappa , F1_second_label, F1_third_label, F1_fourth_label
       
"""
#*********************************************************************************************************
"""      
copy_of_var = type('copy_of_var', var.__bases__, dict(var.__dict__))       # First create copy, otherwise it changes both 
var_CT=copy_of_var()

var_CT.Movingwindow=47 # WIndow size for moving average
var_CT.postaveraging=1
var_CT.exceptNOF=1 #Which Number of Features (NOF) should be used with moving average?  all =oth tzero; only some or all except some defined in FEAT
var_CT.FEAT=[1,2,27,28] # FRO CT
#----------------------------
var_CT.SamplingMeth='NONE'  # 'NONE' 'SMOTE'  or 'ADASYN'
#----------------------------
var_CT.PolyTrans=0#use polinominal transformation on the Features specified in FEATp
var_CT.onlyNOpF=1 # [0,1,2,27,28,29]
#FEATp=[1,2,27,28] # FRO CT
var_CT.FEATp=[1,2,27,28]
#--------------------
var_CT.N=50 # Estimators for the trees
var_CT.crit='entropy' #gini or entropy
var_CT.msl=3  #min_sample_leafe
var_CT.deciding_performance_measure='Kappa' #Kappa , F1_second_label, F1_third_label, F1_fourth_label
var_CT.Used_classifier='RF' #RF=random forest ; ERF= extreme random forest; TR= Decission tree; GB= Gradient boosting

"""
#*********************************************************************************************************
"""
copy_of_var2 = type('copy_of_var2', var.__bases__, dict(var.__dict__))       # First create copy, otherwise it changes both 
var_IS=copy_of_var2()

var_IS.Movingwindow=47 # WIndow size for moving average
var_IS.postaveraging=1
var_IS.exceptNOF=1 #Which Number of Features (NOF) should be used with moving average?  all =oth tzero; only some or all except some defined in FEAT
var_IS.FEAT=[0,1,2] # FRO CT
#----------------------------
var_IS.SamplingMeth='NONE'  # 'NONE' 'SMOTE'  or 'ADASYN'
#---------------------------
var_IS.PolyTrans=1#use polinominal transformation on the Features specified in FEATp
var_IS.onlyNOpF=1 # [0,1,2,27,28,29]
var_IS.FEATp=[0,1,2,27,28]
#--------------------
var_IS.N=100 # Estimators for the trees
var_IS.crit='gini' #gini or entropy
var_IS.msl=5  #min_sample_leafe
var_IS.deciding_performance_measure='Kappa' #Kappa , F1_second_label (label[1]), F1_third_label (label[2]), F1_fourth_label (label[3])
var_IS.Used_classifier='ERF' #RF=random forest ; ERF= extreme random forest; TR= Decission tree; GB= Gradient boosting

"""
**************************************************************************
Finished Loading data declaration
**************************************************************************      
"""

"""
CHECKUP
"""
####Cheking for miss spelling
if var.Used_classifier not in Klassifier:
       sys.exit('Misspelling in Used_classifier')    
if var.SamplingMeth not in SampMeth:
       sys.exit('Misspelling in SamplingMeth')   
if var.WhichMix not in Whichmix:
       sys.exit('Misspelling in WhichMix')         
  
"""
Loading Data
"""
Class_dict, features_dict, features_indx=Feature_names()

# CHOOSING WHICH FEATURE MATRIX IS USED
def loadingdata_LOOCV(whichMix,choosenlabel):
       if choosenlabel=='QS':
              var_load=var
       if choosenlabel=='CT':
              var_load=var_CT
       if choosenlabel=='IS':
              var_load=var_IS              
              
       if var.WhichMix=='perSession':
              babies, AnnotMatrix_each_patient,FeatureMatrix\
              =Loading_data_perSession(var_load.dataset, var_load.selectedbabies, var_load.lst, var_load.ux, var_load.scaling,\
                            var_load.LoosingAnnot5, var_load.LoosingAnnot6, var_load.LoosingAnnot6_2, var_load.direction6, var_load.plotting, var_load.Smoothing_short, var_load.Pack4,var_load.merge34,\
                            var_load.Movingwindow, var_load.preaveraging, var_load.postaveraging, var_load.exceptNOF, var_load.onlyNOF, var_load.FEAT,\
                            var_load.PolyTrans, var_load.ExpFactor, var_load.exceptNOpF, var_load.onlyNOpF, var_load.FEATp,var_load.RBFkernel)       
              
       elif var.WhichMix=='all':              
              babies, AnnotMatrix_each_patient, FeatureMatrix\
              =Loading_data_all(var_load.dataset,var_load.selectedbabies,var_load.lst,var_load.ux,var_load.scaling,\
                            var_load.LoosingAnnot5,var_load.LoosingAnnot6,var_load.LoosingAnnot6_2,var_load.direction6,var_load.plotting,var_load.Smoothing_short,var_load.Pack4,var_load.merge34,\
                            var_load.Movingwindow,var_load.preaveraging,var_load.postaveraging,var_load.exceptNOF,var_load.onlyNOF,var_load.FEAT,\
                            var_load.PolyTrans,var_load.ExpFactor,var_load.exceptNOpF,var_load.onlyNOpF,var_load.FEATp,var_load.RBFkernel)
              
       y_each_patient,classpredictions,probabilities,Fimportance,RES_F1_macro,RES_KAPPA,RES_F1_micro,RES_F1_weigth,RES_F1_all,RES_scoring,\
       RES_Kappa_Performance,RES_F1_mall\
       =leave_one_out_cross_validation(babies,AnnotMatrix_each_patient,FeatureMatrix,\
         var_load.label,var_load.classweight, var_load.Used_classifier, var_load.drawing, var_load.lst,var.ChoosenKind,var_load.SamplingMeth,var_load.probability_threshold,var_load.plotting,var_load.compare,var_load.saving,\
         var_load.N,var_load.crit,var_load.msl,var_load.deciding_performance_measure)

       RES_F1_all_mean=array(mean(RES_F1_all,0))       
              
       return y_each_patient,classpredictions,probabilities,Fimportance,\
       RES_F1_macro,RES_KAPPA,RES_F1_micro,RES_F1_weigth,RES_F1_all,RES_scoring,\
       RES_Kappa_Performance,RES_F1_mall,RES_F1_all_mean\



"""
LOOCV ************************************************************************
"""   
"""
RUN 1 QS
"""    
if 2 in var.label:
       y_each_patient,RES_classpredictions_QS,probabilities,RES_Fimportance_QS,_,\
       RES_Kappa_QS,_,_,RES_F1_all_QS,_, _,_,RES_F1_all_QS_mean\
       =loadingdata_LOOCV(var.WhichMix,'QS')       
       
       classpredictions=RES_classpredictions_QS[:]
"""
RUN 2  CT
"""

"""
RUN 3  IS
"""
if 6 in var.label:
       y_each_patient,RES_classpredictions_IS,probabilities,RES_Fimportance_IS,_,\
       RES_Kappa_IS,_,_,RES_F1_all_IS,_,_,_,RES_F1_all_IS_mean\
       =loadingdata_LOOCV(var.WhichMix,'IS')   
       
       #Optimize prediction by taking predictions for specific classes from differnt classifiers
       #The base predicitions are the one from QS classifier. Over that each 4(care taking) is decided/changed by the classifer results for CT
#       classpredictions=RES_classpredictions_QS[:]
       for o in range(len(RES_classpredictions_IS)):
              for p in range(len(RES_classpredictions_IS[o])):  
       #              ind=classpredictionsCT[o]==4
                     if RES_classpredictions_IS[o][p]==6: 
                            classpredictions[o][p]=6
                     elif classpredictions[o][p]==6 and RES_classpredictions_IS[o][p]!=6: # CT determines if 4 or not
                            classpredictions[o][p]=RES_classpredictions_IS[o][p]
                    
                            
if 4 in var.label:

       y_each_patient,RES_classpredictions_CT,probabilities,RES_Fimportance_CT,_,\
       RES_Kappa_CT,_,_,RES_F1_all_CT,_,_,_,RES_F1_all_CT_mean\
       =loadingdata_LOOCV(var.WhichMix,'CT')   
       
       #Optimize prediction by taking predictions for specific classes from differnt classifiers
       #The base predicitions are the one from QS classifier. Over that each 4(care taking) is decided/changed by the classifer results for CT
#       classpredictions=RES_classpredictions_QS[:]
       for o in range(len(RES_classpredictions_CT)):
              for p in range(len(RES_classpredictions_CT[o])):  
       #              ind=classpredictionsCT[o]==4
                     if RES_classpredictions_CT[o][p]==4: 
                            classpredictions[o][p]=4
                     elif classpredictions[o][p]==4 and RES_classpredictions_CT[o][p]!=4: # CT determines if 4 or not
                            classpredictions[o][p]=RES_classpredictions_CT[o][p]
                                              
                            
""" 
Overall perfomrance analysis
"""
                            
# Kappa over all annotations and predictions merged together
                     
                     
tmp_orig=vstack(y_each_patient)
tmp_pred=hstack(classpredictions)

#Performance of optimized predictions 
RES1_final_F1_all=zeros(shape=(len(var.selectedbabies),len(var.label)))
RES1_final_Kappa=list()

for K in range(len(var.selectedbabies)):
       RES1_final_Kappa.append(cohen_kappa_score(y_each_patient[K].ravel(),classpredictions[K],labels=var.label)) # Find the threshold where Kapaa gets max
       RES1_final_F1_all[K]=f1_score(y_each_patient[K].ravel(), classpredictions[K],labels=var.label, average=None)#, pos_label=None)
       
RES1_final_Kappa.append(mean(RES1_final_Kappa))
RES1_final_F1_mall=array(mean(RES1_final_F1_all,0))    
RES1_final_Kappa_overall=cohen_kappa_score(tmp_orig.ravel(),tmp_pred.ravel(),labels=var.label)





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

if var.saving:
    print("saved at: %s" %zeit)
print("Console 1 : "); print(description)
