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
from Plotting_Feature_importance import PlottingFeatureImportance

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
Klassifier=['RF','ERF','TR','GB', 'LR']
SampMeth=['NONE','SMOTE','ADASYN']
Whichmix=['perSession', 'all']


#_Labels_ECG_Featurelist_Scoring_classweigt_C_gamma

description='_123456_cECG_lst_micro_'
consoleinuse='4'
dispinfo=0
savepath='/home/310122653/Pyhton_Folder/cECG/Results/'
histogram=1
"""
**************************************************************************
Loading data declaration & Wrapper variables
0,1,2 = ECG
3,4,5,6,7,8,9,10,11,12,13,14,15,16,17= HRV time domain
18,19,20,21,22,23,24,25,26,27,28 = HRV freq domain
29,30,31,32,33
**************************************************************************
"""
#lstQS=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,32]#QS
#lstCTW=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,32]# CWT
#lstIS=[4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,32]#IS
#
#lstQS= [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,32,33] 
##lstIS= [0,1,2,3,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,29,32,33]  
#lstCTW=[4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,32]

lstQS= [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,32,33] 
lstCTW=[1,2,4,5,6,7,10,11,12,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,32]
lstIS= [1,2,4,5,6,7,10,11,12,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,32]#IS


#AVERAGING
FensterQS=10 ; FensterCTW=47; FensterIS=20 # at start
FensterQS=20 ; FensterCTW=47; FensterIS=20
ExFeatQS=1; ExFeatCTW=0; ExFeatIS=0

FEATaQS=[lstQS.index(0),lstQS.index(25),lstQS.index(33)]# 9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26
FEATaIS=[0,25,33]
FEATaCTW=[0,25,33]

#POLY
PolyTQS=0; 
#FEATpQS=[0,10,27]#A
#FEATpQS=[0,10,27,15,21]#AB
#FEATpQS=[0,10,27,15,21,9,18]#ABC
FEATpQS=[11,14,29,30,31]#[lstIS.index(11),lstIS.index(12),lstIS.index(13),lstIS.index(14),lstIS.index(10),lstIS.index(24),lstIS.index(29),lstIS.index(32)]#[0,3,4,5]#12

PolyTCTW=1; 
#FEATpCTW=[0,9,27]#A
#FEATpCTW=[0,9,27,10,16,19]#AB
#FEATpCTW=[0,9,27,10,16,19,3,13]#ABC
FEATpCTW=[lstIS.index(11),lstIS.index(12),lstIS.index(14),lstIS.index(10),lstIS.index(24),lstIS.index(29),lstIS.index(32)]#[lstCTW.index(11),lstCTW.index(27),lstCTW.index(29)]

PolyTIS=1; 
#FEATpIS=[0,13,30]#A
#FEATpIS=[0,13,30,21,24]#AB
#FEATpIS=[0,13,30,21,24,18]#ABC
FEATpIS=[lstIS.index(11),lstIS.index(12),lstIS.index(14),lstIS.index(10),lstIS.index(24),lstIS.index(29),lstIS.index(32)]#456 11 12 27       
       

ASQS= [0,0.69]#[0.65,0]#[0.63,0.7]
ASCTW=[0,0.65]
ASIS= [0,0.67]#0.67,0]#[0.73,0.7]

NQS=100; mslQS=5 #100 2
NCTW=500 ; mslCTW=2
NIS=500; mslIS=3 #500#3








Rpeakmethod='R' #R or M
dataset='cECG'  # Either ECG or cECG and later maybe MMC or InnerSense
#***************
selectedbabies =[0,1,2,3,5,6,7,8] #0-8 ('4','5','6','7','9','10','11','12','13')
#selectedbabies=[0,1,2,3,5,6,7,8]
label=[1,2,3,4,6] # 1=AS 2=QS 3=Wake 4=Care-taking 5=NA 6= transition
#---------------------------
# Feature list
lst = lstQS

#lst_old=[3,4,5,6,7,8,9,10,11,14,15,16,17,18,19,20,21,22,23,24,25,26] # From first paper to compare with new features
#lst=lst_old
#---------------------------
ux=0 # if using this on Linux cluster use 1 to change adresses
scaling='Z' # Scaling Z or MM 
#---------------------------
Movingwindow=FensterQS # WIndow size for moving average
preaveraging=0
postaveraging=1
exceptNOF=ExFeatQS #Which Number of Features (NOF) should be used with moving average?  all =oth tzero; only some or all except some defined in FEAT
onlyNOF=0 # [0,1,2,27,28,29]
FEAT=FEATaQS
#----------------------------
PolyTrans=PolyTQS#use polinominal transformation on the Features specified in FEATp
ExpFactor=2# which degree of polinomonal (2)
exceptNOpF= 0#Which Number of Features (NOpF) should be used with polynominal fit?  all =0; only some or all except some defined in FEATp
onlyNOpF=1 # [0,1,2,27,28,29]
FEATp=FEATpQS
#---------------------------
SamplingMeth='NONE'  # 'NONE' 'SMOTE'  or 'ADASYN' #For up and downsampling of data
ChoosenKind=0   # 0-3['regular','borderline1','borderline2','svm'] only when using SMOTE
#---------------------------
probability_threshold=1 # 1 to use different probabilities tan 0.5 to decide on the class. At the moment it is >=0.2 for any other calss then AS
ASprobLimit=ASQS# Determine the AS lower limit for the probability for which another class is chosen than AS. For: [3 labels, >3 labels]
WhichMix='perSession' #perSession or all  # determine how the data was scaled. PEr session or just per patient
#--------------------
Used_classifier='RF' #RF=random forest ; ERF= extreme random forest; TR= Decission tree; GB= Gradient boosting
N=NQS # Estimators for the trees
crit='gini' #gini or entropy method for trees 
msl=mslQS  #min_sample_leafe
deciding_performance_measure='F1_second_label' #Kappa , F1_second_label, F1_third_label, F1_fourth_label
drawing=0 # draw a the tree structure

#Abstellgleis
#----------------------------
classweight=1 # If classweights should be automatically ('balanced') determined and used for trainnig use: 0; IF they should be calculated by own function use 1
saving=0     
plotting=0 # plot annotations over time per patient
compare=0 # additional plot  
#---------------------------
LoosingAnnot5=0# exchange state 5 if inbetween another state with this state (also only if length <= x)
LoosingAnnot6=0  #Exchange state 6 with the following or previouse state (depending on direction)
LoosingAnnot6_2=0 # as above, but chooses always 2 when 6 was lead into with 1
direction6=0 # if State 6 should be replaced with the state before, use =1; odtherwise with after, use =0. Annotators used before.
Smoothing_short=0 # # short part of any annotation are smoothed out. 
Pack4=0 # State 4 is often split in multible short parts. Merge them together as thebaby does not calm downin 1 min
merge34=1
if merge34 and 3 in label:
              label.remove(3)
       
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
              babies, AnnotMatrix_each_patient,FeatureMatrix_each_patient\
              =Loading_data_perSession(dataset, selectedbabies, lst, Rpeakmethod,ux, scaling,\
                            LoosingAnnot5, LoosingAnnot6, LoosingAnnot6_2, direction6, plotting, Smoothing_short, Pack4, merge34,\
                            Movingwindow, preaveraging, postaveraging, exceptNOF, onlyNOF, FEAT,\
                            PolyTrans, ExpFactor, exceptNOpF, onlyNOpF, FEATp,dispinfo)       
              
       elif WhichMix=='all':              
              babies, AnnotMatrix_each_patient, FeatureMatrix_each_patient\
              =Loading_data_all(dataset,selectedbabies,lst, Rpeakmethod,ux,scaling,\
                            LoosingAnnot5,LoosingAnnot6,LoosingAnnot6_2,direction6,plotting,Smoothing_short,Pack4,merge34,\
                            Movingwindow,preaveraging,postaveraging,exceptNOF,onlyNOF,FEAT,\
                            PolyTrans,ExpFactor,exceptNOpF,onlyNOpF,FEATp)
              
       """
       LOOCV ************************************************************************
       """                
       y_each_patient,\
       classpredictions,\
       Probabilities,\
       Fimportance,\
       F1_macro,\
       Kappa,\
       F1_micro,\
       F1_weigth,\
       F1_all,\
       scoring,\
       Kappa_Performance,\
       F1_all_mean\
       =leave_one_out_cross_validation(babies,AnnotMatrix_each_patient,FeatureMatrix_each_patient,\
                label,classweight, Used_classifier, drawing, lst,ChoosenKind,SamplingMeth,probability_threshold,ASprobLimit,plotting,compare,saving,\
                N,crit,msl,deciding_performance_measure,dispinfo)
       
       RES_F1_all_IS_mean=array(mean(F1_all,0))              
       meanFimportance=mean(Fimportance,0)
       return  babies, y_each_patient, classpredictions,Probabilities, Fimportance, Kappa,F1_all,meanFimportance


babies,y_each_patient,classpredictions_QS,probabilities_QS,Fimportance_QS,Kappa_QS,F1_all_QS,FimportanceMean_QS\
= loadingdata(WhichMix)                  

F1_all_QS_mean=array(mean(F1_all_QS,0))

"""
RUN 2  CT
"""
if 4 in label:         
       lst = lstCTW
       #---------------------------
       scaling='Z' # Scaling Z or MM 
       #---------------------------
       Movingwindow=FensterCTW # WIndow size for moving average
       preaveraging=0
       postaveraging=1
       exceptNOF=ExFeatCTW #Which Number of Features (NOF) should be used with moving average?  all =oth tzero; only some or all except some defined in FEAT
       onlyNOF=0 # [0,1,2,27,28,29]
       FEAT=FEATaCTW
       #----------------------------
       PolyTrans=PolyTCTW#use polinominal transformation on the Features specified in FEATp
       ExpFactor=2# which degree of polinomonal (2)
       exceptNOpF= 0#Which Number of Features (NOpF) should be used with polynominal fit?  all =0; only some or all except some defined in FEATp
       onlyNOpF=1 # [0,1,2,27,28,29]
#       FEATp=[lst.index(0),lst.index(3),lst.index(4),lst.index(5)]#456 11 12 27       
#       FEATp=[0,3,4,5,30]
       FEATp=FEATpCTW
       #---------------------------
       SamplingMeth='NONE'  # 'NONE' 'SMOTE'  or 'ADASYN' #For up and downsampling of data
       ChoosenKind=1   # 0-3['regular','borderline1','borderline2','svm'] only when using SMOTE
       #---------------------------
       probability_threshold=1 # 1 to use different probabilities tan 0.5 to decide on the class. At the moment it is >=0.2 for any other calss then AS
       ASprobLimit=ASCTW# Determine the AS lower limit for the probability for which another class is chosen than AS. For: [3 labels, >3 labels]
       #--------------------
       Used_classifier='RF' #RF=random forest ; ERF= extreme random forest; TR= Decission tree; GB= Gradient boosting
       N=NCTW # Estimators for the trees
       crit='entropy' #gini or entropy method for trees 
       msl=mslCTW  #min_sample_leafe
       deciding_performance_measure='Kappa' #Kappa , F1_second_label, F1_third_label, F1_fourth_label 
       
       """
       LOOCV
       """
       babies, y_each_patient,classpredictions_CT,probabilities_CT,Fimportance_CT,Kappa_CT,F1_all_CT,FimportanceMean_CT\
       = loadingdata(WhichMix)  
       
       F1_all_CT_mean=array(mean(F1_all_CT,0))

"""" 
Run 3 IS
"""
if 6 in label:
       #      0 1 2 3 4 5 6  7  8  9  10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 28 29 30
       lst = lstIS 
       #---------------------------
       scaling='Z' # Scaling Z or MM 
       #---------------------------
       Movingwindow=FensterIS # WIndow size for moving average
       preaveraging=0
       postaveraging=1
       exceptNOF=ExFeatIS #Which Number of Features (NOF) should be used with moving average?  all =oth tzero; only some or all except some defined in FEAT
       onlyNOF=0 # [0,1,2,27,28,29]
       FEAT=FEATaIS #[lst.index(4),lst.index(5),lst.index(6),lst.index(25)]
       #----------------------------
       PolyTrans=PolyTIS#use polinominal transformation on the Features specified in FEATp
       ExpFactor=2# which degree of polinomonal (2)
       exceptNOpF= 0#Which Number of Features (NOpF) should be used with polynominal fit?  all =0; only some or all except some defined in FEATp
       onlyNOpF=1 # [0,1,2,27,28,29]
#       FEATp=[lst.index(8),lst.index(9),lst.index(10),lst.index(15),lst.index(16),lst.index(23)]#456 11 12 27       
       FEATp=FEATpIS
#       FEATp=[0,3,4,5,6]
#       ---------------------------
       SamplingMeth='NONE'  # 'NONE' 'SMOTE'  or 'ADASYN' #For up and downsampling of data
       ChoosenKind=0   # 0-3['regular','borderline1','borderline2','svm'] only when using SMOTE
       #---------------------------
       probability_threshold=1 # 1 to use different probabilities tan 0.5 to decide on the class. At the moment it is >=0.2 for any other calss then AS
       ASprobLimit=ASIS# Determine the AS lower limit for the probability for which another class is chosen than AS. For: [3 labels, >3 labels]
       #--------------------
       Used_classifier='RF' #RF=random forest ; ERF= extreme random forest; TR= Decission tree; GB= Gradient boosting
       N=NIS # Estimators for the trees
       crit='gini' #gini or entropy method for trees 
       msl=mslIS  #min_sample_leafe
       deciding_performance_measure='Kappa' #Kappa , F1_second_label, F1_third_label, F1_fourth_label

       """
       LOOCV ************************************************************************
       """       
       babies, y_each_patient, classpredictions_IS,probabilities_IS,Fimportance_IS,Kappa_IS,F1_all_IS,FimportanceMean_IS\
       = loadingdata(WhichMix)  
       
       F1_all_IS_mean=array(mean(F1_all_IS,0))

       
#Optimize prediction by taking predictions for specific classes from differnt classifiers
#The base predicitions are the one from QS classifier. Over that each 4(care taking) is decided/changed by the classifer results for CT
      
classpredictions=classpredictions_QS[:]
if 4 in label: 
       for o in range(len(classpredictions)):
              for p in range(len(classpredictions[o])):  
                     if classpredictions_CT[o][p]==4 :#and probabilities_CT[o][p,label.index(4)]>0.2 and probabilities_QS[o][p,label.index(2)]<0.3  :  
                            classpredictions[o][p]=4
                     elif classpredictions[o][p]==4 and classpredictions_CT[o][p]!=4: # CT determines if 4 or not
                            classpredictions[o][p]=classpredictions_CT[o][p]
                     elif classpredictions[o][p]!=4 and classpredictions_CT[o][p]!=4:
                            classpredictions[o][p]=classpredictions[o][p]
                            
if 6 in label and 4 not in label: 
       for o in range(len(classpredictions)):
              for p in range(len(classpredictions[o])):  
                     if classpredictions_IS[o][p]==6 and probabilities_IS[o][p,label.index(6)]>0.2 and probabilities_QS[o][p,label.index(2)]<0.3  :   
                            classpredictions[o][p]=6
                     elif classpredictions[o][p]==6 and classpredictions_IS[o][p]!=6: # CT determines if 6 or not
                            classpredictions[o][p]=classpredictions_IS[o][p]
                     elif classpredictions[o][p]!=6 and classpredictions_IS[o][p]!=6:
                            classpredictions[o][p]=classpredictions[o][p]
                            
if 4 in label and 6 in label:
       for o in range(len(classpredictions)):
              for p in range(len(classpredictions[o])):
                     if classpredictions_CT[o][p]==4:                      
                            classpredictions[o][p]=4
                     elif classpredictions[o][p]==4 and classpredictions_CT[o][p]!=4: # CT determines if 4 or not
                            classpredictions[o][p]=classpredictions_CT[o][p]
                     elif classpredictions[o][p]!=4 and classpredictions_CT[o][p]!=4:
                            classpredictions[o][p]=classpredictions[o][p]
                     if classpredictions_IS[o][p]==6:
                            if classpredictions[o][p]==4: # if there is already class CT and CT probability is not 20% larger than Is, replace with IS, otherwise leave 4
                                   if probabilities_IS[o][p,label.index(6)]>1.2*probabilities_CT[o][p,label.index(4)]: # if the probability for CT is 20% higher than for IS take CT, otherwise IS
                                          classpredictions[o][p]=6
                                   else: 
                                          classpredictions[o][p]=4
                            else:
                                   classpredictions[o][p]=6
                            
                     elif classpredictions[o][p]==6 and classpredictions_IS[o][p]!=6: # CT determines if 6 or not
                            classpredictions[o][p]=classpredictions_IS[o][p]
                     elif classpredictions[o][p]!=6 and classpredictions_IS[o][p]!=6:
                            classpredictions[o][p]=classpredictions[o][p]          
# Kappa over all annotations and predictions merged together
tmp_orig=vstack(y_each_patient)
tmp_pred=hstack(classpredictions)

#Performance of optimized predictions 
RES1_F1_all=zeros(shape=(len(babies),len(label)))
KonfMAT=list()
KonfMATall=list()
RES1_Kappa=list()

for K in range(len(babies)):
       KonfMAT.append(confusion_matrix(y_each_patient[K].ravel(), classpredictions[K], labels=label, sample_weight=None))
       RES1_Kappa.append(cohen_kappa_score(y_each_patient[K].ravel(),classpredictions[K],labels=label)) # Find the threshold where Kapaa gets max
       RES1_F1_all[K]=f1_score(y_each_patient[K].ravel(), classpredictions[K],labels=label, average=None)#, pos_label=None)
RES1_kappa_STD=std(RES1_Kappa)       
RES1_Kappa.append(mean(RES1_Kappa))
RES1_F1_all_mean=array(mean(RES1_F1_all,0))    
RES1_KAPPA_overall=cohen_kappa_score(tmp_orig.ravel(),tmp_pred.ravel(),labels=label)
KonfMATall.append(confusion_matrix(tmp_orig.ravel(), tmp_pred.ravel(), labels=label, sample_weight=None))


#PlottingFeatureImportance(Fimportance_QS,Fimportance_CT,Fimportance_IS,features_dict)

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
disp(  RES1_Kappa[-1])
disp(RES1_kappa_STD)
disp (RES1_KAPPA_overall)
