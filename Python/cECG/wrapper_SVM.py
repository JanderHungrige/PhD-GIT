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
#get_ipython().magic('reset -sf')

from platform import python_version
print ('Python version: ', sep=' ', end='', flush=True);print( python_version())	



from Loading_5min_mat_files_cECG import Loading_data_all,Loading_data_perSession,Feature_names,Loading_Annotations
from Classifier_routines import SVM_routine_no_sampelWeight
from Classifier_routines import Validate_with_classifier
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
Klassifier=['RF','ERF','TR','GB']
SampMeth=['NONE','SMOTE','ADASYN']
Whichmix=['perSession', 'all']



"""
**************************************************************************
CHANGE THE DATASET IN Loading_5min_mat_files_cECG.py IF USING ECG OR cECG
**************************************************************************
"""
#_Labels_ECG_Featurelist_Scoring_classweigt_C_gamma

description='_12_ECG_lst_Kappa_SMOTE3_Kernel_3_001_61_direct1_51_Z'
consoleinuse='4'

savepath='/home/310122653/Pyhton_Folder/cECG/Results/'

#### SELECTING THE LABELS FOR SELECTED BABIES
label=array([1,2]) # 1=AS 2=QS 3=Wake 4=Care-taking 5=NA 6= transition
babies =[0,1,2,3,4,5,6,7,8] #0-8
    
#### CREATE ALL POSSIBLE COMBINATIONS OUT OF 30 FEATURES. STOP AT Ncombos FEATURE SET(DUE TO SVM COMPUTATION TIME)
lst = [0,1,2,3,4,5,6,7,8,9,10,11,12,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29]
lst_old=[3,4,5,6,7,8,9,10,11,13,15,16,17,18,19,20,21,22,23,24,25,26] # From first paper to compare with new features
#lst=lst_old
"""
**************************************************************************
CHANGE THE DATASET IN Loading_5min_mat_files_cECG.py IF USING ECG OR cECG
**************************************************************************
"""

Ncombos=1
preGridsearch=1
finalGridsearch=1
plotting_grid=0
c=3
gamma=0.001

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
       
combs=[]
bestAUCs=nan
Subsets=list() 
Performance=list()
ValidatedPerformance_macro=list()
ValidatedPerformance_K=list()
ValidatedPerformance_micro=list()
ValidatedPerformance_weigth=list()  
ValidatedPerformance_all=zeros(shape=(len(babies),len(label)))
BiasInfluence=list()
collected_mean_auc=[]


#for i in range(1, len(lst)+1):
####Cheking for miss spelling      
if SamplingMeth not in SampMeth:
       sys.exit('Misspelling in SamplingMeth')   
if WhichMix not in Whichmix:
       sys.exit('Misspelling in WhichMix')         
     
# CHOOSING WHICH FEATURE MATRIX IS USED

"""
START 
"""      
i=1    
els = [list(x) for x in itertools.combinations(lst, i)]
combs.extend(els)  
combs_short=[combs for combs in combs if len(combs) <= Ncombos] #due to computation time we stop with 5 Features per set.

babies, AnnotMatrix_each_patient, FeatureMatrix_each_patient\
              =Loading_data_perSession(dataset,selectedbabies,lst,ux,scaling,\
                            LoosingAnnot5,LoosingAnnot6,LoosingAnnot6_2,direction6,plotting,Smoothing_short,Pack4,merge34,\
                            Movingwindow,preaveraging,postaveraging,exceptNOF,onlyNOF,FEAT,\
                            PolyTrans,ExpFactor,exceptNOpF,onlyNOpF,FEATp,RBFkernel)
              
""" 
GRID SEARCH FOR C AND GAMMA
"""
if preGridsearch:
    # For pregridsearch take all label
    # for final gridsearch use the labels you are investigating
    if SVMtype=='Linear':
           gridC=float64([1])#([1,4,10,100,1000])
           gridY=float64([1])
    elif SVMtype=='Kernel':
           gridC=float64([1])#([1,4,10,100,1000])
           gridY=float64([1])#([10,3,1,0.5,0.1,0.05,0.01,0.005,0.001,0.0005,0.0001])
    [c,gamma]=GridSearch_all(plotting_grid,gridC,gridY,lst,label,babies,AnnotMatrix_each_patient,FeatureMatrix_each_patient,classweight,\
                                ChoosenKind,SamplingMeth,probability_threshold,SVMtype,strategie,deciding_performance_measure)
    disp(c);disp(gamma)
#    sys.exit('Jan werth first gridsearch')
    
"""
BRUTE FORCE
"""   
for V in range(len(babies)):
    print('**************************')
    print('Validating on patient: %i' %V)
    # Resetting variables after loop
    m=[]
    bestAUCs=list()
    idx_m=list()
    selectedF=list()
    collected_mean_auc=list()
    counter=0
    
    
    lstcpy=lst[:]
    selected_babies=list(delete(babies,babies[V-1]))# Babies to train and test on ; j-1 as index starts with 0
    selected_validation=babies[V-1]# Babies to validate on 
    #### Create Matrices for selected babies
    
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
        
        resultsF1_maco,resultsK,resultsF1_micro,resultsF1_weight,resultsF1_all \
        =SVM_routine_no_sampelWeight(Xfeat,y_each_patient,selected_babies,label,classweight,c,gamma,\
                                            ChoosenKind,SamplingMeth,probability_threshold,SVMtype,strategie,deciding_performance_measure)
        collected_mean_auc.append(resultsK) # This collects the mean AUC of each itteration. As we want to know whcih combination is the best, we collect all mean AUCs and search for the maximum later
        print('BF Round: %i of:%i' %(Fc+1,len(combs_short)))    
           
    """ 
    GREEDY FORWARD SEARCH
    """
    # 1:take best combo and add 1 other feature to it4
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
    print('-----');print('Subset: ', sep=' ', end='', flush=True); print(selectedF); print('max AUC: %.4f' %m); print('-----')
    
    for j in range(len(selectedF)):
        lstcpy.remove(selectedF[j]) # Remove the top feature from the list
    
    lst2=selectedF[:] # rewrite the selected features into a new list that we can experimentally add new features
    for l in range(len(lstcpy)):
        collected_mean_auc_new=[]
    #    print(lst)
        for k in range(len(lstcpy)): # going through all the leftover features and addign them one by one to the selected set -> test if better
            lst2+=[lstcpy[k]] # adding next feature for test to the lst
    
            # CREATING THE DATASET WITH x numbers of features WITH SPECIFIC LABELS FOR SELECTED BABIES  
            Xfeat=[val[:,lst2] for sb, val in enumerate(FeatureMatrix_auswahl)] # using the feature values for features in lst2 to run
            Xfeat=[val[idx[sb],:] for sb, val in enumerate(Xfeat)]   #choosing only the values which are taged with the used labels (AS=1 QS=2 ...)
        
            resultsF1_maco,resultsK,resultsF1_micro,resultsF1_weight,resultsF1_all \
            =SVM_routine_no_sampelWeight(Xfeat,y_each_patient,selected_babies,label,classweight,c,gamma,\
                                                ChoosenKind,SamplingMeth,probability_threshold,SVMtype,strategie,deciding_performance_measure)

            collected_mean_auc_new.append(resultsK)
            
            lst2=selectedF[:]  # reset lst2 
                   
        # Find the best AUC for that run  
        m_new=max(collected_mean_auc_new)      
        if m_new >= m: # if there is an increase in performance with added feature
            if counter>0:
                selecteF=tmpF[:] # if there is a jump for selected features as ther was no increase for e.g. 2 additional features but with the thirdthere is, selectedF is now on the same level again. 
                
            bestAUCs.append(m_new)
            idx_m=[i for i, j in enumerate(collected_mean_auc_new) if j == m_new] # find the feature indx which increased performance
            addition=lstcpy[idx_m[0]] # get the feature (=feature idx) to add into the selectedF
            selectedF+=[addition]# add new found feature
            lstcpy.remove(addition) # Remove the top feature from the list
            m=m_new #new maximum value to compare to
            counter=0 # reset the counter for adding Featurees even when not higher in value
            if len(label)==2:
                print('max AUC: %.4f' %m); print('Subset: '); print(selectedF); print('')                      
            else:
                print('max F1|Kappa: %.4f' %m); print('Subset: '); print(selectedF); print('')                      
        else:
            
            tmpF=selectedF[:]# coppy selected features. Put more Features into them for trial     
            if 'addition' in locals(): # If new feature was found
                   tmpF+=[addition]# add new found feature to the trial set
            counter=counter+1
        if counter==16:
            break
                                         
        if len(label)==2:
            print('maximal AUC: %.4f' %m);print('the perfomrance did not increase'); print('Calculation stopped')
        else:
            print('max F1|Kappa: %.4f' %m); print('Subset: '); print(selectedF); print('')       
        lst2=selectedF[:]    #set lst2 to the new subfeature set
        print('Round: %i of:%i' %(l+1,(len(lstcpy))))    

    """
    VALIDATION
    """        
    #Validate with left out patient 
    # Run the classifier with the selected FEature subset in selecteF
    AnnotMatrix_auswahl_valid=[AnnotMatrix_each_patient[k] for k in babies]              # get the annotation values for selected babies
    FeatureMatrix_auswahl_valid=[FeatureMatrix_each_patient[k] for k in babies]
    idx_valid=[in1d(AnnotMatrix_each_patient[sb],label) for sb in babies]#.values()]     # which are the idices for AnnotMatrix_each_patient == label
    idx_valid=[nonzero(idx_valid[sb])[0] for sb in range(len(babies))]#.values()]              # get the indices where True
    y_each_patient_valid=[val[idx_valid[sb],:] for sb, val in enumerate(AnnotMatrix_auswahl_valid) if sb in range(len(babies))] #get the values for y from idx and label
    Xfeat_valid=[val[:,selectedF] for sb, val in enumerate(FeatureMatrix_auswahl_valid)] # using the feature values for features in lst2 to run
    Xfeat_valid=[val[idx_valid[sb],:] for sb, val in enumerate(Xfeat_valid)]  
    
    resultsF1_macro,resultsK,resultsF1_micro,resultsF1_weight,resultsF1_all \
    =Validate_with_classifier(Xfeat_valid,y_each_patient_valid,selected_babies,selected_validation,label,classweight,c,gamma,\
                                       ChoosenKind,SamplingMeth,probability_threshold,SVMtype,strategie,deciding_performance_measure)
       
#    sys.exit('Jan werth 206')

    Subsets.append(selectedF) # Saving best Subset and performance to be compared with other validated sets 
    Performance.append(m)
    ValidatedPerformance_macro.append(resultsF1_macro)
    ValidatedPerformance_K.append(resultsK)
    ValidatedPerformance_micro.append(resultsF1_micro)
    ValidatedPerformance_weigth.append(resultsF1_weight)    
    ValidatedPerformance_all[V]=resultsF1_all

    BiasInfluence.append(Performance[V]-ValidatedPerformance_micro[V])   
    
"""
CHOOSE COMMON FEATURES
"""
from collections import Counter
import operator
allusedFeatures=list()
Feature_ranking=dict()
for n in range(len(Subsets)):
   allusedFeatures.extend(Subsets[n]) # merging all used features
Feature_ranking=dict(Counter(allusedFeatures))
Feature_ranking_sorted = sorted(Feature_ranking.items(), key=operator.itemgetter(0))   #as touples

"""
PERFOMRANCE FOR COMMON FEATURE SET WITH GRIDSEARCH
"""
Common_Features=list({k:v for (k,v) in Feature_ranking.items() if v > 3}) # find all FEatures which appear more than 3 times

    #### CREATING THE DATASET WITH x numbers of features WITH SPECIFIC LABELS FOR SELECTED BABIES  
Xfeat_final=[val[:,Common_Features] for sb, val in enumerate(FeatureMatrix_auswahl)] # selecting the features to run
Xfeat_final=[val[idx[sb],:] for sb, val in enumerate(Xfeat_final)]   #selecting the datapoints in label    


if finalGridsearch:
       if SVMtype=='Linear':
              gridC=float64([1,4,10,100,1000])
              gridY=float64([1])
       elif SVMtype=='Kernel':
              gridC=float64([1,4,10,100,1000])
              gridY=float64([3,1,0.5,0.1,0.05,0.01,0.005,0.001,0.0005,0.0001])   
              
       [c,gamma]=GridSearch_commonFeatures(plotting_grid,gridC,gridY,lst,label,Xfeat_final,y_each_patient,selected_babies,classweight,\
                                              ChoosenKind,SamplingMeth,probability_threshold,SVMtype,strategie)
       print('The final choosen C and gamma are: C: %.2f gamma: %.2f'%(c,gamma))
#    sys.exit('Jan werth final Gridsearch')
#    input("Press Enter to continue...")


resultsF1_maco_test,resultsK_test,resultsF1_micro_test,resultsF1_weight_test,resultsF1_all_test \
=SVM_routine_no_sampelWeight(Xfeat,y_each_patient,selected_babies,label,classweight,c,gamma,\
                                       ChoosenKind,SamplingMeth,probability_threshold,SVMtype,strategie,deciding_performance_measure)

"""
ENDING stuff
"""
        
if saving:      
    save(savepath + 'Common_Features' + description, Common_Features)     
    
    save(savepath + 'ValidatedPerformance_macro' + description, ValidatedPerformance_macro)     
    save(savepath + 'ValidatedPerformance_K' + description, ValidatedPerformance_K)     
    save(savepath + 'ValidatedPerformance_micro' + description, ValidatedPerformance_micro)
    save(savepath + 'ValidatedPerformance_weigth' + description, ValidatedPerformance_weigth)     
    save(savepath + 'ValidatedPerformance_all' + description, ValidatedPerformance_all)  
    
    save(savepath + 'resultsF1_maco_test' + description, resultsF1_maco_test)     
    save(savepath + 'resultsK_test' + description, resultsK_test)     
    save(savepath + 'resultsF1_micro_test' + description, resultsF1_micro_test)     
    save(savepath + 'resultsF1_weight_test' + description, resultsF1_weight_test)     
    save(savepath + 'resultsF1_all_test' + description, resultsF1_all_test)  
    
    save(savepath + 'C' + description, c) 
    save(savepath + 'gamma' + description, gamma)  
#    import scipy.io as sio
    #sio.savemat('C:/Users/310122653/Dropbox/PHD/python/cECG/Results/', bestAUCs)        

import time
t=time.localtime()
zeit=time.asctime()
print('FINISHED Console ' + consoleinuse)
print("--- %i seconds ---" % (time.time() - start_time))
print("--- %i min ---" % (time.time() - start_time)/60)
print("--- %i h ---" % (time.time() - start_time)/3600)

if saving:
    print("saved at: %s" %zeit)
print("Console 1 : "); print(description)
