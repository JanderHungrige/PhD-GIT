# -*- coding: utf-8 -*-
"""
Created on Fri Aug 18 09:20:57 2017

@author: 310122653
"""
# input: 
# Xfeat: the Feature values as a list per patient
# y_each_patient: the annotations for each Feature value as a list
# Seleted babies: The patients the classifier should use for train and testing as a list [0,1,2,3,4...] 
# label: THe class labels as array e.g. array([1,2]) 1=AS 2=QS ,...
# classweight: just 1 or 0. if classweigths should be calculated outomatically use 1 otherhwise no classweight calculation (DO NOT CONFUSE WITH SAMPLEWEIGHT)

# output: the collected mean AUC of each itteration of train and testing with leave one patient out
import itertools
from matplotlib import *
from numpy import *
from pylab import *
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import StratifiedKFold
from sklearn.cross_validation import cross_val_score
from sklearn.cross_validation import KFold
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LinearRegression
#import xgboost as xgb
#from xgboost.sklearn import XGBClassifier
#from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score 
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import cohen_kappa_score
from sklearn import svm, cross_validation
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.linear_model import Perceptron
from sklearn.metrics import f1_score
from compute_class_weight import*

from Use_imbalanced_learn import cmplx_Oversampling

import pdb# use pdb.set_trace() as breakpoint

#probthres=0.25
probthres_Grid=[0.10,0.15,0.2,0.22,0.25,0.27,0.3,0.33,0.35,0.37,0.4,0.42,0.44,0.5]
"""
Without Sample weights
"""
def Classifier_routine_no_sampelWeight(Xfeat,y_each_patient,selected_babies,label,classweight,C,gamma,\
                                       ChoosenKind,SamplingMeth,probability_threshold,SVMtype,strategie):
#### TRAIN CLASSIFIER
    meanaccLOO=[];accLOO=[];testsubject=[];tpr_mean=[];counter=0;
    mean_tpr = 0.0;mean_fpr = np.linspace(0, 1, 100)
    F1_macro_collect=[];F1_micro_collect=[];F1_weight_collect=[];F1_all_collect=[];K_collect=[]
    preliminaryK=zeros(len(probthres_Grid))
    
    #CREATING TEST AND TRAIN SETS
    for j in range(len(selected_babies)-1):
        print('.' , sep=' ', end='', flush=True)      # print in one line   
        Selected_training=list(delete(selected_babies,selected_babies[j]))# Babies to train on; j-1 as index starts with 0
        Selected_test=selected_babies[j]#summation-sum(Selected_training) #Baby to test on
        testsubject.append(Selected_test)
        X_train= [Xfeat[np.where(np.array(selected_babies)==k)[0][0]] for k in Selected_training] # combine only babies to train on in list
        y_train=[y_each_patient[np.where(np.array(selected_babies)==k)[0][0]] for k in Selected_training]
        X_train= vstack(X_train) # mergin the data from each list element into one matrix 
        X_test=Xfeat[Selected_test]
        y_train=vstack(y_train)
        y_test=y_each_patient[Selected_test]
        y_old=y_train[:]
    
    #SAMPLING TO EQUALIZE CLASS IMBALANCE
    
        X_train,y_train=cmplx_Oversampling(X_train,y_train,ChoosenKind,SamplingMeth,label)
        
    #CALCULATE THE WEIGHTS DUE TO CLASS IMBALANCE
        class_weight='balanced'
        classlabels=ravel(y_old) # y_test has to be a 1d array for compute_class_weight       
        
        # Now test if all labels are actually in the data. Otheriwse error with compute_class_weight. If not make the found labels the newe labels. If the new label is 1 then classsification does not work, therefore skip class_weigth , therefore CW       
        if (classweight==1) and len(unique(classlabels))==len(label):
            cW=compute_class_weight(class_weight, label, classlabels)
            cWdict=dict(zip(label,cW))#the class weight need to be a dictionarry of the form:{class_label : value}
            CW=1
        elif(classweight==1) and len(unique(classlabels))!=len(label):
            CW_label=unique(classlabels) #which values arein an array
            if len(CW_label)==1:                
                print('classweight config skiped once as only one class exist')
                CW=0
            else:
                print('used labels are:',CW_label, 'instead of:',label)            
                cW=compute_class_weight(class_weight, CW_label, classlabels)
                cWdict=dict(zip(label,cW))#the class weight need to be a dictionarry of the form:{class_label : value}
                CW=1
                                           
    #THE SVM
        if SVMtype=='Linear':              
               if (classweight==1) and CW==1: 
                    clf = svm.LinearSVC( C=C, class_weight=cWdict, random_state=42, multi_class=strategie )
               else:
                    clf = svm.LinearSVC( C=C,random_state=42, multi_class=strategie )  
        elif SVMtype=='Kernel':                    
               if (classweight==1) and CW==1: 
                    clf = svm.SVC(kernel='rbf',gamma=gamma, C=C, class_weight=cWdict,cache_size=500, probability=True, random_state=42)
               else:
                    clf = svm.SVC(kernel='rbf',gamma=gamma, C=C,cache_size=500, probability=True, random_state=42)
            

#Performance analysis        
#        sys.exit('Jan werth')
        if len(label)<2:
            print("please use at least two labels")
            break
            

 # F1 Kappa
        else: 
               prediction=clf.fit(X_train,y_train.ravel()).predict(X_test) # prediction decide on 0.5 proability which class to take
               probs=(clf.fit(X_train,y_train.ravel()).predict_proba(X_test)) # with the calculated probabilities we can choose our own threshold
               if probability_threshold: # prediction takes always proability 0.5 to deside. Here we deside based on other way lower proabilities. deciding if any other than AS has slightly elevated probabilities
                      for k in range(len(probthres_Grid)): # try differnt Trhesholds for the probability
                             preliminary_pred=copy(prediction[:])
                             probthres=probthres_Grid[k]
                             for i in range(len(probs)):
                                    if len(label)>2:
                                           if any(probs[i,1:]>=probthres): #IF THE PROBABILITY IS HIGHER THAN ... USE THAT CLASS INSTEAD OF AS
                                                  highprob=np.argmax(probs[i,1:]) # otherwise search for max prob of the labels other than AS
                                                  preliminary_pred[i]=label[highprob+1]# change the label in predictions to the new found label; +1 as we cut the array before by 1. Otherwise false index
                                                                                  
                                    elif(probs[i,1])>=probthres: # if we have only two labels searching for max does not work
                                           preliminary_pred[i]=label[1]# CHange the label in prediction to the second label
#To change klassifier for perfomance measure       
                             preliminaryK[k]=cohen_kappa_score(y_test.ravel(),preliminary_pred,labels=label) # Find the threshold where Kapaa gets max
#To change klassifier for perfomance measure  
                      maxK=preliminaryK.argmax(axis=0)
                      print('Used probability Thresh: %.2f' % probthres_Grid[maxK])
                      probthres=probthres_Grid[maxK] #repeat creating the predictions with the optimal probabilty threshold
                      for i in range(len(probs)):
                             if len(label)>2:
                                    if any(probs[i,1:]>=probthres): #IF THE PROBABILITY IS HIGHER THAN ... USE THAT CLASS INSTEAD OF AS
                                           highprob=np.argmax(probs[i,1:]) # otherwise search for max prob of the labels other than AS
                                           prediction[i]=label[highprob+1]# change the label in predictions to the new found label; +1 as we cut the array before by 1. Otherwise false index
                                                                           
                             elif(probs[i,1])>=probthres: # if we have only two labels searching for max does not work
                                    prediction[i]=label[1]# CHange the label in prediction to the second label
        
                             
            

                             
    #           F1=f1_score(y_test, probas_[:, 1], labels=list(label), pos_label=2, average=None, sample_weight=None) 
        tmpf1_macro=f1_score(y_test.ravel(), prediction, average='macro')#, pos_label=None)
        tmpf1_micro=f1_score(y_test.ravel(), prediction,average='micro')
        tmpf1_weight=f1_score(y_test.ravel(), prediction,average='weighted')
        tmpf1_all=f1_score(y_test.ravel(), prediction,labels=label, average=None)#, pos_label=None)
            
        tmpK=cohen_kappa_score(y_test.ravel(),prediction,labels=label)
                     
        F1_macro_collect.append(tmpf1_macro);tmpf1_macro=[] 
        F1_micro_collect.append(tmpf1_micro);tmpf1_micro=[] 
        F1_weight_collect.append(tmpf1_weight);tmpf1_weight=[] 
        F1_all_collect.append(tmpf1_all);tmpf1_all=[] 
            
        K_collect.append(tmpK);tmpK=[]  
                             
    resultsF1_maco=mean(F1_macro_collect)
    resultsF1_micro=mean(F1_micro_collect)
    resultsF1_weight=mean(F1_weight_collect)
    resultsF1_all=mean(np.array(F1_all_collect), axis=0) 
    resultsK=mean(K_collect)    
             
    return resultsF1_maco,resultsK,resultsF1_micro,resultsF1_weight,resultsF1_all 
            #F1 returns with average='none' a F1 score for each label or macro=meaned


#%%
""""
VALIDATION
"""             
def Validate_with_classifier(Xfeat,y_each_patient,selected_babies,selected_test,label,classweight,C,gamma,\
                                       ChoosenKind,SamplingMeth,probability_threshold,SVMtype,strategie):
  
#### TRAIN CLASSIFIER
    meanaccLOO=[];accLOO=[];testsubject=[];tpr_mean=[];counter=0;
    mean_tpr = 0.0;mean_fpr = np.linspace(0, 1, 100)
    F1_macro_collect=[];F1_micro_collect=[];F1_weight_collect=[];F1_all_collect=[];K_collect=[]
    preliminaryK=zeros(len(probthres_Grid))

    #CREATING TEST AND TRAIN SETS
    Selected_training=selected_babies
    X_train= [Xfeat[np.where(np.array(selected_babies)==k)[0][0]] for k in Selected_training] # combine only babies to train on in list
    y_train=[y_each_patient[np.where(np.array(selected_babies)==k)[0][0]] for k in Selected_training]
    X_train= vstack(X_train) # mergin the data from each list element into one matrix 
    X_test=Xfeat[selected_test]
    y_train=vstack(y_train)
    y_test=y_each_patient[selected_test]
    y_old=y_train[:]

#SAMPLING TO EQUALIZE CLASS IMBALANCE
   
    X_train,y_train=cmplx_Oversampling(X_train,y_train,ChoosenKind,SamplingMeth,label)    
    
#CALCULATE THE WEIGHTS DUE TO CLASS IMBALANCE
    class_weight='balanced'
    classlabels=ravel(y_old) # y_test has to be a 1d array for compute_class_weight      
    
    # Now test if all labels are actually in the data. Otheriwse error with compute_class_weight. If not make the found labels the newe labels. If the new label is 1 then classsification does not work, therefore skip class_weigth , therefore CW       
    if (classweight==1) and len(unique(classlabels))==len(label):
        cW=compute_class_weight(class_weight, label, classlabels)
        cWdict=dict(zip(range(len(label)),cW))#the class weight need to be a dictionarry of the form:{class_label : value}
        CW=1
    elif(classweight==1) and len(unique(classlabels))!=len(label):
        CW_label=unique(classlabels) #which values arein an array
        if len(CW_label)==1:                
            print('classweight config skiped once as only one class exist')
            CW=0
        else:
            print('used labels are:',CW_label, 'instead of:',label)            
            cW=compute_class_weight(class_weight, CW_label, classlabels)
            cWdict=dict(zip(range(len(CW_label)),cW))#the class weight need to be a dictionarry of the form:{class_label : value}
            CW=1
            
    #THE SVM
    if SVMtype=='Linear':              
        if (classweight==1) and CW==1: 
             clf = svm.LinearSVC( C=C, class_weight=cWdict,cache_size=500, random_state=42, multi_class=strategie )
        else:
             clf = svm.LinearSVC( C=C,cache_size=500, random_state=42, multi_class=strategie )  
    elif SVMtype=='Kernel':                    
        if (classweight==1) and CW==1: 
             clf = svm.SVC(kernel='rbf',gamma=gamma, C=C, class_weight=cWdict,cache_size=500, probability=True, random_state=42)
        else:
             clf = svm.SVC(kernel='rbf',gamma=gamma, C=C,cache_size=500, probability=True, random_state=42)
        
#Performance analysis        
#        sys.exit('Jan werth')
    if len(label)<2:
        print("please use at least two labels")
        
 # F1 Kappa
    else: 
        prediction=clf.fit(X_train,y_train.ravel()).predict(X_test) # prediction decide on 0.5 proability which class to take
        probs=(clf.fit(X_train,y_train.ravel()).predict_proba(X_test)) # with the calculated probabilities we can choose our own threshold
        if probability_threshold: # prediction takes always proability 0.5 to deside. Here we deside based on other way lower proabilities. deciding if any other than AS has slightly elevated probabilities
               for k in range(len(probthres_Grid)): # try differnt Trhesholds for the probability
                      preliminary_pred=copy(prediction[:])
                      probthres=probthres_Grid[k]
                      for i in range(len(probs)):
                             if len(label)>2:
                                    if any(probs[i,1:]>=probthres): #IF THE PROBABILITY IS HIGHER THAN ... USE THAT CLASS INSTEAD OF AS
                                           highprob=np.argmax(probs[i,1:]) # otherwise search for max prob of the labels other than AS
                                           preliminary_pred[i]=label[highprob+1]# change the label in predictions to the new found label; +1 as we cut the array before by 1. Otherwise false index
                                                                                  
                             elif(probs[i,1])>=probthres: # if we have only two labels searching for max does not work
                                    preliminary_pred[i]=label[1]# CHange the label in prediction to the second label
#!!!!!!!! To change klassifier for perfomance measure       
                      preliminaryK[k]=cohen_kappa_score(y_test.ravel(),preliminary_pred,labels=label) # Find the threshold where Kapaa gets max
#!!!!!!!! To change klassifier for perfomance measure  
               maxK=preliminaryK.argmax(axis=0)
               print('Used probability Thresh: %.2f' % probthres_Grid[maxK])
               probthres=probthres_Grid[maxK] #repeat creating the predictions with the optimal probabilty threshold
               for i in range(len(probs)):
                      if len(label)>2:
                             if any(probs[i,1:]>=probthres): #IF THE PROBABILITY IS HIGHER THAN ... USE THAT CLASS INSTEAD OF AS
                                    highprob=np.argmax(probs[i,1:]) # otherwise search for max prob of the labels other than AS
                                    prediction[i]=label[highprob+1]# change the label in predictions to the new found label; +1 as we cut the array before by 1. Otherwise false index
                                                                           
                      elif(probs[i,1])>=probthres: # if we have only two labels searching for max does not work
                             prediction[i]=label[1]# CHange the label in prediction to the second label
                             
#        F1=f1_score(y_test, probas_[:, 1], labels=list(label), pos_label=2, average=None, sample_weight=None) 
        tmpf1_macro=f1_score(y_test.ravel(), prediction, average='macro')#, pos_label=None)
        tmpf1_micro=f1_score(y_test.ravel(), prediction,average='micro')
        tmpf1_weight=f1_score(y_test.ravel(), prediction,average='weighted')
        tmpf1_all=f1_score(y_test.ravel(), prediction,labels=label, average=None)#, pos_label=None)
        
        tmpK=cohen_kappa_score(y_test.ravel(),prediction,labels=label)
                 
        F1_macro_collect.append(tmpf1_macro);tmpf1_macro=[] 
        F1_micro_collect.append(tmpf1_micro);tmpf1_micro=[] 
        F1_weight_collect.append(tmpf1_weight);tmpf1_weight=[] 
        F1_all_collect.append(tmpf1_all);tmpf1_all=[] 
        
        K_collect.append(tmpK);tmpK=[]  
                         
        resultsF1_maco=mean(F1_macro_collect)
        resultsF1_micro=mean(F1_micro_collect)
        resultsF1_weight=mean(F1_weight_collect)
        resultsF1_all=mean(F1_all_collect)
        
        resultsK=mean(K_collect)    
                  
    return resultsF1_maco,resultsK,resultsF1_micro,resultsF1_weight,resultsF1_all 
        #F1 returns with average='none' a F1 score for each label or macro=meaned
###########################################################################################################    
########################################################################################################### 
#%%
'''
Random Forrest     
'''   
def Classifier_random_forest(Xfeat_test, Xfeat,y_each_patient_test, y_each_patient, selected_babies, \
                              selected_test, label,classweight, Used_classifier, drawing, lst, ChoosenKind,\
                              SamplingMeth,probability_threshold,ASprobLimit,N,crit,msl,deciding_performance_measure,dispinfo):
       

#### CREATING THE sampleweight FOR SELECTED BABIES  
#### TRAIN CLASSIFIER
    meanaccLOO=[];accLOO=[];testsubject=[];tpr_mean=[];counter=0;
    mean_tpr = 0.0;mean_fpr = np.linspace(0, 1, 100)
    F1_macro_collect=[];F1_micro_collect=[];F1_weight_collect=[];F1_all_collect=[];K_collect=[]
    preliminaryK=zeros(len(probthres_Grid))

#CREATING TEST AND TRAIN SETS
    Selected_training=selected_babies
    X_train= [Xfeat[np.where(np.array(selected_babies)==k)[0][0]] for k in Selected_training] # combine only babies to train on in list
    y_train=[y_each_patient[np.where(np.array(selected_babies)==k)[0][0]] for k in Selected_training] 
    X_test=Xfeat_test[selected_test]
    y_test=y_each_patient_test[selected_test]
    X_train= vstack(X_train) # mergin the data from each list element into one matrix 
    y_train=vstack(y_train)
    y_old=y_train[:]

#SAMPLING TO EQUALIZE CLASS IMBALANCE
   
    X_train,y_train=cmplx_Oversampling(X_train,y_train,ChoosenKind,SamplingMeth,label)    
    
#CALCULATE THE WEIGHTS DUE TO CLASS IMBALANCE
    class_weight='balanced'
    classlabels=ravel(y_old) # y_test has to be a 1d array for compute_class_weight       
    # Now test if all labels are actually in the data. Otheriwse error with compute_class_weight. If not make the found labels the newe labels. If the new label is 1 then classsification does not work, therefore skip class_weigth , therefore CW       
    if (classweight==1) and len(unique(classlabels))==len(label):
        cW=compute_class_weight(class_weight, label, classlabels)
        cWdict=dict(zip(label,cW))#the class weight need to be a dictionarry of the form:{class_label : value}
        CW=1
    elif(classweight==1) and len(unique(classlabels))!=len(label):
        CW_label=unique(classlabels) #which values arein an array
        if len(CW_label)==1:        
            if dispinfo:
                   print('classweight config skiped once as only one class exist')
            CW=0
        else:
            if dispinfo:
                   print('used labels are:',CW_label, 'instead of:',label)            
            cW=compute_class_weight(class_weight, CW_label, classlabels)
            cWdict=dict(zip(label,cW))#the class weight need to be a dictionarry of the form:{class_label : value}
            CW=1
    if dispinfo:       
           disp(cWdict)       
            
            
#The Random Forest / Extreme Random Forest / Gradiant boosting

    if Used_classifier=='TR':
        if (classweight==1) and CW==1: 
             clf = tree.DecisionTreeClassifier(criterion=crit, splitter="best", max_depth=None,\
                                               min_samples_split=2, min_samples_leaf=msl, \
                                               min_weight_fraction_leaf=0.0, max_features=None, \
                                               random_state=42, max_leaf_nodes=None, min_impurity_decrease=0.0,\
                                               min_impurity_split=None, class_weight=cWdict, presort=False)
                           
        else:
             clf = tree.DecisionTreeClassifier(criterion=crit, splitter="best", max_depth=None,\
                                               min_samples_split=2, min_samples_leaf=msl, \
                                               min_weight_fraction_leaf=0.0, max_features=None, \
                                               random_state=42, max_leaf_nodes=None, min_impurity_decrease=0.0,\
                                               min_impurity_split=None,  presort=False)   
    
    
    if Used_classifier=='RF':
        if (classweight==1) and CW==1: 
             clf = RandomForestClassifier(n_estimators=N, criterion=crit, max_depth=None, \
                                          min_samples_split=2, min_samples_leaf=msl, min_weight_fraction_leaf=0.0,\
                                          max_features="auto", max_leaf_nodes=None, min_impurity_decrease=0.0,\
                                          min_impurity_split=None, bootstrap=True, oob_score=False,\
                                          n_jobs=1, random_state=42, verbose=0, warm_start=False,\
                                          class_weight=cWdict)
                           
        else:
             clf = RandomForestClassifier(n_estimators=N, criterion=crit, max_depth=None, \
                                          min_samples_split=2, min_samples_leaf=msl, min_weight_fraction_leaf=0.0,\
                                          max_features="auto", max_leaf_nodes=None, min_impurity_decrease=0.0,\
                                          min_impurity_split=None, bootstrap=True, oob_score=False,\
                                          n_jobs=1, random_state=42, verbose=0, warm_start=False,\
                                          )
    elif Used_classifier=='ERF':         
        if (classweight==1) and CW==1: 
             clf = ExtraTreesClassifier(n_estimators=N, criterion=crit, max_depth=None,\
                                          min_samples_split=2, min_samples_leaf=msl, \
                                          max_features="auto", max_leaf_nodes=None, min_impurity_decrease=0.0,\
                                          min_impurity_split=None, bootstrap=True, oob_score=False,\
                                          n_jobs=1, random_state=42, verbose=0, warm_start=False,\
                                          class_weight=cWdict)
        else:
             clf = ExtraTreesClassifier(n_estimators=N, criterion=crit, max_depth=None,\
                                          min_samples_split=2, min_samples_leaf=msl, min_weight_fraction_leaf=0.0,\
                                          max_features="auto", max_leaf_nodes=None, min_impurity_decrease=0.0,\
                                          min_impurity_split=None, bootstrap=True, oob_score=False,\
                                          n_jobs=1, random_state=42, verbose=0, warm_start=False,\
                                          )
    elif Used_classifier=='GB':
      clf = GradientBoostingClassifier(loss="deviance", learning_rate=0.1, n_estimators=1000, subsample=1, \
                          criterion='friedman_mse', min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0,\
                          max_depth=30, min_impurity_decrease=0.0, min_impurity_split=None, init=None, \
                          random_state=42, max_features=None, verbose=0, max_leaf_nodes=None, warm_start=False, presort='auto')
      
    elif Used_classifier=='LR':
           clf=LinearRegression(fit_intercept=True, normalize=False, copy_X=True, n_jobs=1)
        
                    
       
        
#Performance analysis        
#        sys.exit('Jan werth')
    if len(label)<2:
        print("please use at least two labels")
        
 # F1 Kappa
    else: 
        prediction=clf.fit(X_train,y_train.ravel()).predict(X_test) # prediction decide on 0.5 proability which class to take
        probs=(clf.fit(X_train,y_train.ravel()).predict_proba(X_test)) # with the calculated probabilities we can choose our own threshold
        if probability_threshold: # prediction takes always proability 0.5 to deside. Here we deside based on other way lower proabilities. deciding if any other than AS has slightly elevated probabilities
               for k in range(len(probthres_Grid)): # try differnt Trhesholds for the probability
                      preliminary_pred=copy(prediction[:])
                      probthres=probthres_Grid[k]
                      for i in range(len(probs)):
                             if len(label)==3:
                                    if any(probs[i,1:]>=probthres) and probs[i,0]<ASprobLimit[0]: #IF THE PROBABILITY IS HIGHER THAN ... USE THAT CLASS INSTEAD OF AS. But if AS is over ~0.7 still take AS
                                           highprob=np.argmax(probs[i,1:]) # otherwise search for max prob of the labels other than AS
                                           preliminary_pred[i]=label[highprob+1]# change the label in predictions to the new found label; +1 as we cut the array before by 1. Otherwise false index
                             if len(label)>3:
                                    if any(probs[i,1:]>=probthres) and probs[i,0]<ASprobLimit[1]: #IF THE PROBABILITY IS HIGHER THAN ... USE THAT CLASS INSTEAD OF AS. But if AS is over ~0.7 still take AS
                                           highprob=np.argmax(probs[i,1:]) # otherwise search for max prob of the labels other than AS
                                           preliminary_pred[i]=label[highprob+1]# change the label in predictions to the new found label; +1 as we cut the array before by 1. Otherwise false index
                                                                                                                                                                    
                             elif(probs[i,1])>=probthres: # if we have only two labels searching for max does not work
                                    preliminary_pred[i]=label[1]# CHange the label in prediction to the second label
#!!!!!!!! To change klassifier for perfomance measure 
                      if deciding_performance_measure=='Kappa':              
                             preliminaryK[k]=cohen_kappa_score(y_test.ravel(),preliminary_pred,labels=label) # Find the threshold where Kapa gets max
                      elif deciding_performance_measure=='F1_second_label':
                             preliminaryK[k]=f1_score(y_test.ravel(), preliminary_pred,labels=label, average=None)[1]
                      elif deciding_performance_measure=='F1_third_label':
                             preliminaryK[k]=f1_score(y_test.ravel(), preliminary_pred,labels=label, average=None)[2]
                      elif deciding_performance_measure=='F1_fourth_label':
                             preliminaryK[k]=f1_score(y_test.ravel(), preliminary_pred,labels=label, average=None)[3]       
#!!!!!!!! To change klassifier for perfomance measure  
               maxK=preliminaryK.argmax(axis=0)
               if dispinfo:
                      print('Used probability Thresh: %.2f' % probthres_Grid[maxK])
               probthres=probthres_Grid[maxK] #repeat creating the predictions with the optimal probabilty threshold
               for i in range(len(probs)):
                      if len(label)==3:
                             if any(probs[i,1:]>=probthres) and probs[i,0]<ASprobLimit[0]: #IF THE PROBABILITY IS HIGHER THAN ... USE THAT CLASS INSTEAD OF AS. But if AS is over ~0.7 still take AS
                                    highprob=np.argmax(probs[i,1:]) # otherwise search for max prob of the labels other than AS
                                    prediction[i]=label[highprob+1]# change the label in predictions to the new found label; +1 as we cut the array before by 1. Otherwise false index
                      if len(label)>3:
                             if any(probs[i,1:]>=probthres) and probs[i,0]<ASprobLimit[1]: #IF THE PROBABILITY IS HIGHER THAN ... USE THAT CLASS INSTEAD OF AS. But if AS is over ~0.7 still take AS
                                    highprob=np.argmax(probs[i,1:]) # otherwise search for max prob of the labels other than AS
                                    prediction[i]=label[highprob+1]# cha                                                                           
                      elif(probs[i,1])>=probthres: # if we have only two labels searching for max does not work
                             prediction[i]=label[1]# CHange the label in prediction to the second label
        
        scoring=clf.score(X_test, y_test.ravel(),sample_weight=None)
        Fimportances=clf.feature_importances_
        if Used_classifier!='GB':
               Dpath=clf.decision_path(X_train)        
        
        resultsF1_macro=f1_score(y_test.ravel(), prediction, average='macro')#, pos_label=None)
        resultsF1_micro=f1_score(y_test.ravel(), prediction,average='micro')
        resultsF1_weight=f1_score(y_test.ravel(), prediction,average='weighted')
        resultsF1_all=f1_score(y_test.ravel(), prediction,labels=label, average=None)#, pos_label=None)
        
        resultsK=cohen_kappa_score(y_test.ravel(),prediction,labels=label)
        
        if drawing and Used_classifier=='TR':
               import graphviz                     
               from Loading_5min_mat_files_cECG import Class_dict,features_dict
               usedfeatures=list((features_dict[k]) for k in lst) #create a dict only with the usedfeatures in lst out of all which are in features_dict
               usedlabels=list((Class_dict[k]) for k in label) #create a dict only with the usedfeatures in lst out of all which are in features_dict
        
               with open("RF.txt", "w") as f:
                      f = tree.export_graphviz(clf, out_file=f,feature_names=usedfeatures,class_names=usedlabels,filled=True, rounded=True )
#               with open("RF.dot", "w") as f:
#                      f = tree.export_graphviz(clf, out_file=f)
               with open("RF.svc", "w") as f:
                      f = tree.export_graphviz(clf, out_file=f)
#               dot -Tpdf RF.dot -o RF.pdf
#               open -a preview RF.pdf

               
#               dot_data = tree.export_graphviz(clf, out_file=None) 
#               graph = graphviz.Source(dot_data) 
#               graph.render("Jan") 
#              
#               dot_data = tree.export_graphviz(clf, out_file=None, 
#                                        feature_names=usedfeatures,  
#                                        class_names=usedlabels,  
#                                        filled=True, rounded=True,  
#                                        special_characters=True)  
#               graph = graphviz.Source(dot_data)  
#               graph 
#                 
        
                  
    return resultsF1_macro,resultsK,resultsF1_micro,resultsF1_weight,resultsF1_all,Fimportances,scoring,prediction,probs
        #F1 returns with average='none' a F1 score for each label or macro=meaned
        
