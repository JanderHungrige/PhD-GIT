# -*- coding: utf-8 -*-
"""
Created on Mon Sep 25 11:34:48 2017

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
from sklearn.metrics import accuracy_score 
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import f1_score
from sklearn import svm, cross_validation
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.linear_model import Perceptron

def Classifier_routine_no_sampelWeight(Xfeat,y_each_patient,selected_babies,label,classweight):
    summation=sum(selected_babies)
#### TRAIN CLASSIFIER
    meanaccLOO=[];accLOO=[];testsubject=[];tpr_mean=[];counter=0;
    mean_tpr = 0.0;mean_fpr = np.linspace(0, 1, 100)

    #CREATING TEST AND TRAIN SETS
    for j in range(len(selected_babies)):
        Selected_training=list(delete(selected_babies,selected_babies[j-1]))# Babies to train on; j-1 as index starts with 0
        Selected_test=selected_babies[j]#summation-sum(Selected_training) #Baby to test on
        testsubject.append(Selected_test)
        X_train= [Xfeat[np.where(np.array(selected_babies)==k)[0][0]] for k in Selected_training] # combine only babies to train on in list
        y_train=[y_each_patient[np.where(np.array(selected_babies)==k)[0][0]] for k in Selected_training]
        X_train= vstack(X_train) # mergin the data from each list element into one matrix 
        X_test=Xfeat[Selected_test]
        y_train=vstack(y_train)
        y_test=y_each_patient[Selected_test]
        
    #CALCULATE THE WEIGHTS DUE TO CLASS IMBALANCE
        class_weight='balanced'
        classes=label
        classlabels=ravel(y_test) # y_test has to be a 1d array for compute_class_weight
        if (classweight==1):
            cW=compute_class_weight(class_weight, classes, classlabels)
            cWdict={1:cW[0]};cWdict={2:cW[1]} #the class weight need to be a dictionarry of the form:{class_label : value}
            
    #THE SVM
        if (classweight==1):
             clf = svm.SVC(kernel='rbf', class_weight=cWdict, probability=True, random_state=42)
        else:
            clf = svm.SVC(kernel='rbf',gamma=0.2, C=1, probability=True, random_state=42)
            
        probas_=clf.fit(X_train,y_train).predict_proba(X_test)  
        
    # ROC and AUC
        fpr, tpr, thresholds = roc_curve(y_test, probas_[:, 1], pos_label=2)
        if isnan(sum(tpr))== False and isnan(sum(fpr))==False:
            mean_tpr += interp(mean_fpr, fpr, tpr)
            mean_tpr[0] = 0.0
            counter+=1
    
        roc_auc = auc(fpr, tpr)
        
    # F1
        if (classweight==1):
            sklearn.metrics.f1_score(y_test, probas_[:, 1], labels=None, pos_label=1, average=’None’, sample_weight=cWdict) 
        else:
            sklearn.metrics.f1_score(y_test, probas_[:, 1], labels=None, pos_label=1, average=’None’, sample_weight=None) 
            
            
    
    mean_tpr /= counter#len(selected_babies)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr) #Store all mean AUC for each combiation 
    #collected_mean_auc.append(mean_auc) Do that outside of fucntion, otherwise always set to []
    return mean_auc
    
###########################################################################################################    
###########################################################################################################    
"""
With Sample weights
"""
# input: 
# Xfeat: the Feature values as a list per patient
# y_each_patient: the annotations for each Feature value as a list
# Seleted babies: The patients the classifier should use for train and testing as a list [0,1,2,3,4...] 
# SampleWeigth: The sample weights calculated for the class imbalance from the file compute_class_weight.py
# label: THe class labels as array e.g. array([1,2]) 1=AS 2=QS ,...
# classweight: just 1 or 0. if classweigths should be calculated outomatically use 1 otherhwise no classweight calculation (DO NOT CONFUSE WITH SAMPLEWEIGHT)


# output: the collected mean AUC of each itteration of train and testing with leave one patient out


def Classifier_routine_with_sampelWeight(Xfeat,y_each_patient,selected_babies,sampleWeights,label,classweight):
    summation=sum(selected_babies)
#### CREATING THE sampleweight FOR SELECTED BABIES  
    smplwght=[val[idx[sb],:] for sb, val in enumerate(sampleWeights) if sb in range(len(selected_babies))] # selecting the sampleweight

#### TRAIN CLASSIFIER
    meanaccLOO=[];accLOO=[];testsubject=[];tpr_mean=[];counter=0;
    mean_tpr = 0.0;mean_fpr = np.linspace(0, 1, 100)

    #CREATING TEST AND TRAIN SETS
    for j in range(len(selected_babies)):
        Selected_training=delete(selected_babies,selected_babies[j])# Babies to train on 0-7
        Selected_test=summation-sum(Selected_training) #Babie to test on
        testsubject.append(Selected_test)
        X_train= [Xfeat[k] for k in Selected_training] # combine only babies to train on in list
        y_train=[y_each_patient[k] for k in Selected_training]
        X_train= vstack(X_train) # mergin the data from each list element into one matrix 
        X_test=Xfeat[Selected_test]
        y_train=vstack(y_train)
        y_test=y_each_patient[Selected_test]
        
    #CALCULATE THE WEIGHTS DUE TO CLASS IMBALANCE
        class_weight='balanced'
        classes=label
        classlabels=ravel(y_test) # y_test has to be a 1d array for compute_class_weight
        if (classweight==1):
            cW=compute_class_weight(class_weight, classes, classlabels)
            cWdict={1:cW[0]};cWdict={2:cW[1]} #the class weight need to be a dictionarry of the form:{class_label : value}
            
    #THE SVM
        if (classweight==1) and (weighting_each_sample==0):# as baby 7 does not have two classes, it is not unbalnced
             clf = svm.SVC(kernel='rbf', class_weight=cWdict, probability=True, random_state=42)
        elif (classweight==0) and (weighting_each_sample==0):
            clf = svm.SVC(kernel='rbf',gamma=0.2, C=1, probability=True, random_state=42)
        elif(classweight==0) and (weighting_each_sample==1):
             clf = svm.SVC(kernel='rbf',gamma=0.2, C=1, sample_weight=smplwght, probability=True, random_state=42)
        elif(classweight==1) and (weighting_each_sample==1):
             clf = svm.SVC(kernel='rbf', class_weight=cWdict, sample_weight=smplwght, probability=True, random_state=42)
                   
        probas_=clf.fit(X_train,y_train).predict_proba(X_test)  
        
    # ROC and AUC
        fpr, tpr, thresholds = roc_curve(y_test, probas_[:, 1], pos_label=2)
        if isnan(sum(tpr))== False and isnan(sum(fpr))==False:
            mean_tpr += interp(mean_fpr, fpr, tpr)
            mean_tpr[0] = 0.0
            counter+=1
    
        roc_auc = auc(fpr, tpr)
    
    mean_tpr /= counter#len(selected_babies)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr) #Store all mean AUC for each combiation 
    #collected_mean_auc.append(mean_auc) Do that outside of fucntion, otherwise always set to []
    return mean_auc
    