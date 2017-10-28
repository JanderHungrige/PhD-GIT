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
from sklearn.metrics import accuracy_score 
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import cohen_kappa_score
from sklearn import svm, cross_validation
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.linear_model import Perceptron
from sklearn.metrics import f1_score
from compute_class_weight import*

import pdb# use pdb.set_trace() as breakpoint
"""
Without Sample weights
"""
def Classifier_routine_no_sampelWeight(Xfeat,y_each_patient,selected_babies,label,classweight,C,gamma):
#### TRAIN CLASSIFIER
    meanaccLOO=[];accLOO=[];testsubject=[];tpr_mean=[];counter=0;
    mean_tpr = 0.0;mean_fpr = np.linspace(0, 1, 100)
    F1_macro_collect=[];F1_micro_collect=[];F1_weight_collect=[];F1_all_collect=[];K_collect=[]

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
        
    #CALCULATE THE WEIGHTS DUE TO CLASS IMBALANCE
        class_weight='balanced'
        classlabels=ravel(y_test) # y_test has to be a 1d array for compute_class_weight       
        
        # Now test if all labels are actually in the data. Otheriwse error with compute_class_weight. If not make the found labels the newe labels. If the new label is 1 then classsification does not work, therefore skip class_weigth , therefore CW       
        if (classweight==1) and len(unique(classlabels))==len(label):
            cW=compute_class_weight(class_weight, label, classlabels)
            cWdict={1:cW[0]};cWdict={2:cW[1]} #the class weight need to be a dictionarry of the form:{class_label : value}
            CW=1
        elif(classweight==1) and len(unique(classlabels))!=len(label):
            CW_label=unique(classlabels) #which values arein an array
            if len(CW_label)==1:                
                print('classweight config skiped once as only one class exist')
                CW=0
            else:
                print('used labels are:',CW_label, 'instead of:',label)            
                cW=compute_class_weight(class_weight, CW_label, classlabels)
                cWdict={1:cW[0]};cWdict={2:cW[1]} #the class weight need to be a dictionarry of the form:{class_label : value}            
                CW=1
            
    #THE SVM
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
            prediction=clf.fit(X_train,y_train.ravel()).predict(X_test)
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


def Classifier_routine_with_sampleWeight(Xfeat,y_each_patient,selected_babies,label,sampleWeights,classweight):
#### CREATING THE sampleweight FOR SELECTED BABIES  
    smplwght=[val[idx[sb],:] for sb, val in enumerate(sampleWeights) if sb in range(len(selected_babies))] # selecting the sampleweight

#### TRAIN CLASSIFIER
    meanaccLOO=[];accLOO=[];testsubject=[];tpr_mean=[];counter=0;
    mean_tpr = 0.0;mean_fpr = np.linspace(0, 1, 100)
    F1_macro_collect=[];F1_micro_collect=[];F1_weight_collect=[];F1_all_collect=[];K_collect=[]
    #CREATING TEST AND TRAIN SETS
    for j in range(len(selected_babies)):
        print('.' , sep=' ', end='', flush=True)          
        Selected_training=delete(selected_babies,selected_babies[j])# Babies to train on 0-7
        Selected_validate=summation-sum(Selected_training) #Babie to test on
        testsubject.append(Selected_validate)
        X_train= [Xfeat[k] for k in Selected_training] # combine only babies to train on in list
        y_train=[y_each_patient[k] for k in Selected_training]
        X_train= vstack(X_train) # mergin the data from each list element into one matrix 
        X_validate=Xfeat[Selected_validate]
        y_train=vstack(y_train)
        y_validate=y_each_patient[Selected_validate]
        
    #CALCULATE THE WEIGHTS DUE TO CLASS IMBALANCE
        class_weight='balanced'
        classlabels=ravel(y_validate) # y_validate has to be a 1d array for compute_class_weight       
        
        # Now test if all labels are actually in the data. Otheriwse error with compute_class_weight. If not make the found labels the newe labels. If the new label is 1 then classsification does not work, therefore skip class_weigth , therefore CW       
        if (classweight==1) and len(unique(classlabels))==len(label):
            cW=compute_class_weight(class_weight, label, classlabels)
            cWdict={1:cW[0]};cWdict={2:cW[1]} #the class weight need to be a dictionarry of the form:{class_label : value}
            CW=1
        elif(classweight==1) and len(unique(classlabels))!=len(label):
            CW_label=unique(classlabels) #which values arein an array
            if len(CW_label)==1:                
                print('classweight config skiped once as only one class exist')
                CW=0
            else:
                print('used labels are:',CW_label, 'instead of:',label)            
                cW=compute_class_weight(class_weight, CW_label, classlabels)
                cWdict={1:cW[0]};cWdict={2:cW[1]} #the class weight need to be a dictionarry of the form:{class_label : value}            
                CW=1
            
    #THE SVM
        if (classweight==1) and CW==1 and (sampleWeights==0):
             clf = svm.SVC(kernel='rbf', class_weight=cWdict, probability=True, random_state=42)
        elif(classweight==0) and (sampleWeights==1):
             clf = svm.SVC(kernel='rbf',gamma=0.2, C=1, sample_weight=smplwght, probability=True, random_state=42)
        elif(classweight==1) and CW==1 and (sampleWeights==1): 
             clf = svm.SVC(kernel='rbf', class_weight=cWdict, sample_weight=smplwght, probability=True, random_state=42)
        else:
            clf = svm.SVC(kernel='rbf',gamma=0.2, C=1, probability=True, random_state=42)
            
     
        probas_=clf.fit(X_train,y_train).predict_proba(X_validate)  
        
#Performance analysis        
#        sys.exit('Jan werth')
        if len(label)<2:
            print("please use at least two labels")
            
 # F1 Kappa
        else:        
            prediction=clf.fit(X_train,y_train.ravel()).predict(X_validate)
    #           F1=f1_score(y_validate, probas_[:, 1], labels=list(label), pos_label=2, average=None, sample_weight=None) 
            tmpf1_macro=f1_score(y_validate.ravel(), prediction, average='macro')#, pos_label=None)
            tmpf1_micro=f1_score(y_validate.ravel(), prediction,average='micro')
            tmpf1_weight=f1_score(y_validate.ravel(), prediction,average='weighted')
            tmpf1_all=f1_score(y_validate.ravel(), prediction,labels=label, average=None)#, pos_label=None)
            
            tmpK=cohen_kappa_score(y_validate.ravel(),prediction,labels=label)
                     
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
                      
    return resultsF1_maco,resultsK,resultsF1_micro,resultsF1_weight,resultsF1_all #F1
            #F1 returns with average='none' a F1 score for each label or macro=meaned
###########################################################################################################    
###########################################################################################################   
""""
VALIDATION
"""             
def Validate_with_classifier(Xfeat,y_each_patient,selected_babies,selected_test,label,classweight,C,gamma):
  
#### TRAIN CLASSIFIER
    meanaccLOO=[];accLOO=[];testsubject=[];tpr_mean=[];counter=0;
    mean_tpr = 0.0;mean_fpr = np.linspace(0, 1, 100)
    F1_macro_collect=[];F1_micro_collect=[];F1_weight_collect=[];F1_all_collect=[];K_collect=[]


    #CREATING TEST AND TRAIN SETS
    Selected_training=selected_babies
    X_train= [Xfeat[np.where(np.array(selected_babies)==k)[0][0]] for k in Selected_training] # combine only babies to train on in list
    y_train=[y_each_patient[np.where(np.array(selected_babies)==k)[0][0]] for k in Selected_training]
    X_train= vstack(X_train) # mergin the data from each list element into one matrix 
    X_test=Xfeat[selected_test]
    y_train=vstack(y_train)
    y_test=y_each_patient[selected_test]
    
#CALCULATE THE WEIGHTS DUE TO CLASS IMBALANCE
    class_weight='balanced'
    classlabels=ravel(y_test) # y_test has to be a 1d array for compute_class_weight       
    
    # Now test if all labels are actually in the data. Otheriwse error with compute_class_weight. If not make the found labels the newe labels. If the new label is 1 then classsification does not work, therefore skip class_weigth , therefore CW       
    if (classweight==1) and len(unique(classlabels))==len(label):
        cW=compute_class_weight(class_weight, label, classlabels)
        cWdict={1:cW[0]};cWdict={2:cW[1]} #the class weight need to be a dictionarry of the form:{class_label : value}
        CW=1
    elif(classweight==1) and len(unique(classlabels))!=len(label):
        CW_label=unique(classlabels) #which values arein an array
        if len(CW_label)==1:                
            print('classweight config skiped once as only one class exist')
            CW=0
        else:
            print('used labels are:',CW_label, 'instead of:',label)            
            cW=compute_class_weight(class_weight, CW_label, classlabels)
            cWdict={1:cW[0]};cWdict={2:cW[1]} #the class weight need to be a dictionarry of the form:{class_label : value}            
            CW=1
            
    #THE SVM
    if (classweight==1) and CW==1: 
         clf = svm.SVC(kernel='rbf',gamma=gamma, C=C, class_weight=cWdict, cache_size=500,probability=True, random_state=42)
    else:
         clf = svm.SVC(kernel='rbf',gamma=gamma, C=C, probability=True,cache_size=500, random_state=42)
        
#Performance analysis        
#        sys.exit('Jan werth')
    if len(label)<2:
        print("please use at least two labels")
        
 # F1 Kappa
    else:        
        prediction=clf.fit(X_train,y_train.ravel()).predict(X_test)
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
        resultsF1_all=mean(F1_all_collect)
        
        resultsK=mean(K_collect)    
                  
    return resultsF1_maco,resultsK,resultsF1_micro,resultsF1_weight,resultsF1_all 
        #F1 returns with average='none' a F1 score for each label or macro=meaned
'''
Random Forrest     
'''   
def Classifier_random_forest(Xfeat,y_each_patient,selected_babies,selected_test,label,classweight,Used_classifier,drawing):
#### CREATING THE sampleweight FOR SELECTED BABIES  
#### TRAIN CLASSIFIER
    meanaccLOO=[];accLOO=[];testsubject=[];tpr_mean=[];counter=0;
    mean_tpr = 0.0;mean_fpr = np.linspace(0, 1, 100)
    F1_macro_collect=[];F1_micro_collect=[];F1_weight_collect=[];F1_all_collect=[];K_collect=[]


    #CREATING TEST AND TRAIN SETS
    Selected_training=selected_babies
    X_train= [Xfeat[np.where(np.array(selected_babies)==k)[0][0]] for k in Selected_training] # combine only babies to train on in list
    y_train=[y_each_patient[np.where(np.array(selected_babies)==k)[0][0]] for k in Selected_training]
    X_train= vstack(X_train) # mergin the data from each list element into one matrix 
    X_test=Xfeat[selected_test]
    y_train=vstack(y_train)
    y_test=y_each_patient[selected_test]
    
#CALCULATE THE WEIGHTS DUE TO CLASS IMBALANCE
    class_weight='balanced'
    classlabels=ravel(y_test) # y_test has to be a 1d array for compute_class_weight       
    
    # Now test if all labels are actually in the data. Otheriwse error with compute_class_weight. If not make the found labels the newe labels. If the new label is 1 then classsification does not work, therefore skip class_weigth , therefore CW       
    if (classweight==1) and len(unique(classlabels))==len(label):
        cW=compute_class_weight(class_weight, label, classlabels)
        cWdict={1:cW[0]};cWdict={2:cW[1]} #the class weight need to be a dictionarry of the form:{class_label : value}
        CW=1
    elif(classweight==1) and len(unique(classlabels))!=len(label):
        CW_label=unique(classlabels) #which values arein an array
        if len(CW_label)==1:                
            print('classweight config skiped once as only one class exist')
            CW=0
        else:
            print('used labels are:',CW_label, 'instead of:',label)            
            cW=compute_class_weight(class_weight, CW_label, classlabels)
            cWdict={1:cW[0]};cWdict={2:cW[1]} #the class weight need to be a dictionarry of the form:{class_label : value}            
            CW=1
    #The Random Forest / Extreme Random Forest
    
    tree.DecisionTreeClassifier
    if Used_classifier=='TR':
        if (classweight==1) and CW==1: 
             clf = tree.DecisionTreeClassifier(criterion="gini", splitter="best", max_depth=None,\
                                               min_samples_split=2, min_samples_leaf=1, \
                                               min_weight_fraction_leaf=0.0, max_features=None, \
                                               random_state=None, max_leaf_nodes=None, min_impurity_decrease=0.0,\
                                               min_impurity_split=None, class_weight=cWdict, presort=False)
                           
        else:
             clf = tree.DecisionTreeClassifier(criterion="gini", splitter="best", max_depth=None,\
                                               min_samples_split=2, min_samples_leaf=1, \
                                               min_weight_fraction_leaf=0.0, max_features=None, \
                                               random_state=None, max_leaf_nodes=None, min_impurity_decrease=0.0,\
                                               min_impurity_split=None, class_weight=None, presort=False)   
    
    
    if Used_classifier=='RF':
        if (classweight==1) and CW==1: 
             clf = RandomForestClassifier(n_estimators=10, criterion="gini", max_depth=None, \
                                          min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0,\
                                          max_features="auto", max_leaf_nodes=None, min_impurity_decrease=0.0,\
                                          min_impurity_split=None, bootstrap=True, oob_score=False,\
                                          n_jobs=1, random_state=42, verbose=0, warm_start=False,\
                                          class_weight=cWdict)
                           
        else:
             clf = RandomForestClassifier(n_estimators=10, criterion="gini", max_depth=None, \
                                          min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0,\
                                          max_features="auto", max_leaf_nodes=None, min_impurity_decrease=0.0,\
                                          min_impurity_split=None, bootstrap=True, oob_score=False,\
                                          n_jobs=1, random_state=42, verbose=0, warm_start=False,\
                                          class_weight=None)
    elif Used_classifier=='ERF':         
        if (classweight==1) and CW==1: 
             clf = ExtraTreesClassifier(n_estimators=10, criterion="gini", max_depth=None,\
                                          min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0,\
                                          max_features="auto", max_leaf_nodes=None, min_impurity_decrease=0.0,\
                                          min_impurity_split=None, bootstrap=True, oob_score=False,\
                                          n_jobs=1, random_state=42, verbose=0, warm_start=False,\
                                          class_weight=cWdict)
        else:
             clf = ExtraTreesClassifier(n_estimators=10, criterion="gini", max_depth=None,\
                                          min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0,\
                                          max_features="auto", max_leaf_nodes=None, min_impurity_decrease=0.0,\
                                          min_impurity_split=None, bootstrap=True, oob_score=False,\
                                          n_jobs=1, random_state=42, verbose=0, warm_start=False,\
                                          class_weight=None)         
        
#Performance analysis        
#        sys.exit('Jan werth')
    if len(label)<2:
        print("please use at least two labels")
        
 # F1 Kappa
    else: 
        prediction=clf.fit(X_train,y_train.ravel()).predict(X_test)
        
        scoring=clf.score(X_test, y_test.ravel(),  sample_weight=None)
        Fimportances=clf.feature_importances_
        Dpath=clf.decision_path(X_train)        
        
        resultsF1_macro=f1_score(y_test.ravel(), prediction, average='macro')#, pos_label=None)
        resultsF1_micro=f1_score(y_test.ravel(), prediction,average='micro')
        resultsF1_weight=f1_score(y_test.ravel(), prediction,average='weighted')
        resultsF1_all=f1_score(y_test.ravel(), prediction,labels=label, average=None)#, pos_label=None)
        
        resultsK=cohen_kappa_score(y_test.ravel(),prediction,labels=label)
        
        if drawing:
               import graphviz
               from Loading_5min_mat_files_cECG import Class_dict,features_dict
               dot_data = tree.export_graphviz(clf, out_file=None) 
               graph = graphviz.Source(dot_data) 
#              graph.render() 
              
               dot_data = tree.export_graphviz(clf, out_file=None, 
                                        feature_names=features_dict,  
                                        class_names=Class_dict,  
                                        filled=True, rounded=True,  
                                        special_characters=True)  
               graph = graphviz.Source(dot_data)  
               graph 
                 
        
                  
    return resultsF1_macro,resultsK,resultsF1_micro,resultsF1_weight,resultsF1_all,Fimportances,scoring
        #F1 returns with average='none' a F1 score for each label or macro=meaned