

# -*- coding: utf-8 -*-
"""
Created on Fri Dec 16 22:23:10 2016

@author: 310122653
"""

from Loading_5min_mat_files import AnnotMatrix_each_patient, FeatureMatrix_each_patient, Class_dict, features_dict, features_indx

import itertools
from matplotlib import *
from numpy import *
from pylab import *
from sklearn.preprocessing import StandardScaler
#from sklearn.cross_validation import StratifiedKFold
from sklearn.cross_validation import cross_val_score
from sklearn.cross_validation import KFold
from sklearn.metrics import accuracy_score 
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc
from sklearn import svm, cross_validation
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
#from sklearn.linear_model import Perceptron
from compute_class_weight import *   


import time

start_time = time.time()

classweight=1 # If classweights should be automatically determined and used for trainnig use: 1 else:0

#### CREATE ALL POSSIBLE COMBINATIONS OUT OF 17 FEATURES. STOP AT 5 FEATURE SET(DUE TO SVM COMPUTATION TIME)
lst = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19]
combs=[]
for i in range(1, len(lst)+1):
    els = [list(x) for x in itertools.combinations(lst, i)]
    combs.extend(els)  
combs_short=[combs for combs in combs if len(combs) <= 5] #due to computation time we stop with 5 Features per set.

#### SCALE FEATURES
sc = StandardScaler()
for i in range(len(FeatureMatrix_each_patient)):
    sc.fit(FeatureMatrix_each_patient[i])
    FeatureMatrix_each_patient[i]=sc.transform(FeatureMatrix_each_patient[i])

collected_mean_auc=[]
    
#### SELECT FEATURE SET
for Fc in range(len(combs_short)):

  
#### SELECTING THE LABELS FOR SELECTED BABIES
    label=array([1,2])
    selected_babies =[0,1,2,3,4,5,6,7] #0-7
#    selected_babies =[0,1,2,3,4,5,6] #0-7

    summation=sum(selected_babies)
    AnnotMatrix_auswahl=[AnnotMatrix_each_patient[k] for k in selected_babies]              # get the annotation values for selected babies
    FeatureMatrix_auswahl=[FeatureMatrix_each_patient[k] for k in selected_babies]
    
    idx=[in1d(AnnotMatrix_each_patient[sb],label) for sb in selected_babies]#.values()]     # which are the idices for AnnotMatrix_each_patient == label
    idx=[nonzero(idx[sb])[0] for sb in range(len(selected_babies))]#.values()]              # get the indices where True
    y_each_patient=[val[idx[sb],:] for sb, val in enumerate(AnnotMatrix_auswahl) if sb in range(len(selected_babies))] #get the values for y from idx and label

#### CREATING THE DATASET WITH x numbers of features WITH SPECIFIC LABELS FOR SELECTED BABIES  
    Xfeat=[val[:,combs_short[Fc]] for sb, val in enumerate(FeatureMatrix_auswahl)] # selecting the features to run
    Xfeat=[val[idx[sb],:] for sb, val in enumerate(Xfeat)]   #selecting the datapoints in label


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
        from compute_class_weight import *   
        class_weight='balanced'
        classes=label  
        if Selected_test==0:
            classlabels=[1]*61
            classlabels= append(classlabels,[2]*39)
        if Selected_test==1:
            classlabels=[1]*63
            classlabels= append(classlabels,[2]*37)        
        if Selected_test==2:
            classlabels=[1]*63
            classlabels= append(classlabels,[2]*37) 
        if Selected_test==3:
            classlabels=[1]*75
            classlabels= append(classlabels,[2]*25)        
        if Selected_test==4:
            classlabels=[1]*63
            classlabels= append(classlabels,[2]*37) 
        if Selected_test==5:
            classlabels=[1]*61
            classlabels= append(classlabels,[2]*39)   
        if Selected_test==6:
            classlabels=[1]*71
            classlabels= append(classlabels,[2]*29)  
        if Selected_test==7:
            classlabels=[1]*71
            classlabels= append(classlabels,[2]*29)        
            
        if (classweight==1) and (Selected_test!=7):# as baby 7 does not have two classes, it is not unbalnced
            cW=compute_class_weight(class_weight, classes, classlabels)
            cWdict={1:cW[0]};cWdict={2:cW[1]} #the class weight need to be a dictionarry of the form:{class_label : value}
        
    #THE SVM
        if (classweight==1):# as baby 7 does not have two classes, it is not unbalnced
             clf = svm.SVC(kernel='rbf',gamma=0.2, class_weight=cWdict, probability=True, random_state=42)
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
    
    mean_tpr /= counter#len(selected_babies)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr) #Store all mean AUC for each combiation 
    collected_mean_auc.append(mean_auc)
    print('Round: %i of:%i' %(Fc,len(combs_short))) 

save('D://Dropbox//PHD//python//Outfiles_after_Leuven//outfile_5_incl_age_afterLeuven_5min', collected_mean_auc)
save('D://Dropbox//PHD//python//Outfiles_after_Leuven//combs_short_5_incle_aga_afterLeuven_5min', combs_short)   
 
#save('C://Users//C3PO//Documents//PhD//Matrizen//outfile_7_latest_incl_age', collected_mean_auc)
#save('C://Users//C3PO//Documents//PhD//Matrizen//combs_short_7_age', combs_short)


# find all maxima. If collected_mean_auc is not calculated but loaded, the name has to be changed to file name
#m=max(outfile_5)
#[i for i, j in enumerate(outfile_5) if j == m]
#outfile_5[7254]=0.1
#outfile_5[40]

print("--- %s seconds ---" % (time.time() - start_time))
