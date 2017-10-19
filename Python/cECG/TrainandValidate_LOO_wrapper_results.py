# -*- coding: utf-8 -*-
"""
Created on Wed Dec 21 07:46:43 2016

@author: 310122653
"""
# This is the One used for the 2th paper

from Loading_5min_mat_files import AnnotMatrix_each_patient, FeatureMatrix_each_patient, Class_dict, features_dict, features_indx

from matplotlib import *
from numpy import *
from pylab import *

from sklearn.cross_validation import StratifiedKFold
from sklearn.cross_validation import cross_val_score
from sklearn.cross_validation import KFold

#from sklearn.metrics import *
from sklearn.metrics import accuracy_score 
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc

from sklearn import svm, cross_validation
#from sklearn.svm import SVC

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron
from compute_class_weight import *   
import itertools
from itertools import cycle #cycle line styles


import time
start_time = time.time()

classweight=0 # do we want to concider the class imbalance? then =1 
savefig=1


### Create all feature combiations
lst = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19]
combs=[]
for i in range(1, len(lst)+1):
    els = [list(x) for x in itertools.combinations(lst, i)]
    combs.extend(els)  
combs_short=[combs for combs in combs if len(combs) <= 5] #due to computation time we stop with 5 Features per set.

### load result from the wrapper
#AUCs=load ('C:/Users/C3PO/Documents/PhD/outfiles/outfile_5_latest.npy')
#AUCs=load ('C://Users//C3PO//Documents//PhD//Matrizen//outfile_3_latest.npy')
#combs_short=load('C://Users//C3PO//Documents//PhD//Matrizen//combs_short_3.npy')

AUCs=load("C://Users//310122653//Dropbox//PHD//python//Inner Sence//outfiles/outfile_5_nachLeuven_5min.npy")
combs_short=load("C://Users//310122653//Dropbox//PHD//python//Inner Sence//outfiles/combs_short_5_nachLeuven_5min.npy")
m=max(AUCs)
idx_m=[i for i, j in enumerate(AUCs) if j == m]
#combinations[idx_m]=[0.1] #to test what would be thesecond maximum. Delete the max,run it again
selecetedF=combs_short[idx_m[0]]# which features to train and test on
#selecetedF = [x - 1 for x in selecetedF]

### normalizing data
sc = StandardScaler()
for i in range(len(FeatureMatrix_each_patient)):
    sc.fit(FeatureMatrix_each_patient[i])
    FeatureMatrix_each_patient[i]=sc.transform(FeatureMatrix_each_patient[i])

#--------------------------------LEAVE ONE OUT---------------------------------

## Train SVM and validate with leave one out    
#(The file with the HRV features of all patients together is not standardized, the one for each individual patient is)

#SELECTING THE LABELS FOR SELECTED BABIES
label=array([1,2])
selected_babies =[0,1,2,3,4,5,6,7] #0-7
summation=sum(selected_babies)
AnnotMatrix_auswahl=[AnnotMatrix_each_patient[k] for k in selected_babies]              # get the annotation values for selected babies
FeatureMatrix_auswahl=[FeatureMatrix_each_patient[k] for k in selected_babies]

idx=[in1d(AnnotMatrix_each_patient[sb],label) for sb in selected_babies]#.values()]     # which are the idices for AnnotMatrix_each_patient == label
idx=[nonzero(idx[sb])[0] for sb in range(len(selected_babies))]#.values()]              # get the indices where True
y_each_patient=[val[idx[sb],:] for sb, val in enumerate(AnnotMatrix_auswahl) if sb in range(len(selected_babies))] #get the values for y from idx and label

#CREATING THE DATASET WITH selsected features WITH SPECIFIC LABELS FOR SELECTED BABIES  
Xfeat=[val[:,selecetedF] for sb, val in enumerate(FeatureMatrix_auswahl)] # selecting the features to run
Xfeat=[val[idx[sb],:] for sb, val in enumerate(Xfeat)]#.values()]                                               #selecting the datapoints in label

#TRAIN CLASSIFIER
meanaccLOO=[];accLOO=[];testsubject=[];tpr_mean=[];counter=0;
mean_tpr = 0.0;mean_fpr = np.linspace(0, 1, 100)
meanFPrate=[];meanTPrate=[];meanSPC=[];meanAcc=[];meanNegPredVal=[];meanPosPredVal=[];confmatsum=([[0,0],[0,0]])

print ('initialisation process complete')
from itertools import cycle #cycle line styles
lines = ['-','--','-.',':']
linecycler = cycle(lines)
    
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
    print ('create test and training set for preterm: ', j+1)
#CALCULATE THE WEIGHTS DUE TO CLASS IMBALANCE
    class_weight='balanced'
    classes=label
    classlabels=ravel(y_test) # y_test has to be a 1d array for compute_class_weight
    if (classweight==1): #and (Selected_test!=7):# as baby 7 does not have two classes, it is not unbalnced
        cW=compute_class_weight(class_weight, classes, classlabels)
        cWdict={1:cW[0]};cWdict={2:cW[1]} #the class weight need to be a dictionarry of the form:{class_label : value}
        print('cW for patient %i: AS %.3f  QS %.3f' %(j+1,cW[0],cW[1]))   
#THE SVM
    if (classweight==1): #and (Selected_test!=7):# as baby 7 does not have two classes, it is not unbalnced
         clf = svm.SVC(kernel='rbf', class_weight=cWdict, probability=True, random_state=42)
    else:
        clf = svm.SVC(kernel='rbf',gamma=0.2, C=3.7, probability=True, random_state=42)
    print('SVM setup for preterm: ',j+1 ,' complete' )          
    probas_=clf.fit(X_train,y_train).predict_proba(X_test)  
    print('probas calculated for preterm: ', j+1)
# ROC and AUC
    fpr, tpr, thresholds = roc_curve(y_test, probas_[:, 1], pos_label=2)
    if isnan(sum(tpr))== False and isnan(sum(fpr))==False:
        mean_tpr += interp(mean_fpr, fpr, tpr)
        mean_tpr[0] = 0.0
        counter+=1

    roc_auc = auc(fpr, tpr)
    print('ROC and AUC calculated for preterm: ', j+1)
    
# Confusion Matrix
    if (Selected_test!=7):# as baby 7 does not have two classes, it is not from binary
#        y_pred=clf.predict(X_test)    
        y_pred=clf.fit(X_train,y_train).predict(X_test)            
        confmat=confusion_matrix(y_true=y_test, y_pred=y_pred)  
        confmatsum+=confmat
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
#        assert confmat.shape == (2,2) #"Confusion matrix should be from binary classification only."  
        #SENSITIVITY SPECIFICITY ACCURACY...
        tn=confmat[0,0]; fp=confmat[0,1]; fn=confmat[1,0]; tp=confmat[1,1];
        NP=fn+tp; NN=tn+fp; N=NP+NN; 
        FPrate=(fp/(fp+tn));        meanFPrate.append(FPrate)
        TPrate=(tp/(tp+fn));        meanTPrate.append(TPrate)
        SPC=tn/(fp+tn);             meanSPC.append(SPC)
        Acc=((tp+tn)/N);            meanAcc.append(Acc)
        NegPredVal=(1-fn/(fn+tn));  meanNegPredVal.append(NegPredVal)
        PosPredVal=(tp/(tp+fp));    meanPosPredVal.append(PosPredVal)
        

    if (Selected_test!=7):
        figure(2, facecolor='white');
        plot(fpr, tpr, lw=1, ls=next(linecycler), label='Pat. %d (AUC = %0.2f)' % (j+1, roc_auc) )
        plt.xlim([-0.05, 1.05])
        plt.ylim([-0.05, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')       
        plt.legend(loc="lower right")
        plt.show() 
        plt.rcParams.update({'font.size': 20})        
#        if savefig==1:
#            plt.savefig('C:/Users/310122653/Documents/PhD/Article 1/FIgures/Autosave/ROC age 0.85.svg',format='svg', dpi=600)
#            plt.savefig('C:/Users/310122653/Documents/PhD/Article 1/FIgures/Autosave/ROC age 0.85.eps', format='eps', dpi=1000)
#            plt.savefig('C:/Users/310122653/Documents/PhD/Article 1/FIgures/Autosave/ROC age 0.85.tiff', format='tiff', dpi=600)        
#            manager = plt.get_current_fig_manager() #maximize window size
#            manager.window.showMaximized()
            
        
mean_tpr /= counter#len(selected_babies)
mean_tpr[-1] = 1.0
mean_auc = auc(mean_fpr, mean_tpr)      
#figure(facecolor="white"); # if extra figure for mean ROC, activate this figure
plot(mean_fpr, mean_tpr, label='Mean ROC (AUC = %0.2f)' % mean_auc, lw=2.3)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate') 
legend(loc=4)
plt.rcParams.update({'font.size': 20})
if savefig==1:
    plt.savefig('C:/Users/310122653/Documents/PhD/Article 1/FIgures/Autosave/ROC_and_meanROC  0.85.svg',format='svg', dpi=600)
    plt.savefig('C:/Users/310122653/Documents/PhD/Article 1/FIgures/Autosave/ROC_and_meanROC  0.85.eps', format='eps', dpi=1000)
    plt.savefig('C:/Users/310122653/Documents/PhD/Article 1/FIgures/Autosave/ROC_and_meanROC  0.85.tiff', format='tiff', dpi=600)     
    manager = plt.get_current_fig_manager() #maximize window size
    manager.window.showMaximized()


print('acc: %d' %Acc)
print('Max AUC: %.3f' %m)
for feat in range(len(selecetedF)):
   print ('The top features are: %s'%(features_dict[selecetedF[feat]]))  #features_dict[selecetedF[1] for j in range(len(selecetedF))])
# To see what are the performances of following subsets and what are the featrues   
for sets in range(9):
    AUCs[idx_m]=[0.1] #to test what would be thesecond maximum. Delete the max,run it again
    m=max(AUCs)
    idx_m=[i for i, j in enumerate(AUCs) if j == m]
    selecetedF=combs_short[idx_m[0]]# which features to train and test on
    print('with a performance of %.3f' %m)
    for feat in range(len(selecetedF)):
        print('the %dth set, top features are %s' %(sets, features_dict[selecetedF[feat]]))

print("--- %s seconds ---" % (time.time() - start_time))

#Latest on 17.5.2017