# -*- coding: utf-8 -*-
"""
Created on Wed Oct 18 16:03:32 2017

@author: 310122653
"""

#        elif len(label)==2:
## ROC and 
#            probas_=clf.fit(X_train,y_train.ravel()).predict_proba(X_test)  
#        
#            fpr, tpr, thresholds = roc_curve(y_test, probas_[:, 1], pos_label=2)
#            if isnan(sum(tpr))== False and isnan(sum(fpr))==False:# only if there are no nans to the mean AUC...
#                mean_tpr += interp(mean_fpr, fpr, tpr)
#                mean_tpr[0] = 0.0
#                counter+=1
#       
#                roc_auc = auc(fpr, tpr)# eingerueckt
#            
#                mean_tpr /= counter#len(selected_babies)                  
#                mean_tpr[-1] = 1.0
#                mean_auc = auc(mean_fpr, mean_tpr) #Store all mean AUC for each combiation 
#            #collected_mean_auc.append(mean_auc) Do that outside of fucntion, otherwise always set to []
#            
#                results=mean_auc
#            else:#... otherwise use the mean AUC from before
#                print('on of the classes in one session does not exist, therefore nanin tpr,fpr and no AUC possibel')
#                
                
#            return results #mean_auc
        
#        elif len(label)>2: # use F1 sscore for multi label classification performance measurement