# -*- coding: utf-8 -*-
"""
Created on Sun Oct 29 18:39:46 2017

@author: 310122653
"""
from numpy import *
def cmplx_Oversampling(X,Y,ChoosenKind,SamplingMeth,label):
    from imblearn.over_sampling import SMOTE, ADASYN
    #smote Oversampling
    Kindtype=['regular','borderline1','borderline2','svm']
    # OVersampling can be done with ratios for each class, so that unimportant classes are not oversampled too much
    #http://contrib.scikit-learn.org/imbalanced-learn/stable/generated/imblearn.over_sampling.SMOTE.html
    AS=round(count_nonzero(Y==1))
    QS=round(count_nonzero(Y==2)*1.9)
    W=round(count_nonzero(Y==3)*1.3)
    CT=round(count_nonzero(Y==4)*1.8)
    NA=round(count_nonzero(Y==5)*1.2)
    IS=round(count_nonzero(Y==6)*1.8)
    states={1:AS,2:QS,3:W,4:CT,5:NA,6:IS}
    Verhaeltniss = {your_key: states[your_key] for your_key in label }
    

    if SamplingMeth=='SMOTE':
           X_resampled, y_resampled = SMOTE(kind=Kindtype[ChoosenKind],random_state=42,ratio=Verhaeltniss).fit_sample(X, Y)
    elif SamplingMeth=='ADASYN': 
           X_resampled, y_resampled = ADASYN(random_state=42,ratio=Verhaeltniss).fit_sample(X, Y)
    elif SamplingMeth=='NONE':
           X_resampled=X
           y_resampled=Y             
    else:
           disp:'Choose imbalance_learn Methode. ADASYN or SMOTE'
           return
    return X_resampled, y_resampled


#def cmplx_Undersampling(X,Y,ChoosenKind,SamplingMeth):