# -*- coding: utf-8 -*-
"""
Created on Sun Oct 29 18:39:46 2017

@author: 310122653
"""
def cmplx_Oversampling(X,Y,ChoosenKind,SamplingMeth):
    from imblearn.over_sampling import SMOTE, ADASYN
    #smote Oversampling
    Kindtype=['regular','borderline1','borderline2','svm']
    
    if SamplingMeth=='SMOTE':
           X_resampled, y_resampled = SMOTE(kind=Kindtype[ChoosenKind]).fit_sample(X, Y)
    elif SamplingMeth=='ADASYN': 
           X_resampled, y_resampled = ADASYN().fit_sample(X, Y)
    else:
           disp:'Choose imba;ance_learn Methode. ADASYN or SMOTE'
           return
    return X_resampled, y_resampled


#def cmplx_Undersampling(X,Y,ChoosenKind,SamplingMeth):