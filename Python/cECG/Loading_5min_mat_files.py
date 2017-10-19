# -*- coding: utf-8 -*-
"""
Created on Sat Dec 17 21:10:17 2016

@author: 310122653
"""
# this is the one used for the 2th paper
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 25 14:02:00 2016

@author: 310122653
"""

#def loadingMatrizen():
    #When importing a file, Python only searches the current directory, 
    #the directory that the entry-point script is running from, and sys.path 
    #which includes locations such as the package installation directory 
import scipy.io as sio
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
import math

#folder=('C://Users//310122653//Documents//PhD//InnerSense Data//Matlab//saves//Matrizen//')
folder=('E:/InnerSence/processed data/Matrices/')
#folder=('C://Users//C3PO//Documents//PhD//testmarix//')

# ONLY 5 MIN FEATURES AND ANNOTATIONS
#dateien_each_patient="SW_5min_feature_Matrix_patient_","SW_5min_annotation_Matrix_patient_" #non scaled values. The values should be scaled over all patient and not per patient. Therfore this is better
#dateien_each_patient="SW_30s_feature_Matrix_patient_","SW_30s_annotation_Matrix_patient_" #non scaled values. The values should be scaled over all patient and not per patient. Therfore this is better
dateien_each_patient="Feature_Matrix_","Annotation_Matrix_ASQS_","sampleWeight_"  #non scaled values. The values should be scaled over all patient and not per patient. Therfore this is better
windowlength="300"
Neonate='3','4','5','6','7','9','13','15'

FeatureMatrix_each_patient=[0]*len(Neonate)
AnnotMatrix_each_patient=[0]*len(Neonate)
sampleWeights=[0]*len(Neonate)

# IMPORTING *.MAT FILES
for j in range(len(dateien_each_patient)): # j=0 Features  j=1 Annotations
    for k in range(len(Neonate)):
        Dateipfad=folder+dateien_each_patient[j]+Neonate[k]+"_win_"+windowlength+".mat" #Building foldername
        sio.absolute_import   
        matlabfile=sio.loadmat(r'{}'.format(Dateipfad)) 
    
# REWRITING FEATURES AND ANNOTATIONS    
    #FeatureMatrix = 0 when no annotastions. Annotation Matrix =Nan. This is changed here into 666 and label 666 is never used. 
#        If deleting nans or zeres this can lead to poblems as the annotations are created with overlapping windows over 30s windows. If Nans are 1/4 more then AS or QS the epoch becomes nan. 
#       The Featurematrix has still values. Deleting Nanas would crete the probleme here. jsut exchanging the annotation epochs with 666 where nans are dominant is the solution.
        if j==0:
            FeatureMatrix_each_patient[k]=matlabfile.get('Matrix') 
            FeatureMatrix_each_patient[k]=FeatureMatrix_each_patient[k].transpose() # transpose to datapoints,features
#            FeatureMatrix_each_patient[k]=FeatureMatrix_each_patient[k][~np.isnan(FeatureMatrix_each_patient[k]).any(axis=1)]#deleting NAN and turning Matrix to datapoints,Features
#            FeatureMatrix_each_patient[k]=FeatureMatrix_each_patient[k].loc[:, (FeatureMatrix_each_patient[k] != 0).all(axis=0)] # *.loc means  choose this:  deletes all columns(axis0 here correct) when there are all 0: http://stackoverflow.com/questions/21164910/delete-column-in-pandas-based-on-condition
        if j==1:
            AnnotMatrix_each_patient[k]=matlabfile.get('Matrix')  
            AnnotMatrix_each_patient[k]=AnnotMatrix_each_patient[k].transpose() # transpose to datapoints,annotations
#            AnnotMatrix_each_patient[k]= np.delete(AnnotMatrix_each_patient[k],(1,2), axis=1) #Reduce AnnotationMatrix to Nx1
#            AnnotMatrix_each_patient[k]=AnnotMatrix_each_patient[k][~np.isnan(AnnotMatrix_each_patient[k]).any(axis=1)]#deleting NAN and turning Matrix to datapoints,Features
#            AnnotMatrix_each_patient[k]= np.nan_to_num(AnnotMatrix_each_patient[k])    #This replaces Nans with zero. Thereby nothing is shiftet and label 0 is just never used. 
            tmpor=AnnotMatrix_each_patient[k] 
            tmpor[np.isnan(tmpor)]=0  
            AnnotMatrix_each_patient[k]=tmpor
#           FeatureMatrix_each_patient[k]=FeatureMatrix_each_patient[k].loc[:, (FeatureMatrix_each_patient[k] != 0).all(axis=0)] # *.loc means  choose this:  deletes all columns(axis0 here correct) when there are all 0: http://stackoverflow.com/questions/21164910/delete-column-in-pandas-based-on-condition
        if j==3:
            sampleWeights[k]=matlabfile.get('Matrix') 
            sampleWeights[k]=sampleWeights[k].transpose()
            
##create feature dictionary with all names and the indices
features_indx={"BpE":                   [0],
               "NN10":                  [1],
               "NN20":                  [2],
               "NN30":                  [3],
               "NN50":                  [4],  
               "pNN10":                 [5],
               "pNN20":                 [6],
               "pNN30":                 [7],
               "pNN50":                 [8],
               "RMSSD":                 [9],
               "SDNN":                  [10],
                "HF":                   [11],
                "HFnorm":               [12],
                "LF":                   [13],
                "LFnorm":               [14],
                "ratioLFHF":            [15],
                "sHF":                  [16],
                "totpower":             [17],
                "uHF":                  [18],
                "VLF":                  [19],




}    
#                   
Class_dict={1:'AS',2:'QS',3:'AW',4:'QW' ,6:'Trans'} #AS:active sleep   QS:Quiet sleep  AW:Active wake  QW:Quiet wake  Trans:Transition  Pos:Position         
Class_dict={1:'AS',2:'QS',3:'W' ,6:'Trans'} #AS:active sleep   QS:Quiet sleep  AW:Active wake  QW:Quiet wake  Trans:Transition  Pos:Position         

#
features_dict={
               0:"Bpe",
               1:"NN10",                  
               2:"NN20",                
               3:"NN30",                 
               4:"NN50", 
               5:"RMSSD",                 
               6:"SDNN",                 
               7:"pNN10",               
               8:"pNN20",               
               9:"pNN30",               
               10:"pNN50",              
                11:"HF",                 
                12:"HFnorm",
                13:"LF",                  
                14:"LFnorm",  
                15:"VLF",                
                16:"ratioLFHF",           
                17:"sHF",
                18:"totpower",
                19:"uHF",

              }     

            
#            
###create feature dictionary with all names and the indices
#features_indx={"BpE":                   [1],
#               "NN10":                  [2],
#               "NN20":                  [3],
#               "NN30":                  [4],
#               "NN50":                  [5],  
#               "pNN10":                 [6],
#               "pNN20":                 [7],
#               "pNN30":                 [8],
#               "pNN50":                 [9],
#               "RMSSD":                 [10],
#               "SDNN":                  [11],
#                "HF":                   [12],
#                "HF_normalized":        [13],
#                "HFnorm":               [14],
#                "HFnorm_normalized":    [15],
#                "LF":                   [16],
#                "LFnormalized":         [17],
#                "LFnorm":               [18],
#                "LFnorm_normalized":    [19],
#                "ratioLFHF":            [20],
#                "ratioLFHF_normalized": [21],
#                "sHF":                  [22],
#                "sHF_normalized":       [23],
#                "sHFnorm":              [24],
#                "sHFnorm_normalized":   [25],
#                "totpower":             [26],
#                "totpower_normalized":  [27],
#                "uHF":                  [28],
#                "uHF_normalized":       [28],
#                "uHFnorm":              [29],
#                "uHFnorm_normalized":   [30],
#                "VLF":                  [31],
#                "VLF_normalized":       [32],
#
#
#
#
#}    
##                   
#Class_dict={1:'AS',2:'QS',3:'AW',4:'QW' ,6:'Trans'} #AS:active sleep   QS:Quiet sleep  AW:Active wake  QW:Quiet wake  Trans:Transition  Pos:Position         
#Class_dict={1:'AS',2:'QS',3:'W' ,6:'Trans'} #AS:active sleep   QS:Quiet sleep  AW:Active wake  QW:Quiet wake  Trans:Transition  Pos:Position         
#
##
#features_dict={
#               1:"Bpe",
#               2:"NN10",                  
#               3:"NN20",                
#               4:"NN30",                 
#               5:"NN50",                  
#               6:"pNN10",               
#               7:"pNN20",               
#               8:"pNN30",               
#               9:"pNN50",              
#               10:"RMSSD",                 
#               11:"SDNN",                 
#                12:"HF",                 
#                13:"HF_normalized",       
#                14:"HFnorm",
#                15:"HFnorm_normalized",   
#                16:"LF",                  
#                17:"LFnormalized",       
#                18:"LFnorm",               
#                19:"LFnorm_normalized",   
#                20:"ratioLFHF",           
#                21:"ratioLFHF_normalized",
#                22:"sHF",
#                23:"sHF_normalized",
#                24:"sHFnorm",
#                25:"sHFnorm_normalized",
#                26:"totpower",
#                27:"totpower_normalized",
#                28:"uHF",
#                29:"uHF_normalized",
#                30:"uHFnorm",
#                31:"uHFnorm_normalized",
#                32:"VLF",
#                33:"VLF_normalized",
#              }     

#
feature_idx = dict((y,x) for x,y in features_dict.items())

              
#return (AnnotMatrix_each_patient, FeatureMatrix_each_patient, Class_dict, features_dict, features_indx)
              #Latest on 17.5.2017