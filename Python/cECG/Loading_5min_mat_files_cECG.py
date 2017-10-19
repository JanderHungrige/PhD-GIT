# -*- coding: utf-8 -*-
"""
Created on Fri Sep  8 22:21:39 2017

@author: 310122653
"""
#LOADING cECG Matlab data (Feature and annotation Matrices)

#def loadingMatrizen():
    #When importing a file, Python only searches the current directory, 
    #the directory that the entry-point script is running from, and sys.path 
    #which includes locations such as the package installation directory 
import scipy.io as sio
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
import math

#***************
dataset='ECG'  # Either ECG or cECG and later maybe MMC or InnerSense
#***************

#folder=('/home/310122653/Pyhton_Folder/cECG/Matrices/')
if 'ECG'== dataset:
    folder=('C:/Users/310122653/Dropbox/PHD/python/cECG/Matrices/')
if 'cECG'==dataset:
    folder=('C:/Users/310122653/Dropbox/PHD/python/cECG/cMatrices/')
    
    
# ONLY 5 MIN FEATURES AND ANNOTATIONS
dateien_each_patient="FeatureMatrix_","Annotations_" #non scaled values. The values should be scaled over all patient and not per patient. Therfore this is better
windowlength="30"
Neonate='4','5','6','7','9','10','11','12','13'

FeatureMatrix_each_patient=[0]*len(Neonate)
AnnotMatrix_each_patient=[0]*len(Neonate)

# IMPORTING *.MAT FILES
for j in range(len(dateien_each_patient)): # j=0 Features  j=1 Annotations
    for k in range(len(Neonate)):
        Dateipfad=folder+dateien_each_patient[j]+Neonate[k]+"_win_"+windowlength+".mat" #Building foldername
        sio.absolute_import   
        matlabfile=sio.loadmat(r'{}'.format(Dateipfad)) 
    
# REWRITING FEATURES AND ANNOTATIONS    
    #NANs should already be deleted. Not scaled.
    #NANs can be in as there are only NaNs with NaN annotations. Nan al label is not used
        if j==0:
            FeatureMatrix_each_patient[k]=matlabfile.get('FeatureMatrix') 
            FeatureMatrix_each_patient[k]=FeatureMatrix_each_patient[k].transpose() # transpose to datapoints,features
#            FeatureMatrix_each_patient[k]=FeatureMatrix_each_patient[k][~np.isnan(FeatureMatrix_each_patient[k]).any(axis=1)]#deleting NAN and turning Matrix to datapoints,Features

        elif j==1:
            AnnotMatrix_each_patient[k]=matlabfile.get('Annotations')  
            AnnotMatrix_each_patient[k]=AnnotMatrix_each_patient[k].transpose() # transpose to datapoints,annotations
#            AnnotMatrix_each_patient[k]= np.delete(AnnotMatrix_each_patient[k],(1,2), axis=1) #Reduce AnnotationMatrix to Nx1
#            AnnotMatrix_each_patient[k]=AnnotMatrix_each_patient[k][~np.isnan(AnnotMatrix_each_patient[k]).any(axis=1)]#deleting NAN and turning Matrix to datapoints,Features

            
##create feature dictionary with all names and the indices
features_indx={"BpE":                   [1],
               "LineLength":            [2],
               "meanlineLength":        [3],
               "NN10":                  [4],#
               "NN20":                  [5],#
               "NN30":                  [6],#
               "NN50":                  [7], # 
               "pNN10":                 [8],#
               "pNN20":                 [9],#
               "pNN30":                 [10],#
               "pNN50":                 [11],#
               "RMSSD":                 [12],#
               "SDaLL":                 [13],
               "SDANN":                 [14],#
               "SDLL":                  [15],
               "SDNN":                  [16],#
               "HF":                    [17],#
               "HFnorm":                [18],#
               "LF":                    [19],#
               "LFnorm":                [20],#
               "ratioLFHF":             [21],#
               "sHF":                   [22],#
               "sHFnorm":               [23],#
               "totpower":              [24],#
               "uHF":                   [25],#
               "uHFnorm":               [26],#
               "VLF":                   [27],#
               "SampEn":                [28],
               "QSE":                   [29],
               "SEAUC":                 [30]
}    


#                   
Class_dict={1:'AS',2:'QS',3:'W', 4:'caretaking',5:'Unknown',6:'Trans'} #AS:active sleep   QS:Quiet sleep  AW:Active wake  QW:Quiet wake  Trans:Transition  Pos:Position         

#
features_dict={
               1:"Bpe",
               2:"LineLength",
               3:"meanLineLength",
               4:"NN10",                  
               5:"NN20",                
               6:"NN30",                 
               7:"NN50",                  
               8:"pNN10",               
               9:"pNN20",               
               10:"pNN30",               
               11:"pNN50",              
               12:"RMSSD",
               13:"SDaLL",                 
               14:"SDANN", 
               15:"SDLL",
               16:"SDNN",                
               17:"HF",                 
               18:"HFnorm",
               19:"LF",                  
               20:"LFnorm",               
               21:"ratioLFHF",           
               22:"sHF",
               23:"sHFnorm",
               24:"totpower",
               25:"uHF",
               26:"uHFnorm",
               27:"VLF",
               28:"SampEN",
               29:"QSE",
               30:"SEAUC"
               }     

feature_idx_auto = dict((y,x) for x,y in features_dict.items())

              
#return (AnnotMatrix_each_patient, FeatureMatrix_each_patient, Class_dict, features_dict, features_indx)