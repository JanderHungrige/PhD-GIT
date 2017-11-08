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
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

#***************
dataset='ECG'  # Either ECG or cECG and later maybe MMC or InnerSense
#***************
scaling='Z' # SCaling Z or MM 
Annotationchanger=1 #Exchange state 6 with the following state
direction6=0 # if State 6 should be replaced with the state before, use =1; odtherwise with after, use =0
Loosing5=1 # exchange state 5 if inbetween another state with this state (also only if length <= x)
plotting=0

#folder=('/home/310122653/Pyhton_Folder/cECG/Matrices/')
if 'ECG'== dataset:
    folder=('C:/Users/310122653/Dropbox/PHD/python/cECG/Matrices/')
if 'cECG'==dataset:
    folder=('C:/Users/310122653/Dropbox/PHD/python/cECG/cMatrices/')
    
    
# ONLY 5 MIN FEATURES AND ANNOTATIONS
dateien_each_patient="FeatureMatrix_","Annotations_" #non scaled values. The values should be scaled over all patient and not per patient. Therfore this is better
windowlength="30"
Neonate='4','5','6','7','9','10','11','12','13'

FeatureMatrix_each_patient_all=[0]*len(Neonate)
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
            FeatureMatrix_each_patient_all[k]=matlabfile.get('FeatureMatrix') 
            FeatureMatrix_each_patient_all[k]=FeatureMatrix_each_patient_all[k].transpose() # transpose to datapoints,features
#            FeatureMatrix_each_patient[k]=FeatureMatrix_each_patient[k][~np.isnan(FeatureMatrix_each_patient[k]).any(axis=1)]#deleting NAN and turning Matrix to datapoints,Features

        elif j==1:
            AnnotMatrix_each_patient[k]=matlabfile.get('Annotations')  
            AnnotMatrix_each_patient[k]=AnnotMatrix_each_patient[k].transpose() # transpose to datapoints,annotations
            if plotting:
                 plt.plot(AnnotMatrix_each_patient[k])
#            AnnotMatrix_each_patient[k]= np.delete(AnnotMatrix_each_patient[k],(1,2), axis=1) #Reduce AnnotationMatrix to Nx1
#            AnnotMatrix_each_patient[k]=AnnotMatrix_each_patient[k][~np.isnan(AnnotMatrix_each_patient[k]).any(axis=1)]#deleting NAN and turning Matrix to datapoints,Features

      
#### SCALE FEATURES
sc = StandardScaler()
sMM= MinMaxScaler()
for i in range(len(FeatureMatrix_each_patient_all)):
       if scaling=='Z': 
              sc.fit(FeatureMatrix_each_patient_all[i])
              FeatureMatrix_each_patient_all[i]=sc.transform(FeatureMatrix_each_patient_all[i])
       elif scaling=='MM':
              sMM.fit(FeatureMatrix_each_patient_all[i])
              FeatureMatrix_each_patient_all[i]=sMM.transform(FeatureMatrix_each_patient_all[i]) 
       else:
              sys.exit('Misspelling of the scaling type')
              
              

#%%
"""
Creating Feature Matrix per session
"""


#folder=('/home/310122653/Pyhton_Folder/cECG/Matrices/')
if 'ECG'== dataset:
    Sessionfolder=('C:/Users/310122653/Dropbox/PHD/python/cECG/Matrices/Sessions/')
if 'cECG'==dataset:
    Sessionfolder=('C:/Users/310122653/Dropbox/PHD/python/cECG/cMatrices/Sessions/')
	

import os
import glob
from pathlib import Path

FeatureMatrix_each_patient_fromSession=[None]*len(Neonate)
for k in range(len(Neonate)):      
       SessionFileList=[]
       Dateien=glob.glob(Sessionfolder +'FeatureMatrix_'+Neonate[k]+ '_**')
       SessionFileList=[None]*(len(Neonate))
       FeatureMatrix_Session_each_patient=[None]*len(Dateien)

#       FeatureMatrix_each_patient_Session=[0]*len(Neonate)
#       AnnotMatrix_each_patient_Session=[0]*len(Neonate)

# IMPORTING *.MAT FILES
       for j in range(len(Dateien)): # j=0 Features  j=1 Annotations
               sio.absolute_import   
               matlabfile=sio.loadmat(r'{}'.format(Dateien[j])) 
    
# REWRITING FEATURES AND ANNOTATIONS    
    #NANs should already be deleted. Not scaled.
    #NANs can be in as there are only NaNs with NaN annotations. Nan al label is not used
               FeatureMatrix_Session_each_patient[j]=matlabfile.get('FeatureMatrix') 
               FeatureMatrix_Session_each_patient[j]=FeatureMatrix_Session_each_patient[j].transpose() # transpose to datapoints,features
               
       for i in range(len(FeatureMatrix_Session_each_patient)):
              if scaling=='Z': 
                     sc.fit(FeatureMatrix_Session_each_patient[i])
                     FeatureMatrix_Session_each_patient[i]=sc.transform(FeatureMatrix_Session_each_patient[i])
              elif scaling=='MM':
                     sMM.fit(FeatureMatrix_Session_each_patient[i])
                     FeatureMatrix_Session_each_patient[i]=sMM.transform(FeatureMatrix_Session_each_patient[i])  
              else:
                     sys.exit('Misspelling of the scaling type')
                     
       FeatureMatrix_each_patient_fromSession[k]=np.concatenate(FeatureMatrix_Session_each_patient)
                      
#%% Change the annotations
if Annotationchanger:
       count6=0;
       before6=0
       for l in range(len(AnnotMatrix_each_patient)):
              for  M in range(len(AnnotMatrix_each_patient[l])):
                     if (AnnotMatrix_each_patient[l][M]==6): 
                            count6=count6+1
                            before6=int(AnnotMatrix_each_patient[l][M-2])
                     elif AnnotMatrix_each_patient[l][M]!=6 and count6!=0:
                            if direction6:
                                   AnnotMatrix_each_patient[l][M-count6:M]=before6
                                   count6=0
                                   before6=0
                                   
                            else:
                                   AnnotMatrix_each_patient[l][M-count6:M]=int(AnnotMatrix_each_patient[l][M])
                                   count6=0
if Loosing5:
       count5=0
       before5=0
       for l in range(len(AnnotMatrix_each_patient)):
              for  M in range(len(AnnotMatrix_each_patient[l])):
                     if (AnnotMatrix_each_patient[l][M]==5):
                            count5=count5+1
                            before5=int(AnnotMatrix_each_patient[l][M-1])
                     elif AnnotMatrix_each_patient[l][M]!=5 and count5!=0 and count5<=30 and before==int(AnnotMatrix_each_patient[l][M]): # if 5 is inbewteen the same state; and if Not annotatable is on y 10 long(5min)
                            AnnotMatrix_each_patient[l][M-count5:M]=int(AnnotMatrix_each_patient[l][M])
                            count5=0 
                            before5=0
                      
#%%                                  
Class_dict={1:'AS',2:'QS',3:'Wake', 4:'caretaking',5:'Unknown',6:'Trans'} #AS:active sleep   QS:Quiet sleep  AW:Active wake  QW:Quiet wake  Trans:Transition  Pos:Position         

features_dict={
               0:"Bpe",
               1:"LineLength",
               2:"meanLineLength",
               3:"NN10",                  
               4:"NN20",                
               5:"NN30",                 
               6:"NN50",                  
               7:"pNN10",               
               8:"pNN20",               
               9:"pNN30",               
               10:"pNN50",              
               11:"RMSSD",
               12:"SDaLL",                 
               13:"SDANN", 
               14:"SDLL",
               15:"SDNN",                
               16:"HF",                 
               17:"HFnorm",
               18:"LF",                  
               19:"LFnorm",               
               20:"ratioLFHF",           
               21:"sHF",
               22:"sHFnorm",
               23:"totpower",
               24:"uHF",
               25:"uHFnorm",
               26:"VLF",
               27:"SampEN",
               28:"QSE",
               29:"SEAUC"
               }     

features_indx = dict((y,x) for x,y in features_dict.items())