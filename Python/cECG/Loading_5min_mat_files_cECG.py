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
from sklearn.manifold import Isomap
from AnnotationChanger import AnnotationChanger
from sklearn.preprocessing import PolynomialFeatures





def Loading_data_all(dataset,selectedbabies,lst,ux,scaling,\
                     LoosingAnnot5,LoosingAnnot6,LoosingAnnot6_2,direction6,plotting,Smoothing_short,Pack4,\
                     Movingwindow,preaveraging,postaveraging,exceptNOF,onlyNOF,FEAT,\
                     PolyTrans,ExpFactor,exceptNOpF,onlyNOpF,FEATp):
       ##***************
       #dataset='ECG'  # Either ECG or cECG and later maybe MMC or InnerSense
       ##***************
       #selectedbabies =[0,1,3,5,6,7] #0-8 ('4','5','6','7','9','10','11','12','13')
       ##selectedbabies =[0,1,2,3,4,5,6,7,8] #0-8 ('4','5','6','7','9','10','11','12','13')
       ##---------------------------
       #
       ## Feature list
       #lst = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29]
       ##lst_old=[3,4,5,6,7,8,9,10,11,14,15,16,17,18,19,20,21,22,23,24,25,26] # From first paper to compare with new features
       ##lst=lst_old
       ##lst = [0,2,3,4,5,6,7,8,9,10,11,12,14,15,16,17,18,19,20,21,22,23,24,25,26,29]
       ##---------------------------
       #
       #ux=0 # if using this on Linux cluster use 1 to change adresses
       #scaling='Z' # Scaling Z or MM 
       ##---------------------------
       #
       #LoosingAnnot5= 0# exchange state 5 if inbetween another state with this state (also only if length <= x)
       #LoosingAnnot6=0  #Exchange state 6 with the following or previouse state (depending on direction)
       #LoosingAnnot6_2=0 # as above, but chooses always 2 when 6 was lead into with 1
       #direction6=0 # if State 6 should be replaced with the state before, use =1; odtherwise with after, use =0. Annotators used before.
       #plotting=0 #plotting annotations
       #Smoothing_short=0 # # short part of any annotation are smoothed out. 
       #Pack4=0 # State 4 is often split in multible short parts. Merge them together as thebaby does not calm downin 1 min
       ##---------------------------
       #
       #Movingwindow=7 # WIndow size for moving average
       #preaveraging=0
       #postaveraging=1
       #exceptNOF=1 #Which Number of Features (NOF) should be used with moving average?  all =oth tzero; only some or all except some defined in FEAT
       #onlyNOF=0 # [0,1,2,27,28,29]
       #FEAT=[0,1,2]
       ##FEAT=[1,2,27,28] # FRO CT
       ##----------------------------
       #
       #PolyTrans=0#use polinominal transformation on the Features specified in FEATp
       #ExpFactor=2# which degree of polinomonal (2)
       #exceptNOpF= 0#Which Number of Features (NOpF) should be used with polynominal fit?  all =0; only some or all except some defined in FEATp
       #onlyNOpF=1 # [0,1,2,27,28,29]
       ##FEATp=[1,2,27,28] # FRO CT
       #FEATp=[0,3,4,5]
       ##=---------------------------
       #
       #RBFkernel=0
       
       """
       START *************************************************************************
       """
       
       if 'ECG'== dataset:
              if ux:
                     folder=('/home/310122653/Pyhton_Folder/cECG/Matrices/')
              else:
                     folder=('C:/Users/310122653/Dropbox/PHD/python/cECG/Matrices/')
       if 'cECG'==dataset:
              if ux:  
                     folder=('/home/310122653/Pyhton_Folder/cECG/cMatrices/')
              else:
                     folder=('C:/Users/310122653/Dropbox/PHD/python/cECG/cMatrices/')
           
       # ONLY 5 MIN FEATURES AND ANNOTATIONS
       dateien_each_patient="FeatureMatrix_","Annotations_" #non scaled values. The values should be scaled over all patient and not per patient. Therfore this is better
       windowlength="30"
       Neonate_all='4','5','6','7','9','10','11','12','13'
       babies=[i for i in range(len(selectedbabies))]# return to main function
       
       Neonate=[(Neonate_all[i]) for i in selectedbabies];Neonate=tuple(Neonate)
       FeatureMatrix_each_patient_all=[0]*len(Neonate)
       AnnotMatrix_each_patient=[0]*len(Neonate)
       t_a=[0]*len(Neonate)
       
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
                   t_a[k]=np.linspace(0,len(AnnotMatrix_each_patient[k])*30/60,len(AnnotMatrix_each_patient[k]))  
                   if plotting:
                        plt.figure(k) 
                        plt.plot(t_a[k],AnnotMatrix_each_patient[k])
                        plt.title([k])
       #            AnnotMatrix_each_patient[k]= np.delete(AnnotMatrix_each_patient[k],(1,2), axis=1) #Reduce AnnotationMatrix to Nx1
       #            AnnotMatrix_each_patient[k]=AnnotMatrix_each_patient[k][~np.isnan(AnnotMatrix_each_patient[k]).any(axis=1)]#deleting NAN and turning Matrix to datapoints,Features
       
       
       FeatureMatrix_each_patient_all=[val[:,lst] for sb, val in enumerate(FeatureMatrix_each_patient_all)] # selecting only the features in lst
                      
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

       for K in range(len(Neonate)):               
              if PolyTrans:
                     poly = PolynomialFeatures(degree=ExpFactor)
                     NOpF=np.arange(0,(np.size(FeatureMatrix_each_patient_all[K],1))) # create range from 0-29 (lenth of features)
                     if exceptNOpF:
                            NOpF= np.delete(NOpF,FEATp)
                     if onlyNOpF:
                            NOpF=FEATp
                            
                     FeatureMatrix_each_patient_all_poly[K] = poly.fit_transform(FeatureMatrix_each_patient_all[K][:,NOpF]) # transform the old features into old FEatures and feature combinations
                     Nr_of_orig_Features=len(FeatureMatrix_each_patient_all[K][0]) # to check for dublicates a few lines down we need the old length of the Features
                     FeatureMatrix_each_patient_all[K]=np.hstack((FeatureMatrix_each_patient_all[K],FeatureMatrix_each_patient_all_poly[K][:,1:]))#merge the transformed and the old features (transformation only work on FEAT).Not first column as it is all 1
                     
                     Cindex=[]
                     for i in range(Nr_of_orig_Features):# now we search fro dublicates as we merged the old and the transformed features
                            for h in [x for x in range(len(FeatureMatrix_each_patient_all[K][0])) if x!=i]:
                                   dublicate_col=(FeatureMatrix_each_patient_all[K][:,i])==(FeatureMatrix_each_patient_all[K][:,h])
                                   if all(dublicate_col)==True:
                                          Cindex=np.append(Cindex,h)                            
                     FeatureMatrix_each_patient_all[K]=np.delete(FeatureMatrix_each_patient_all[K],Cindex,1)# delet all the doublicate features columns   
                     
              if postaveraging:             
                     NOF=np.arange(0,(np.size(FeatureMatrix_each_patient_all[K],1))) # create range from 0-29 (lenth of features)
                     if exceptNOF:
                            NOF= np.delete(NOF,FEAT)
                     if onlyNOF:
                            NOF=FEAT
                     for F in NOF:#range(np.size(FeatureMatrix_each_patient_fromSession[K],1)):
                            FeatureMatrix_each_patient_all[K][:,F]=\
                            np.convolve(FeatureMatrix_each_patient_all[K][:,F], np.ones((Movingwindow,))/Movingwindow, mode='same')                
                                          

       AnnotMatrix_each_patient=AnnotationChanger(AnnotMatrix_each_patient,LoosingAnnot5,LoosingAnnot6,LoosingAnnot6_2,Smoothing_short,Pack4,direction6)

                     
       return babies, AnnotMatrix_each_patient, FeatureMatrix_each_patient_all
       
                     
                     
                     
       
       #%%
       
def Loading_data_perSession(dataset,selectedbabies,lst,ux,scaling,\
                     LoosingAnnot5,LoosingAnnot6,LoosingAnnot6_2,direction6,plotting,Smoothing_short,Pack4,\
                     Movingwindow,preaveraging,postaveraging,exceptNOF,onlyNOF,FEAT,\
                     PolyTrans,ExpFactor,exceptNOpF,onlyNOpF,FEATp):    
       
       """
       Creating Feature Matrix per session
       """
       
       
       #folder=('/home/310122653/Pyhton_Folder/cECG/Matrices/')
       if 'ECG'== dataset:
              if ux:
                     Sessionfolder=('/home/310122653/Pyhton_Folder/cECG/Matrices/Sessions/')
              else:
                     Sessionfolder=('C:/Users/310122653/Dropbox/PHD/python/cECG/Matrices/Sessions/')
       if 'cECG'==dataset:
              if ux:
                     Sessionfolder=('/home/310122653/Pyhton_Folder/cECG/cMatrices/Sessions/')
              else:
                     Sessionfolder=('C:/Users/310122653/Dropbox/PHD/python/cECG/cMatrices/Sessions/')

       # ONLY 5 MIN FEATURES AND ANNOTATIONS
       dateien_each_patient="FeatureMatrix_","Annotations_" #non scaled values. The values should be scaled over all patient and not per patient. Therfore this is better
       windowlength="30"
       Neonate_all='4','5','6','7','9','10','11','12','13'
       babies=[i for i in range(len(selectedbabies))]# return to main function
       
       Neonate=[(Neonate_all[i]) for i in selectedbabies];Neonate=tuple(Neonate)
       FeatureMatrix_each_patient_all=[0]*len(Neonate)
       AnnotMatrix_each_patient=[0]*len(Neonate)
       t_a=[0]*len(Neonate)                     
       
       import os
       import glob
       from pathlib import Path
       
       sc = StandardScaler()
       sMM= MinMaxScaler()  
       
       FeatureMatrix_each_patient_fromSession=[None]*len(Neonate)
       FeatureMatrix_each_patient_fromSession_poly=[None]*len(Neonate)
       
       AnnotMatrix_each_patient=Loading_Annotations(dataset,selectedbabies,ux,plotting) # loading Annotations
       
       for K in range(len(Neonate)):      
       #       SessionFileList=[]
              Dateien=glob.glob(Sessionfolder +'FeatureMatrix_'+Neonate[K]+ '_**')
       #       SessionFileList=[None]*(len(Neonate))
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
                                     
                      
       #Moving average
       # We use a moving average as the annotations where done on video observations. This are never aprubt observations, therefore we smoothen out the data a bit to come closer to the annotation behaviour
              if preaveraging:
                     for i in range(len(FeatureMatrix_Session_each_patient)):
                            for F in range(np.size(FeatureMatrix_Session_each_patient[i],1)):
                                   FeatureMatrix_Session_each_patient[i][:,F]=\
                                   np.convolve(FeatureMatrix_Session_each_patient[i][:,F], np.ones((Movingwindow,))/Movingwindow, mode='same')  
       #Scaling 
             
              for i in range(len(FeatureMatrix_Session_each_patient)):
                     if scaling=='Z': 
                            sc.fit(FeatureMatrix_Session_each_patient[i])
                            FeatureMatrix_Session_each_patient[i]=sc.transform(FeatureMatrix_Session_each_patient[i])
                     elif scaling=='MM':
                            sMM.fit(FeatureMatrix_Session_each_patient[i])
                            FeatureMatrix_Session_each_patient[i]=sMM.transform(FeatureMatrix_Session_each_patient[i])  
                     else:
                            sys.exit('Misspelling of the scaling type')
                            
              FeatureMatrix_each_patient_fromSession[K]=np.concatenate(FeatureMatrix_Session_each_patient)
              
       FeatureMatrix_each_patient_fromSession=[val[:,lst] for sb, val in enumerate(FeatureMatrix_each_patient_fromSession)] # selecting only the features used iin lst
              
              
       for K in range(len(Neonate)):           
              if PolyTrans:
                     poly = PolynomialFeatures(degree=ExpFactor)
                     NOpF=np.arange(0,(np.size(FeatureMatrix_each_patient_fromSession[K],1))) # create range from 0-29 (lenth of features)
                     if exceptNOpF:
                            NOpF= np.delete(NOpF,FEATp)
                     if onlyNOpF:
                            NOpF=FEATp
                            
                     FeatureMatrix_each_patient_fromSession_poly[K] = poly.fit_transform(FeatureMatrix_each_patient_fromSession[K][:,NOpF]) # transform the old features into old FEatures and feature combinations
                     Nr_of_orig_Features=len(FeatureMatrix_each_patient_fromSession[K][0]) # to check for dublicates a few lines down we need the old length of the Features
                     FeatureMatrix_each_patient_fromSession[K]=np.hstack((FeatureMatrix_each_patient_fromSession[K],FeatureMatrix_each_patient_fromSession_poly[K][:,1:]))#merge the transformed and the old features (transformation only work on FEAT).Not first column as it is all 1
                     
                     Cindex=[]
                     for i in range(Nr_of_orig_Features):# now we search fro dublicates as we merged the old and the transformed features
                            for h in [x for x in range(len(FeatureMatrix_each_patient_fromSession[K][0])) if x!=i]:
                                   dublicate_col=(FeatureMatrix_each_patient_fromSession[K][:,i])==(FeatureMatrix_each_patient_fromSession[K][:,h])
                                   if all(dublicate_col)==True:
                                          Cindex=np.append(Cindex,h)                            
                     FeatureMatrix_each_patient_fromSession[K]=np.delete(FeatureMatrix_each_patient_fromSession[K],Cindex,1)# delet all the doublicate features columns   
       #              lst=range(len(FeatureMatrix_each_patient_fromSession[]))
                     
              if postaveraging:             
                     NOF=np.arange(0,(np.size(FeatureMatrix_each_patient_fromSession[K],1))) # create range from 0-29 (lenth of features)
                     if exceptNOF:
                            NOF= np.delete(NOF,FEAT)
                     if onlyNOF:
                            NOF=FEAT
                     for F in NOF:#range(np.size(FeatureMatrix_each_patient_fromSession[K],1)):
                            FeatureMatrix_each_patient_fromSession[K][:,F]=\
                            np.convolve(FeatureMatrix_each_patient_fromSession[K][:,F], np.ones((Movingwindow,))/Movingwindow, mode='same')                
                     
       AnnotMatrix_each_patient=AnnotationChanger(AnnotMatrix_each_patient,LoosingAnnot5,LoosingAnnot6,LoosingAnnot6_2,Smoothing_short,Pack4,direction6)
               
       ##Non-linear dimensionality reduction through Isometric Mapping                     
       #for K in range(len(Neonate)):                                           
       #       if NonlinTrans:       
       #              IM=Isomap(n_neighbors=5, n_components=2, eigen_solver='auto', tol=0, max_iter=None, path_method='auto', neighbors_algorithm='auto', n_jobs=1)
       #              IM.fit_transform(FeatureMatrix_each_patient_fromSession[K])
       
#                     
#       if plotting:
#              for l in range(len(AnnotMatrix_each_patient)): 
#                 t_a[l]=np.linspace(0,len(AnnotMatrix_each_patient[l])*30/60,len(AnnotMatrix_each_patient[l]))  
#                 plt.figure(l) 
#                 plt.plot(t_a[l],AnnotMatrix_each_patient[l]-0.1)
#                 plt.title([l])                            
                                    
                 
       return babies, AnnotMatrix_each_patient, FeatureMatrix_each_patient_fromSession
#%%      
def Feature_names():

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
       return Class_dict, features_dict, features_indx
#%%
def Loading_Annotations(dataset,selectedbabies,ux,plotting):
       if 'ECG'== dataset:
              if ux:
                     folder=('/home/310122653/Pyhton_Folder/cECG/Matrices/')
              else:
                     folder=('C:/Users/310122653/Dropbox/PHD/python/cECG/Matrices/')
       if 'cECG'==dataset:
              if ux:  
                     folder=('/home/310122653/Pyhton_Folder/cECG/cMatrices/')
              else:
                     folder=('C:/Users/310122653/Dropbox/PHD/python/cECG/cMatrices/')
           
       # ONLY 5 MIN FEATURES AND ANNOTATIONS
       dateien_each_patient="FeatureMatrix_","Annotations_" #non scaled values. The values should be scaled over all patient and not per patient. Therfore this is better
       windowlength="30"
       Neonate_all='4','5','6','7','9','10','11','12','13'
       babies=[i for i in range(len(selectedbabies))]# return to main function
       
       Neonate=[(Neonate_all[i]) for i in selectedbabies];Neonate=tuple(Neonate)
       FeatureMatrix_each_patient_all=[0]*len(Neonate)
       AnnotMatrix_each_patient=[0]*len(Neonate)
       t_a=[0]*len(Neonate)
       
       # IMPORTING *.MAT FILES
       for j in range(len(dateien_each_patient)): # j=0 Features  j=1 Annotations
           for k in range(len(Neonate)):
               Dateipfad=folder+dateien_each_patient[j]+Neonate[k]+"_win_"+windowlength+".mat" #Building foldername
               sio.absolute_import   
               matlabfile=sio.loadmat(r'{}'.format(Dateipfad)) 
           

               if j==1:
                   AnnotMatrix_each_patient[k]=matlabfile.get('Annotations')  
                   AnnotMatrix_each_patient[k]=AnnotMatrix_each_patient[k].transpose() # transpose to datapoints,annotations
                   t_a[k]=np.linspace(0,len(AnnotMatrix_each_patient[k])*30/60,len(AnnotMatrix_each_patient[k]))  
#                   if plotting:
#                        plt.figure(k) 
#                        plt.plot(t_a[k],AnnotMatrix_each_patient[k])
#                        plt.title([k])
                        
       return AnnotMatrix_each_patient
       
