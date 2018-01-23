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
from sklearn.gaussian_process.kernels import RBF
from sklearn.kernel_approximation import RBFSampler




def Loading_data_all(dataset,selectedbabies,lst,Rpeakmethod,ux,scaling,\
                     LoosingAnnot5,LoosingAnnot6,LoosingAnnot6_2,direction6,plotting,Smoothing_short,Pack4,merge34,\
                     Movingwindow,preaveraging,postaveraging,exceptNOF,onlyNOF,FEAT,\
                     PolyTrans,ExpFactor,exceptNOpF,onlyNOpF,FEATp,RBFkernel):

       """
       START *************************************************************************
       """
       if Rpeakmethod == 'R':
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
       elif Rpeakmethod == 'M':
              if 'ECG'== dataset:
                     if ux:
                            folder=('/home/310122653/Pyhton_Folder/cECG/MatricesM/')
                     else:
                            folder=('C:/Users/310122653/Dropbox/PHD/python/cECG/MatricesM/')
              if 'cECG'==dataset:
                     if ux:  
                            folder=('/home/310122653/Pyhton_Folder/cECG/cMatricesM/')
                     else:
                            folder=('C:/Users/310122653/Dropbox/PHD/python/cECG/cMatricesM/')              
           
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
              
              if RBFkernel:
                     rbf_feature = RBFSampler(gamma=10, random_state=42)
                     FeatureMatrix_each_patient_all[K] = rbf_feature.fit_transform(FeatureMatrix_each_patient_all[K])
                            
                     
              if postaveraging:             
                     NOF=np.arange(0,(np.size(FeatureMatrix_each_patient_all[K],1))) # create range from 0-29 (lenth of features)
                     if exceptNOF:
                            NOF= np.delete(NOF,FEAT)
                     if onlyNOF:
                            NOF=FEAT
                     for F in NOF:#range(np.size(FeatureMatrix_each_patient_fromSession[K],1)):
                            FeatureMatrix_each_patient_all[K][:,F]=\
                            np.convolve(FeatureMatrix_each_patient_all[K][:,F], np.ones((Movingwindow,))/Movingwindow, mode='same')                
                                          

       AnnotMatrix_each_patient=AnnotationChanger(AnnotMatrix_each_patient,LoosingAnnot5,LoosingAnnot6,LoosingAnnot6_2,Smoothing_short,Pack4,direction6,merge34)

                     
       return babies, AnnotMatrix_each_patient, FeatureMatrix_each_patient_all
       
                     
                     
                     
       
       #%%
       
def Loading_data_perSession(dataset,selectedbabies,lst,Rpeakmethod,ux,scaling,\
                     LoosingAnnot5,LoosingAnnot6,LoosingAnnot6_2,direction6,plotting,Smoothing_short,Pack4,merge34,\
                     Movingwindow,preaveraging,postaveraging,exceptNOF,onlyNOF,FEAT,\
                     PolyTrans,ExpFactor,exceptNOpF,onlyNOpF,FEATp,dispinfo):    
       
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
              Dateien=glob.glob(Sessionfolder +'FeatureMatrix_'+Neonate[K]+ '_**')
              FeatureMatrix_Session_each_patient=[None]*len(Dateien)

       
       # IMPORTING *.MAT FILES
              for j in range(len(Dateien)): 
                      sio.absolute_import   
                      matlabfile=sio.loadmat(r'{}'.format(Dateien[j])) 
           
       # REWRITING FEATURES AND ANNOTATIONS    
           #NANs should already be deleted. Not scaled.
           #NANs can be in as there are only NaNs with NaN annotations. Nan al label is not used
                      FeatureMatrix_Session_each_patient[j]=matlabfile.get('FeatureMatrix') 
                      FeatureMatrix_Session_each_patient[j]=FeatureMatrix_Session_each_patient[j].transpose() # transpose to datapoints,features
       # TRIMMING THEM IF SESSIONS ARE TO SHORT OR EMPTY               
#              FeatureMatrix_Session_each_patient[1]=[];FeatureMatrix_Session_each_patient[5]=[]# just a test delete
              WelcheSindLeer=list()
              WelcheSindzuKurz=list()
              for j in range(len(Dateien)): 
                     if len(FeatureMatrix_Session_each_patient[j])==0:
                            WelcheSindLeer.append(j) #Just count how many Sessions do not have cECG values. If more than one different strategy is needed than the one below
                     if len(FeatureMatrix_Session_each_patient[j])!=0 and len(FeatureMatrix_Session_each_patient[j])<=2: # If a session is to short, remove it
                            WelcheSindzuKurz.append(j) #Just count how many Sessions do not have cECG values. If more than one different strategy is needed than the one below
                            
              if WelcheSindzuKurz:# deleting the ones that are too short in Annotation MAtrix. Apparently if there is no data(leer), then the Annotations are already shortened. Therefore only corretion for the once which are a bit to short
                     AnnotMatrix_each_patient=correcting_Annotations_length(K,WelcheSindzuKurz,ux,selectedbabies,AnnotMatrix_each_patient,FeatureMatrix_Session_each_patient)
                     WelcheSindLeer.extend(WelcheSindzuKurz)# delete the ones that are zero and the once that are too short
                     
              for index in sorted(WelcheSindLeer, reverse=True):
                     del FeatureMatrix_Session_each_patient[index]
#              FeatureMatrix_Session_each_patient=[m for n, m in enumerate(FeatureMatrix_Session_each_patient) if n not in WelcheSindLeer] #remove empty session


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
#              if RBFkernel:
#                     rbf_feature = RBFSampler(gamma=10, random_state=42)
#                     FeatureMatrix_each_patient_fromSession[K] = rbf_feature.fit_transform(FeatureMatrix_each_patient_fromSession[K])
                                          
              if postaveraging:             
                     NOF=np.arange(0,(np.size(FeatureMatrix_each_patient_fromSession[K],1))) # create range from 0-29 (lenth of features)
                     if exceptNOF:
                            NOF= np.delete(NOF,FEAT)
                     if onlyNOF:
                            NOF=FEAT
                     for F in NOF:#range(np.size(FeatureMatrix_each_patient_fromSession[K],1)):
                            FeatureMatrix_each_patient_fromSession[K][:,F]=\
                            np.convolve(FeatureMatrix_each_patient_fromSession[K][:,F], np.ones((Movingwindow,))/Movingwindow, mode='same')                
                     
       AnnotMatrix_each_patient=AnnotationChanger(AnnotMatrix_each_patient,LoosingAnnot5,LoosingAnnot6,LoosingAnnot6_2,Smoothing_short,Pack4,direction6,merge34)
               
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
                      29:"SEAUC",
                      30:"pDEC",
                      31:"SDDEC",                      
                      32:"LZNN",
                      33:"LZECG"
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
       
#%%
def correcting_Annotations_missing(K,WelcheSindLeer,ux,selectedbabies,AnnotMatrix_each_patient):
# This function is needed to load the ECG if the cECG Is loaded to compare the length of missing cECG value. The missing length can then be deleted from the annotations      
       if ux:
              Sessionfolder=('/home/310122653/Pyhton_Folder/cECG/Matrices/Sessions/')
       else:
              Sessionfolder=('C:/Users/310122653/Dropbox/PHD/python/cECG/Matrices/Sessions/')


       # ONLY 5 MIN FEATURES AND ANNOTATIONS
       dateien_each_patient="FeatureMatrix_","Annotations_" #non scaled values. The values should be scaled over all patient and not per patient. Therfore this is better
       windowlength="30"
       Neonate_all='4','5','6','7','9','10','11','12','13'
       babies=[i for i in range(len(selectedbabies))]# return to main function
       
       Neonate=[(Neonate_all[i]) for i in selectedbabies];Neonate=tuple(Neonate)
       FeatureMatrixECG=[0]*len(Neonate)
       
       import os
       import glob
       from pathlib import Path
             
       FeatureMatrix_each_patient_fromSession=[None]*len(Neonate)            
            
       Dateien=glob.glob(Sessionfolder +'FeatureMatrix_'+Neonate[K]+ '_**')
       FeatureMatrix_Session_each_patient=[None]*len(Dateien)

       
       # IMPORTING *.MAT FILES
       for w in range(len(Dateien)): 
               sio.absolute_import   
               matlabfile=sio.loadmat(r'{}'.format(Dateien[w])) 
    
       # REWRITING FEATURES AND ANNOTATIONS    
           #NANs should already be deleted. Not scaled.
           #NANs can be in as there are only NaNs with NaN annotations. Nan al label is not used
               FeatureMatrix_Session_each_patient[w]=matlabfile.get('FeatureMatrix') 
               FeatureMatrix_Session_each_patient[w]=FeatureMatrix_Session_each_patient[w].transpose() # transpose to datapoints,features
       
       # COllecting all the indices from the parts where single sessions are empty. We get the start index and the lengt. This is combined into on lare arry with all the indices that are missing. Those are then deleted at once from the annotations
       IndexRange=np.zeros(len(WelcheSindLeer))
       startindex=list()
       for l in range(len(WelcheSindLeer)):
              IndexRange[l]=(len(FeatureMatrix_Session_each_patient[WelcheSindLeer[l]])) # collect how many smaples are missing
              VonDa=[[(len(FeatureMatrix_Session_each_patient[t])) for t in range(WelcheSindLeer[l])]] # starting index by collecting all length before the missing one
              VonDa=np.sum(VonDa)
              startindex.append(list(range(VonDa+1,VonDa+1+np.int(IndexRange[l]))))# getting all the start indices in one array/list

       indices=np.hstack(startindex)
       AnnotMatrix_each_patient[K]= np.delete(AnnotMatrix_each_patient[K][:],[indices])

       return AnnotMatrix_each_patient
#%%
def correcting_Annotations_length(K,WelcheSindzuKurz,ux,selectedbabies,AnnotMatrix_each_patient,FeatureMatrix_Session_each_patient):
# This function is needed to load the ECG if the cECG Is loaded to compare the length of missing cECG value. The missing length can then be deleted from the annotations      

       # COllecting all the indices from the parts where single sessions are empty. We get the start index and the lengt. This is combined into on lare arry with all the indices that are missing. Those are then deleted at once from the annotations
       IndexRange=np.zeros(len(WelcheSindzuKurz))
       startindex=list()
       for l in range(len(WelcheSindzuKurz)):
              IndexRange[l]=(len(FeatureMatrix_Session_each_patient[WelcheSindzuKurz[l]])) # collect how many smaples are missing
              VonDa=[[(len(FeatureMatrix_Session_each_patient[t])) for t in range(WelcheSindzuKurz[l])]] # starting index by collecting all length before the missing one
              VonDa=np.int(np.sum(VonDa))
#              startindex.append(list(range(VonDa+1,VonDa+1+np.int(IndexRange[l]))))# getting all the start indices in one array/list
              startindex.append(list(range(VonDa,VonDa+np.int(IndexRange[l]))))# getting all the start indices in one array/list

       indices=np.hstack(startindex)
       indices=sorted(indices, reverse=True )
       
#       for i in range(len(indices)):
#              AnnotMatrix_each_patient[K]= np.delete(AnnotMatrix_each_patient[K],[indices[i],1])
       AnnotMatrix_each_patient[K]= np.delete(AnnotMatrix_each_patient[K][:],[indices])
       AnnotMatrix_each_patient[K]=AnnotMatrix_each_patient[K][:, None] # make it a 2D array. Otherwise error later
       return AnnotMatrix_each_patient


       