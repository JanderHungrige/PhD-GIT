# -*- coding: utf-8 -*-
"""
Created on Fri Nov 24 11:23:32 2017

@author: 310122653
"""

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
from sklearn.decomposition import PCA
from sklearn.gaussian_process.kernels import RBF
from sklearn.kernel_approximation import RBFSampler
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm


from Loading_5min_mat_files_cECG import correcting_Annotations_length
"""
Set variables *****************************************************************
"""
#***************
dataset='cECG'  # Either ECG or cECG and later maybe MMC or InnerSense
#***************
selectedbabies =[0,2,3,5,7] #0-8 ('4','5','6','7','9','10','11','12','13')
#selectedbabies =[0,1,2,3,4,5,6,7,8] #0-8 ('4','5','6','7','9','10','11','12','13')
#---------------------------
label=[1,2,3,4,6]
# Feature list
lst = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29] #SDANN =13
#lst_old=[3,4,5,6,7,8,9,10,11,14,15,16,17,18,19,20,21,22,23,24,25,26] # From first paper to compare with new features
#lst=lst_old
#lst = [0,1,13]

#---------------------------

ux=0 # if using this on Linux cluster use 1 to change adresses
scaling='Z' # Scaling Z or MM 
#---------------------------

LoosingAnnot5= 0# exchange state 5 if inbetween another state with this state (also only if length <= x)
LoosingAnnot6=0  #Exchange state 6 with the following or previouse state (depending on direction)
LoosingAnnot6_2=0 # as above, but chooses always 2 when 6 was lead into with 1
direction6=0 # if State 6 should be replaced with the state before, use =1; odtherwise with after, use =0. Annotators used before.
plotting=0 #plotting annotations
Smoothing_short=0 # # short part of any annotation are smoothed out. 
Pack4=0 # State 4 is often split in multible short parts. Merge them together as thebaby does not calm downin 1 min
merge34=1
if merge34 and 3 in label:
       label.remove(3)
#---------------------------

Movingwindow=10 # WIndow size for moving average
preaveraging=0
postaveraging=0
exceptNOF=1 #Which Number of Features (NOF) should be used with moving average?  all =oth tzero; only some or all except some defined in FEAT
onlyNOF=0 # [0,1,2,27,28,29]
#FEAT=[0,1,2]
FEAT=[1,2,27,28] # FRO CT
#----------------------------

PolyTrans=0#use polinominal transformation on the Features specified in FEATp
exceptNOpF=0 #Which Number of Features (NOpF) should be used with polynominal fit?  all =0; only some or all except some defined in FEATp
onlyNOpF=0 # [0,1,2,27,28,29]
FEATp=[1,2,27,28] # FRO CT
Exponent=2
#=---------------------------

RBFkernel=1

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
              
              


#%%
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

import os
import glob
from pathlib import Path

FeatureMatrix_each_patient_fromSession=[None]*len(Neonate)
FeatureMatrix_each_patient_fromSession_poly=[None]*len(Neonate)

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
              poly = PolynomialFeatures(degree=Exponent)
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
              
       if RBFkernel:
#              RBGkernel=RBF(length_scale=1.0, length_scale_bounds=(1e-05, 100000.0))
#              ParamsRBF[K] =RBFkernel.get_params(deep=True)
#              
              rbf_feature = RBFSampler(gamma=10, random_state=42)
              FeatureMatrix_each_patient_fromSession[K] = rbf_feature.fit_transform(FeatureMatrix_each_patient_fromSession[K])
                     
              
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
 
AnnotMatrix_auswahl_test=[AnnotMatrix_each_patient[k] for k in babies]              # get the annotation values for selected babies
FeatureMatrix_auswahl_test=[FeatureMatrix_each_patient_fromSession[k] for k in babies]
idx_test=[np.in1d(AnnotMatrix_each_patient[sb],label) for sb in babies]#.values()]     # which are the idices for AnnotMatrix_each_patient == label
idx_test=[np.nonzero(idx_test[sb])[0] for sb in range(len(babies))]#.values()]              # get the indices where True
Xfeat_test=[val[idx_test[sb],:] for sb, val in enumerate(FeatureMatrix_auswahl_test)]  
y_each_patient_test=[val[idx_test[sb],:] for sb, val in enumerate(AnnotMatrix_auswahl_test) if sb in range(len(babies))] #get the value



X=np.vstack(Xfeat_test) # mergin the data from each list element into one matrix 
y=np.vstack(y_each_patient_test)


pca=PCA(copy=True, iterated_power='auto', n_components=3, random_state=None,
  svd_solver='auto', tol=0.0, whiten=False)

X_PCa=pca.fit_transform(X)
       
#plt.figure
#plt.scatter(X_PCa[np.where(y==1),0],X_PCa[np.where(y==1),1],label=1)
#plt.scatter(X_PCa[np.where(y==2),0],X_PCa[np.where(y==2),1],label=2); 
#plt.scatter(X_PCa[np.where(y==4),0],X_PCa[np.where(y==4),1],label=4); 
#plt.show()
#plt.legend()

Xas=X_PCa[np.where(y==1),0]
Yas=X_PCa[np.where(y==1),1]
Zas=X_PCa[np.where(y==1),2]

Xqs=X_PCa[np.where(y==2),0]
Yqs=X_PCa[np.where(y==2),1]
Zqs=X_PCa[np.where(y==2),2]

Xct=X_PCa[np.where(y==4),0]
Yct=X_PCa[np.where(y==4),1]
Zct=X_PCa[np.where(y==4),2]

fig = plt.figure()
ax = fig.gca(projection='3d')
ax.scatter(Xas,Yas,Zas)
ax.scatter(Xqs,Yqs,Zqs)
ax.scatter(Xct,Yct,Zct)

ax.legend()

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
