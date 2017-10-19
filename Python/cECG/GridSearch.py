# -*- coding: utf-8 -*-
"""
Created on Wed Oct  4 14:09:18 2017

@author: 310122653
"""

""" 
#GRID SEARCH FOR C AND GAMMA
"""

#from Loading_5min_mat_files_cECG import AnnotMatrix_each_patient, FeatureMatrix_each_patient, Class_dict, features_dict, features_indx
from Classifier_routines import Classifier_routine_no_sampelWeight

import itertools
import numpy 
from matplotlib import *
from numpy import *
from pylab import *
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import StratifiedKFold
from sklearn.cross_validation import cross_val_score
from sklearn.cross_validation import KFold
from sklearn.metrics import accuracy_score 
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc
from sklearn import svm, cross_validation
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.linear_model import Perceptron
import sys #to add strings together
import pdb # use pdb.set_trace() as breakpoint

# input:
# plotting_grid 1 or 0


def GridSearch_all(plotting_grid,gridC,gridY,lst,label,selected_babies,AnnotMatrix_each_patient,FeatureMatrix_each_patient,classweight):
    

    #### Create Matrices for selected babies
    AnnotMatrix_auswahl=[AnnotMatrix_each_patient[k] for k in selected_babies]       # get the annotation values for selected babies
    FeatureMatrix_auswahl=[FeatureMatrix_each_patient[k] for k in selected_babies]   # get the feature values for selected babies
    
    #### SELECT FEATURE SET
    idx=[in1d(AnnotMatrix_each_patient[sb],label) for sb in selected_babies]#.values()]     # which are the idices for AnnotMatrix_each_patient == label
    idx=[nonzero(idx[sb])[0] for sb in range(len(selected_babies))]#.values()]              # get the indices where True
    y_each_patient=[val[idx[sb],:] for sb, val in enumerate(AnnotMatrix_auswahl) if sb in range(len(selected_babies))] #get the values for y from idx and label

    #### CREATING THE DATASET WITH x numbers of features WITH SPECIFIC LABELS FOR SELECTED BABIES  
    Xfeat=[val[:,lst] for sb, val in enumerate(FeatureMatrix_auswahl)] # selecting all Features
    Xfeat=[val[idx[sb],:] for sb, val in enumerate(Xfeat)]   #selecting the datapoints in label
    
  
    differnt_c_y_results=numpy.zeros((len(gridC), len(gridY)))
#    differnt_c_y_results=array([]);differnt_y_results=array([])
    for C in range(len(gridC)):
        for Y in range(len(gridY)):
            print('C:', sep='', end='', flush=True); print(gridC[C], sep='', end='', flush=True)
            print('  Y:', sep='', end='', flush=True); print(gridY[Y], sep='', end='', flush=True)
            result=Classifier_routine_no_sampelWeight(Xfeat,y_each_patient,selected_babies,label,classweight,gridC[C],gridY[Y])\
            [2] # 0-4 at the moment: macro micro weight all kappa #[0] says take only first return value from function
            print(' Result %.3f' % result)
            differnt_c_y_results[C,Y]=result
#            differnt_y_results=[differnt_y_results,result] # create column for each Y
            
#        if not(differnt_c_y_results):# For the first round where there is no variable. Otherwise Matrix error
#            differnt_c_y_results=differnt_y_results
#        else:
#          differnt_c_y_results=column_stack((differnt_c_y_results, differnt_y_results))       # add  collum Y to  

    # finding the row and colum index for the mauimun value
    am = differnt_c_y_results.argmax() # https://stackoverflow.com/questions/11332205/find-row-or-column-containing-maximum-value-in-numpy-array
    c_idx = am % differnt_c_y_results.shape[1] #Collums are gamma Values
    r_idx = am // differnt_c_y_results.shape[1] #Rows are c values
    
    optimal_Y=gridY[c_idx]
    optimal_C=gridC[r_idx]
#    r=differnt_c_y_results.index(max(differnt_c_results))
#    optimal_C=gridC[r]
    
#    sys.exit('Jan')
    if plotting_grid:
        from mpl_toolkits.mplot3d import Axes3D
        gridx=numpy.zeros((len(gridC), len(gridY)))
        gridy=numpy.zeros((len(gridC), len(gridY)))
        for i in range(len(gridY)):
            gridx[:,i]=gridC[:]
        for i in range(len(gridC)):
            gridy[i,:]=gridY[:]
            
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        surf=ax.plot_surface(gridx, gridy, differnt_c_y_results,cmap='viridis')

    c=optimal_C
    gamma=optimal_Y
    return [c,gamma]


def GridSearch_commonFeatures(plotting_grid,gridC,gridY,lst,label,Xfeat,y_each_patient,selected_babies,classweight):

    differnt_c_y_results=numpy.zeros((len(gridC), len(gridY)))
#    differnt_c_y_results=array([]);differnt_y_results=array([])
    for C in range(len(gridC)):
        for Y in range(len(gridY)):
            print('C:', sep='', end='', flush=True); print(gridC[C], sep='', end='', flush=True)
            print('  Y:', sep='', end='', flush=True); print(gridY[Y], sep='', end='', flush=True)
            result=Classifier_routine_no_sampelWeight(Xfeat,y_each_patient,selected_babies,label,classweight,gridC[C],gridY[Y])\
            [2]# 0-4 at the moment: macro micro weight all kappa #[0] says take only first return value from function
            print(' Result %.3f' % result)
            pdb.set_trace()
            differnt_c_y_results[C,Y]=result
#            differnt_y_results=[differnt_y_results,result] # create column for each Y
            
#        if not(differnt_c_y_results):# For the first round where there is no variable. Otherwise Matrix error
#            differnt_c_y_results=differnt_y_results
#        else:
#          differnt_c_y_results=column_stack((differnt_c_y_results, differnt_y_results))       # add  collum Y to  

    # finding the row and colum index for the maximun value
    am = differnt_c_y_results.argmax() # https://stackoverflow.com/questions/11332205/find-row-or-column-containing-maximum-value-in-numpy-array
    c_idx = am % differnt_c_y_results.shape[1] #Colums are gamma Values
    r_idx = am // differnt_c_y_results.shape[1] #Rows are c values
    
    optimal_Y=gridY[c_idx]
    optimal_C=gridC[r_idx]
#    r=differnt_c_y_results.index(max(differnt_c_results))
#    optimal_C=gridC[r]
    
#    sys.exit('Jan')
    if plotting_grid:
        from mpl_toolkits.mplot3d import Axes3D
        gridx=numpy.zeros((len(gridC), len(gridY)))
        gridy=numpy.zeros((len(gridC), len(gridY)))
        for i in range(len(gridY)):
            gridx[:,i]=gridC[:]
        for i in range(len(gridC)):
            gridy[i,:]=gridY[:]
            
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        surf=ax.plot_surface(gridx, gridy, differnt_c_y_results,cmap='viridis')

    c=optimal_C
    gamma=optimal_Y
    return [c,gamma]