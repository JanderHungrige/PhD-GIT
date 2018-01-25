# -*- coding: utf-8 -*-
"""
Created on Thu Jan 25 21:46:55 2018

@author: 310122653
"""

""" 
Plottin Shit
"""

from itertools import*
from numpy import *
from pylab import *
import matplotlib.pyplot as plt
plt.style.use('ggplot')     

from numpy import *
from pylab import *
def PlottingFeatureImportance(FimportanceMean_QS,FimportanceMean_CT,FimportanceMean_IS,feature_dict):
#       hist(Fimportance_QS.ravel(),100)
#       hist(FimportanceMean_QS,len(FimportanceMean_QS))
#       hist(FimportanceMean_IS,31)

       y_pos_QS = arange(len(FimportanceMean_QS))
       plt.bar(y_pos_QS,sorted(FimportanceMean_QS,reverse=True),alpha=0.5,label='QS/AS- Features',color=[0,0,0])
       
       y_pos_CT = arange(len(FimportanceMean_CT))
       plt.bar(y_pos_CT,sorted(FimportanceMean_CT,reverse=True),alpha=0.5,label='QS/AS/CTW- Features',color=[0.6,0.6,0.6])
       
       y_pos_IS = arange(len(FimportanceMean_IS))
       plt.bar(y_pos_IS,sorted(FimportanceMean_IS,reverse=True),alpha=0.5,label='QS/AS/IS- Features',color=[0.3,0.3,0.3])
       
       legend()
       xlabel('Features')
       ylabel('Importance')