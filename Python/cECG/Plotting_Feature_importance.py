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
       
       FimportanceMean_QS=mean(Fimportance_QS,0)
       FimportanceMean_CT=mean(Fimportance_CT,0)
#       FimportanceMean_IS=mean(Fimportance_IS,0)
       normFQS=FimportanceMean_QS/max(FimportanceMean_QS)
#       normFIS=FimportanceMean_IS/max(FimportanceMean_IS)
       normFCT=FimportanceMean_CT/max(FimportanceMean_CT)
       
       plt.figure#(figsize=(15.5,5),layout=' tight' )     
       a=0.25
       y_pos_QS = arange(len(normFQS))
       plt.plot(y_pos_QS,sorted(normFQS,reverse=True),color=[0.3,0.3,0.3],linestyle='--' )#,hatch='//')
       
       y_pos_CT = arange(len(normFCT))
       plt.plot(y_pos_CT,sorted(normFCT,reverse=True),color=[0.5,0.5,0.5],linestyle='--')
       
#       y_pos_IS = arange(len(normFIS))
#       plt.plot(y_pos_IS,sorted(normFIS,reverse=True),color=[1,1,1],linestyle='--')
       
       y_pos_QS = arange(len(normFQS))
       plt.bar(y_pos_QS,sorted(normFQS,reverse=True),alpha=0.7,label='QS/AS- Features',color=[0.3,0.3,0.3])#,hatch='//')
       
       y_pos_CT = arange(len(normFCT))
       plt.bar(y_pos_CT,sorted(normFCT,reverse=True),alpha=0.6,label='QS/AS/CTW- Features',color=[0.5,0.5,0.5])
#       
#       y_pos_IS = arange(len(normFIS))
#       plt.bar(y_pos_IS,sorted(normFIS,reverse=True),alpha=0.6,label='QS/AS/IS- Features',color=[1,1,1])
#       

       legend()
       xlabel('Features')
       ylabel('Importance')       
       
 # top=0.963,
#bottom=0.119,
#left=0.043,
#right=0.988,
#hspace=0.2,
#wspace=0.2
#------------------------------------------------       
      
       a=0.25
       y_pos_QS = arange(len(FimportanceMean_QS))
       plt.bar(y_pos_QS+a,sorted(FimportanceMean_QS,reverse=True),width=a,alpha=0.7,label='QS/AS- Features',color=[0.3,0.3,0.3])#,hatch='//')
       
       y_pos_CT = arange(len(FimportanceMean_CT))
       plt.bar(y_pos_CT,sorted(FimportanceMean_CT,reverse=True),width=a,alpha=0.6,label='QS/AS/CTW- Features',color=[0.5,0.5,0.5])
       
       y_pos_IS = arange(len(FimportanceMean_IS))
       plt.bar(y_pos_IS-a,sorted(FimportanceMean_IS,reverse=True),width=a,alpha=0.6,label='QS/AS/IS- Features',color=[1,1,1])
       
       legend()
       xlabel('Features')
       ylabel('Importance')
#------------------------------------------------       
       figure
       a=0.25
       y_pos_QS = arange(len(FimportanceMean_QS))
       plt.bar(y_pos_QS,sorted(FimportanceMean_QS,reverse=True),alpha=0.7,label='QS/AS- Features',color=[0.3,0.3,0.3])#,hatch='//')
       
       y_pos_CT = arange(len(FimportanceMean_CT))
       plt.bar(y_pos_CT,sorted(FimportanceMean_CT,reverse=True),alpha=0.6,label='QS/AS/CTW- Features',color=[0.5,0.5,0.5])
       
       y_pos_IS = arange(len(FimportanceMean_IS))
       plt.bar(y_pos_IS,sorted(FimportanceMean_IS,reverse=True),alpha=0.6,label='QS/AS/IS- Features',color=[1,1,1])
       
       legend()
       xlabel('Features')
       ylabel('Importance')
#------------------------------------------------       


       
       a=0.25
       y_pos_QS = arange(len(FimportanceMean_QS))
       plt.plot(y_pos_QS,sorted(FimportanceMean_QS,reverse=True),color=[0.3,0.3,0.3])#,hatch='//')
       
       y_pos_CT = arange(len(FimportanceMean_CT))
       plt.plot(y_pos_CT,sorted(FimportanceMean_CT,reverse=True),color=[0.5,0.5,0.5])
       
       y_pos_IS = arange(len(FimportanceMean_IS))
       plt.plot(y_pos_IS,sorted(FimportanceMean_IS,reverse=True),color=[1,1,1])
       
       y_pos_QS = arange(len(FimportanceMean_QS))
       plt.bar(y_pos_QS,sorted(FimportanceMean_QS,reverse=True),alpha=0.7,label='QS/AS- Features',color=[0.3,0.3,0.3])#,hatch='//')
       
       y_pos_CT = arange(len(FimportanceMean_CT))
       plt.bar(y_pos_CT,sorted(FimportanceMean_CT,reverse=True),alpha=0.6,label='QS/AS/CTW- Features',color=[0.5,0.5,0.5])
       
       y_pos_IS = arange(len(FimportanceMean_IS))
       plt.bar(y_pos_IS,sorted(FimportanceMean_IS,reverse=True),alpha=0.6,label='QS/AS/IS- Features',color=[1,1,1])
       
       
       legend()
       xlabel('Features')
       ylabel('Importance')       