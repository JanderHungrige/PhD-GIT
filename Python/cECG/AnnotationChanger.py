# -*- coding: utf-8 -*-
"""
Created on Wed Nov 22 09:11:21 2017

@author: 310122653
"""
def AnnotationChanger(AnnotMatrix_each_patient,LoosingAnnot5,LoosingAnnot6,LoosingAnnot6_2,Smoothing_short,Pack4, direction6, merge34):

#%% This part changes all annotations 5 (Not annotatable) into the value of the surrounding state. 
# The rules are: 
#if sourrounded by the same state; if shorter than 40 epochs change it 
       # OR
# IF the states are not the same but the state 5 is very short (<10) change it to the following state 
       if LoosingAnnot5:
              count5=0
              before5=0
              for l in range(len(AnnotMatrix_each_patient)):
                     for  M in range(len(AnnotMatrix_each_patient[l])):
                            if (AnnotMatrix_each_patient[l][M]==5) and count5==0:
                                   count5=count5+1
                                   before5=int(AnnotMatrix_each_patient[l][M-1])# save the state before state 5
                            elif (AnnotMatrix_each_patient[l][M]==5) and count5!=0:
                                   count5=count5+1
                            elif AnnotMatrix_each_patient[l][M]!=5 and count5!=0 and count5<=40 and before5==int(AnnotMatrix_each_patient[l][M]): # if 5 is inbewteen the same state; and if Not annotatable is on y 10 long(5min)
                                   AnnotMatrix_each_patient[l][M-count5:M]=int(AnnotMatrix_each_patient[l][M])
                                   count5=0 
                                   before5=0  
                            elif AnnotMatrix_each_patient[l][M]!=5 and count5!=0 and count5<=10: # if there is another state following, but the duration of 5 is very short, still change to the following state
                                   AnnotMatrix_each_patient[l][M-count5:M]=int(AnnotMatrix_each_patient[l][M])
                                   count5=0 
                                   before5=0  

#%% This part changes all annotatons if they are 6 (IS) into eihter the followong or the previouse state (depending on variable direction)
                                                     
       if LoosingAnnot6:
              count6=0;
              before6=0
              for l in range(len(AnnotMatrix_each_patient)):
                     for  M in range(len(AnnotMatrix_each_patient[l])):
                            if (AnnotMatrix_each_patient[l][M]==6) and count6==0: 
                                   count6=count6+1
                                   before6=int(AnnotMatrix_each_patient[l][M-1]) # save the state before state 6
                            elif (AnnotMatrix_each_patient[l][M]==6) and count6!=0 and M!=len(AnnotMatrix_each_patient[l])-1: # and not at the end
                                   count6=count6+1
                            elif (AnnotMatrix_each_patient[l][M]!=6 and count6!=0) or (M==len(AnnotMatrix_each_patient[l])-1 and count6!=0): # len is for state at the end
                                   if direction6 and before6!=5:# replace with value before state 6 if not 5 otherwise use exeption and replace with the value after
                                          AnnotMatrix_each_patient[l][M-count6:M]=before6
                                          if M==len(AnnotMatrix_each_patient[l])-1:# if at the end, M is one smaller than len due to range starting from 0
                                                 AnnotMatrix_each_patient[l][M-count6:M+1]=before6
                                          count6=0
                                          before6=0                                   
                                   else:#replace with value after state 6
                                          AnnotMatrix_each_patient[l][M-count6:M]=int(AnnotMatrix_each_patient[l][M])
                                          count6=0
# In this part, similar to the above, the state 6 is changed to the following or the previouse state (depending on variable direction) 
# but here 6 is always changed into 2 if the state begofere 6 was 1 and the one after is 2 to increase QS                                  
#(If before state 6 the state is 1 and following is 2 choose 2. Eerything else is like in Loosing)                           
       if LoosingAnnot6_2: 
              count6=0;
              before6=0
              for l in range(len(AnnotMatrix_each_patient)):
                     for  M in range(len(AnnotMatrix_each_patient[l])):
                            if (AnnotMatrix_each_patient[l][M]==6) and count6==0: 
                                   count6=count6+1
                                   before6=int(AnnotMatrix_each_patient[l][M-1]) # save the state before state 6
                            elif (AnnotMatrix_each_patient[l][M]==6) and count6!=0 and M!=len(AnnotMatrix_each_patient[l])-1: # and not at the end
                                   count6=count6+1
                            elif (AnnotMatrix_each_patient[l][M]!=6 and count6!=0) or (M==len(AnnotMatrix_each_patient[l])-1 and count6!=0): # len is for state at the end
                                   if direction6 and before6==1 and AnnotMatrix_each_patient[l][M]==2:# replace with value before state 6 if not 5 otherwise use exeption and replace with the value after
                                          AnnotMatrix_each_patient[l][M-count6:M]=2
                                          if M==len(AnnotMatrix_each_patient[l])-1:# if at the end, M is one smaller than len due to range starting from 0
                                                 AnnotMatrix_each_patient[l][M-count6:M+1]=2
                                          count6=0
                                          before6=0 
                                   elif direction6 and before6!=5:# replace with value before state 6 if not 5 otherwise use exeption and replace with the value after
                                          AnnotMatrix_each_patient[l][M-count6:M]=before6
                                          if M==len(AnnotMatrix_each_patient[l])-1:# if at the end, M is one smaller than len due to range starting from 0
                                                 AnnotMatrix_each_patient[l][M-count6:M+1]=before6
                                          count6=0
                                          before6=0                                   
                                   else:#replace with value after state 6
                                          AnnotMatrix_each_patient[l][M-count6:M]=int(AnnotMatrix_each_patient[l][M])
                                          count6=0    
                                         
#%% Here we are looking for particular short periods of states to smoothen them out.
# if the duration of a state is longer than 5, remember that state. IF it is shorter than 3, replace with the remembered state.                                    
       if Smoothing_short:
              countSS=0
              for l in range(len(AnnotMatrix_each_patient)):
                     for M in range(1,len(AnnotMatrix_each_patient[l])):# 1 as we want to compare M and M-1, so rnge should start at 1 and not 0
                            if int(AnnotMatrix_each_patient[l][M])==int(AnnotMatrix_each_patient[l][M-1]):
                                   countSS=countSS+1
                            elif (int(AnnotMatrix_each_patient[l][M])!=int(AnnotMatrix_each_patient[l][M-1])) and countSS!=0:
                                   if countSS>5:
                                          rememberSS=int(AnnotMatrix_each_patient[l][M-1])
                                   if countSS<=3:
                                          AnnotMatrix_each_patient[l][M-1-countSS:M]=rememberSS#int(AnnotMatrix_each_patient[l][M])
                                   countSS=0       
                                   countSS=countSS+1

#%% This aims towards state 4 (care taking). CAretakig is sometimes scattered in a short duration. MEaning many changes bewteen 4 and another state. 
#This groups can be merged to on stat 4 as we believe that the HR does not return to "normal" quickly between two state 4 epochs. 
#THIS IS NOT FINISHED                            
       if Pack4:
              answer=[]
              count4=0
              for l in range(len(AnnotMatrix_each_patient)):
                     for M in range(len(AnnotMatrix_each_patient[l])): #stepping 5 min 10*30s
                           AAA=list(AnnotMatrix_each_patient[l]).index(4)
       if merge34:
              for l in range(len(AnnotMatrix_each_patient)):
                     for M in range(1,len(AnnotMatrix_each_patient[l])):# 1 as we want to compare M and M-1, so rnge should start at 1 and not 0
                            if int(AnnotMatrix_each_patient[l][M])==3:
                                   AnnotMatrix_each_patient[l][M]=4
                                                
                           
       return  AnnotMatrix_each_patient                   