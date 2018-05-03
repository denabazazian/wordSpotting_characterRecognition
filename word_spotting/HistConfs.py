# -*- coding: utf-8 -*-
"""
Created on Thu Jan 26 12:52:17 2017

@author: dena
"""

import numpy as np
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt


#read the text files of conf Normalize then plot them 
# in the text file we need the third and the fourth columns

totConf = []
normConf = []

#ConfNormTPGT  = open ('/home/fcn/wordSpotting/GT_TP/ConfNormTPGT.txt').read()
#ConfNormTPGT  = open ('/home/fcn/wordSpotting/PHOC_TNT_ConfTP_fcnEp113/ConfIoU.txt').read()
ConfNormTPGT  = open ('/home/fcn/wordSpotting/PHOC_icdarCh1_results_Localization_fcnEp113_TNT/ConfNorm_GT_TP_RES.txt').read()

numLines =  ConfNormTPGT.count('\n')
confs = ConfNormTPGT.split('\n')
for j in range(0,numLines):  
    
#    confGT = confs[j].split(',')[0]
#    if(np.isnan(float(confGT)) == False):
#      totConf.append(float(confGT))
      
#    confTP = confs[j].split(',')[1]
#    if(np.isnan(float(confTP)) == False):
#      totConf.append(float(confTP))
      
    confRes = confs[j].split(',')[2]
    if(np.isnan(float(confRes)) == False):
      totConf.append(float(confRes))      
          
print 'confTP min:%f'%(np.min(totConf))
print 'confTP max:%f'%(np.max(totConf))

ConfRange = (np.max(totConf)-np.min(totConf))

for i in range(0,len(totConf)):
    normConf.append(((totConf[i])-(np.min(totConf)))/ConfRange)

num_bins = 10    
# the histogram of the confidence 
n, bins, patches = plt.hist(normConf, num_bins, range=[0,1.0], normed = True,  histtype='bar',facecolor='green')
plt.ylim ([0,1.8])
plt.xlabel('confidence')
plt.ylabel('#')
plt.show()










#ConfNormTPGT  = open ('/home/fcn/wordSpotting/PHOC_icdarCh1_results_Localization_fcnEp113_TNT/ConfNorm_GT_TP_RES_10_99.txt').read()
#
#numLines =  ConfNormTPGT.count('\n')
#confs = ConfNormTPGT.split('\n')
#for j in range(0,numLines):  
#    
#    confGT = confs[j].split(',')[0]
#    if(np.isnan(float(confGT)) == False):
#      totConf.append(float(confGT))
#
##    confTP = confs[j].split(',')[1]
##    if(np.isnan(float(confTP)) == False):
##      totConf.append(float(confTP))
#      
##    confRes = confs[j].split(',')[2]
##    if(np.isnan(float(confRes)) == False):
##      totConf.append(float(confRes))
#
#
#
#ConfNormTPGT  = open ('/home/fcn/wordSpotting/PHOC_icdarCh1_results_Localization_fcnEp113_TNT/ConfNorm_GT_TP_RES_100_410.txt').read()
#
#numLines =  ConfNormTPGT.count('\n')
#confs = ConfNormTPGT.split('\n')
#for j in range(0,numLines):  
#    
#    confGT = confs[j].split(',')[0]
#    if(np.isnan(float(confGT)) == False):
#      totConf.append(float(confGT))      
#                
##    confTP = confs[j].split(',')[1]
##    if(np.isnan(float(confTP)) == False):
##      totConf.append(float(confTP))
#      
##    confRes = confs[j].split(',')[2]
##    if(np.isnan(float(confRes)) == False):
##      totConf.append(float(confRes))






