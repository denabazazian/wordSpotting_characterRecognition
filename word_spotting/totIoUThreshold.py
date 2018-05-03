# -*- coding: utf-8 -*-
"""
Created on Thu Jan 26 18:13:32 2017

@author: dena
"""


##### Computing Accuracy for different thresholds of IoU ############
#import numpy as np
#
#IoUThreshold = []
#IoUtot = []
#    # read the IoUs 
#    ##IoUres  = open ('/home/fcn/wordSpotting/PHOC_TNT_ConfTP_fcnEp113/ConfIoU.txt').read()
#for i in range (1,411):
#    IoUres  = open ('/home/fcn/wordSpotting/PHOC_icdarCh1_results_Localization_fcnEp113_TNT/img_%d.txt'%i).read()
#    #Read them one by one
#    # meke the threshold for each one 
#    #then append them to another list
#    numLines =  IoUres.count('\n')
#    IoU = IoUres.split('\n')
#    for j in range(0,numLines):
#        iou = IoU[j].split(',')[9]
#        IoUtot.append(float(iou))
#        if (float(iou) > 0.0000):
#            IoUThreshold.append(1.000)
#        else:
#            IoUThreshold.append(0.000)
# 
#
####Write all the IoUs       :
##AllIoUs = open("/home/fcn/wordSpotting/PHOC_icdarCh1_results_Localization_fcnEp113_TNT/AllIoUs.txt", "w")
##for i in range(0, len(IoUtot)):
##    AllIoUs.write("%f\n"%(IoUtot[i]))
##AllIoUs.close()
#
#    #compute the average 
#length = len(IoUThreshold)
#print 'length is %f'%(length)
#IoUsum = np.sum(IoUThreshold)
#print 'sum is %f'%(IoUsum)
#Average = np.divide(np.sum(IoUThreshold),(len(IoUThreshold)))
#print 'mean IoU is %f'%(Average)

#####ploting IoU ############

import matplotlib.pyplot as plt


y = [0.555774, 0.549874 , 0.533296 ,0.500421, 0.447317 , 0.394493 , 0.327058 , 0.267772, 0.197808 ,0.132903, 0.077831]
x = [ 0.000, 0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]

fig = plt.figure()

ax1 = fig.add_subplot(111)

ax1.set_title("IoUs different thresholds")    
ax1.set_xlabel('IoU')
ax1.set_ylabel('Accuracy')

#ax1.plot(x,y, c='r', label='the data')
ax1.plot(x,y, c='r')

leg = ax1.legend()
plt.grid()
plt.show()

##y = [0.519539, 0.500984 , 0.455721 ,0.392747, 0.332584 , 0.271577 , 0.218161 , 0.171774 , 0.126511 ,0.080686,0.046387] ICDAR-Challenge1_version1
##y = [0.508108, 0.477838 , 0.431351 ,0.362162, 0.301622 , 0.227027 , 0.155676 , 0.089730 , 0.030270 ,0.006486,0.001081] ICDAR-challeng2
##y = [ 0.522162,0.567568 ,0.637838 , 0.698378 ,0.772973 , 0.844324,0.910270 , 0.969730,0.993514]  ICDAR-challeng2_reverse