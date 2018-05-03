# -*- coding: utf-8 -*-
"""
Created on Thu Jan 19 09:40:21 2017

@author: dena
"""

"""
Compute the total IOU according the IOU of the each image
"""
import numpy as np

# read the text file of each image
mainIoU = []
for i in range (801,1001):
    IoUres  = open ('/home/fcn/wordSpotting/icdarCh4_results_Localization_fcnEp63_TNT/img_%d.res.txt'%i).read()
    numWords =  IoUres.count('\n')
    IoUres = IoUres.split('\n')
    if numWords != 0:
        for j in range(0,numWords):
            #Read the last column which is the one for the IoU
            iou = IoUres[j].split(',')[-1]
            #Save it as a one index
            mainIoU.append(float(iou))
        
        
# make the average of all the Indeces
length = len(mainIoU)
print length
IoUsum = np.sum(mainIoU)
print IoUsum
Average = np.divide(np.sum(mainIoU),(len(mainIoU)))
print Average
# write it in a text file for the all the data set
res = open("/home/fcn/wordSpotting/icdarCh4_results_Localization_fcnEp63_TNT/totalIoU.txt", "w")
res.write ("sum of IoU: %f \nlength: %f \nmean IoU: %f\n" %(IoUsum,length, Average))
res.close()