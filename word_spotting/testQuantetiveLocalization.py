#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 29 17:41:52 2016

@author: dena
"""
# python /home/dena/Projects/CharacterDetection/quantetiveResults/testQuantetiveLocalization.py /home/fcn/wordSpotting/icdarCh4_val_fcnEp63/img_*.mat
# cd /home/dena/Projects/CharacterDetection/quantetiveResults
# python testQuantetiveLocalization.py /home/fcn/wordSpotting/icdarCh4_val_fcnEp63/img_*.mat

import scipy
import scipy.io
import sys
import numpy as np
from matplotlib import pyplot as plt
import cv2
from collections import defaultdict
from commands import getoutput as go
import glob, os
import re


def getIimg(matFname,standarise=False):
    matData=scipy.io.loadmat(matFname)['scores']
    expData=np.exp(matData)
    sumData=expData.sum(axis=2)
    #for k in range(exp)
    expData/=expData.sum(axis=2)[:,:,None]
    #print 'MAX PIXEL:',matData.sum(axis=2).max()
    #print 'MAT SHAPE:',matData.shape
    expData/=expData.sum(axis=2)[:,:,None]
    matData=expData
    if standarise:
       for layerNum in range(matData.shape[2]):
           matData[:,:,layerNum]-=matData[:,:,layerNum].mean()
           matData[:,:,layerNum]/=matData[:,:,layerNum].std()
       iimg=matData.cumsum(axis=0).cumsum(axis=1)
       
    return iimg

# read all the word one by one in the text 



alphabet="#abcdefghijklmnopqrstuvwxyz1234567890@"
def getQueryHist(qword,minusOneForMissing=False):
    qHist=np.zeros([len(alphabet)])
    m=defaultdict(lambda: len(alphabet)-1)
    m.update({alphabet[k]:k for k in range(1,len(alphabet))})
    for c in qword.lower():
        qHist[m[c]]+=1
    if minusOneForMissing:
        qHist[qHist==0]=-1
    return qHist[:]


def textProposals(iimg,qHist):
    idxTP=0

    #lines=[l.strip().split(',') for l in open((fname+'.csv')).read().split('\n') if len(l)>0]
    lines=[l.strip().split(',') for l in open(csvName).read().split('\n') if len(l)>0]       
    BB = np.empty([len(lines),5], dtype='f')
    for lineNum in range(0,len(lines)):
        for colNum in range(0,4):
            BB[lineNum,colNum] = lines[lineNum][colNum] #converting the list of list to a numpy array
     

       
    BBsrt = BB[np.argsort(BB[:,-1]),:] #sorting proposals by the scores
    #BB[np.argsort(BB[:,-1])[:500],:] 
    resLTRB=np.empty([len(BBsrt),5])
    BBEnergy=np.empty([len(BBsrt),38])
    BBEnergyNorm=np.empty([len(BBsrt),38])
    qHistW=np.zeros(38)
    qHistNorm=np.zeros(38)
    
    for rr in range(0,len(BBsrt)):
        left = BBsrt[rr,0]
        top = BBsrt[rr,1]
        right = left + BBsrt[rr,2] #right = left + width
        bottom = top + BBsrt[rr,3] #bottom = top - height
        surface =  BBsrt[rr,2] * BBsrt[rr,3]       
        #print 'left: %s, top:%s, right:%s, bottom:%s' %(left,top,right,bottom)
        if right>= iimg.shape[1]:
            right = iimg.shape[1]-1
        if top>= iimg.shape[0]:
            top = iimg.shape[0]-1        
        
        enrgyVect=iimg[bottom,right,:]+iimg[top,left,:]-(iimg[bottom,left,:]+iimg[top,right,:])

        enrgyVect/=surface
        BBEnergy[idxTP,:] = enrgyVect 
        #resLTRB[idxTP,:]=[left,top,right,bottom,np.sum(enrgyVect[1:]*qHist[1:])]
        idxTP+=1
        
    #L1Normalization    
    for rr in range(0,len(BBsrt)):
        BBW = np.sum(BBEnergy[rr,:])
        BBEnergyNorm[rr,:] = BBEnergy[rr,:]/BBW
  
    qHistW = np.sum(qHist)
    qHistNorm = qHist/qHistW
     
    ### Histogram Intersection   
    HistIntersection = np.zeros((len(BBEnergyNorm),1))
         
    qHistNorma = np.zeros((1,len(qHistNorm)))
    for ll in range(0, len(qHistNorm)):
        qHistNorma[0,ll] = qHistNorm[ll]
     
    BBEnergyNorma = np.zeros((1,len(qHistNorm)))
    for rr in range(0,len(BBEnergyNorm)):
        for ll in range(0, len(qHistNorm)):
            BBEnergyNorma[0,ll] = BBEnergyNorm[rr][ll]

        dConcatenate = np.concatenate((BBEnergyNorma,qHistNorma), axis =0)
        HistIntersection[rr] = np.sum(np.min(dConcatenate, axis =0))/np.sum(dConcatenate)
    
    #resLTRB=[BBsrt[:,0], BBsrt[:,1] , BBsrt[:,2]+BBsrt[:,0] , BBsrt[:,3]+BBsrt[:,1] , np.min(dConcatenate, axis =0)/np.sum(dConcatenate)]
    resLTRBInter=np.empty([BBsrt.shape[0],5])
    resLTRBInter[:,0]= BBsrt[:,0]
    resLTRBInter[:,1]= BBsrt[:,1]
    resLTRBInter[:,2]= BBsrt[:,2]+BBsrt[:,0]
    resLTRBInter[:,3]= BBsrt[:,3]+BBsrt[:,1] 
    resLTRBInter[:,4]= HistIntersection[:,0]
    #resLTRB=[BBsrt[:,0], BBsrt[:,1] , BBsrt[:,2]+BBsrt[:,0] , BBsrt[:,3]+BBsrt[:,1] , HistIntersection[:,0]]
    return resLTRBInter
    

def plotRectangles(rects,transcriptions,bgrImg,rgbCol):
    bgrCol=np.array(rgbCol)[[2,1,0]]
    res=bgrImg.copy()
    pts=np.empty([rects.shape[0],5,1,2])
    if rects.shape[1]==4:
        x=rects[:,[0,2,2,0,0]]
        y=rects[:,[1,1,3,3,1]]
    elif rects.shape[1]==8:
        x=rects[:,[0,2,4,6,0]]
        y=rects[:,[1,3,5,7,1]]
    else:
        print rects
        raise Exception()
    pts[:,:,0,0]=x
    pts[:,:,0,1]=y
    pts=pts.astype('int32')
    ptList=[pts[k,:,:,:] for k in range(pts.shape[0])]
    if not (transcriptions is None):
        for rectNum in range(rects.shape[0]):
            res=cv2.putText(res,transcriptions[rectNum],(rects[rectNum,0],rects[rectNum,1]),1,cv2.FONT_HERSHEY_PLAIN,bgrCol)
    res=cv2.polylines(res,ptList,False,bgrCol)
    return res



if __name__=='__main__':
    ### get the directory of the sourceImages, matfiles, CSV, vocabulary
    #print "length of the sysargv, numer of images in this experiment is %d"%len(sys.argv[1:])
    imgNum = 0 
    mainIOU = np.zeros((1,(len(sys.argv[1:]))))    
    for mat_name in sys.argv[1:]:
        #sys.argv -> mat_name = '/home/fcn/wordSpotting/icdarCh4_val_fcnEp63/img_804.mat'        
        print mat_name

        ### Read a mat file to get the integral image and the heat map from softmax
        iimg=getIimg(mat_name,standarise=True)
        #get the vocabulary for the correspond image
        #voc  = open (mat_name.split('.')[0]+'.voc100.txt').read()
        #Read the vocabulary from the gt
        voc  = open (mat_name.split('.')[0]+'.gt.txt').read()
        numQWords =  voc.count('\n')
        voc=re.sub(r'[^\x00-\x7f]',r'',voc)
        voc = voc.split('\r\n')
        csvName = mat_name.split('.')[0]+'.csv'
        #print csvName
        fileName = mat_name.split('.')[0].split('/')[-1]
        #textRes = '/home/fcn/wordSpotting/icdarCh4_results_fcnEp63/'+fileName+'.res.txt'
        textRes = '/home/fcn/wordSpotting/icdarCh4_results_Localization_fcnEp63/'+fileName+'.res.txt'
        #textIOU = '/home/fcn/wordSpotting/icdarCh4_results_Localization_fcnEp63/'+fileName+'.iou.txt'
        subIOU = np.zeros((1,numQWords))
        #with open(textIOU, "w") as iou_file:
        with open(textRes, "w") as res_file:  
            img=cv2.imread(mat_name.split('.')[0]+'.jpg')    
            cv_size = lambda img: tuple(img.shape[1::-1])
            width, height = cv_size(img)
            #fig = plt.figure() #figsize=(width, height),frameon=False
            #plt.imshow(img) 
            idnqw = 0                              
            for nqw in range(0,numQWords): 
                #qword = voc[nqw]
                if (len(voc[nqw])>0 and voc[nqw].split(',')[-1].strip()!='###'):
                   idnqw +=1
                   qword = voc[nqw].split(',')[-1]
                   print 'query word is %s'%qword
                   qHist = getQueryHist(qword)
                   #print qHist
                   res=textProposals(iimg,qHist)
                   #print res
                   surf=(res[:,0]-res[:,2])*(res[:,1]-res[:,3])
                   res=res[surf>1200,:]
                   idx=np.argsort(res[:,4])
                   #print 'idx[-1] devided by surface is %f'%np.multiply((np.divide(idx[-1],surf)),100)
                   #if (idx[-1] > 7800.0000):
                   [l,t,r,b]=res[idx[-1],:4]                   
                   #print 'idx[-1]is %f'%idx[-1]
                   #print 'surface is %f'%((t-b)*(l-r))
                   #print 'idx[-1] devided by surface is %f'%((idx[-1]/((t-b)*(l-r)))*100)
                                      
                   gtLeft=np.min([int(voc[nqw].split(',')[0]),int(voc[nqw].split(',')[6])],axis=0)
                   gtTop=np.min([int(voc[nqw].split(',')[1]),int(voc[nqw].split(',')[3])],axis=0)
                   gtRight=np.max([int(voc[nqw].split(',')[2]),int(voc[nqw].split(',')[4])],axis=0)
                   gtBottom=np.max([int(voc[nqw].split(',')[5]),int(voc[nqw].split(',')[7])],axis=0)
                   gtWidth=np.absolute(gtRight-gtLeft)
                   gtHeight=np.absolute(gtTop-gtBottom)
                   resLeft=int(l)
                   resTop=int(t)
                   resRight=int(r)
                   resBottom=int(b)
                   resWidth=np.absolute(resRight-resLeft)
                   resHeight=np.absolute(resTop-resBottom)
                   intL=np.max([resLeft,gtLeft],axis=0)
                   intT=np.max([resTop,gtTop],axis=0)
                   intR=np.min([resRight,gtRight],axis=0)
                   intB=np.min([resBottom,gtBottom],axis=0)
                   intW=(intR-intL)+1
                   if intW<0:
                       intW = 0
                   else:
                       intW=intW
                   #intW[intW<0]=0 #'numpy.int64' object does not support item assignment
                   intH=(intB-intT)+1
                   if intH<0:
                       intH = 0
                   else:
                       intH=intH
                   #intH[intH<0]=0 #'numpy.int64' object does not support item assignment
                   I=intH*intW
                   U=resWidth*resHeight+gtWidth*gtHeight-I
                   IoU=I/(U+.0000000001)
                   print " the subIoU is %f"%IoU
                   subIOU[0][nqw] = IoU
                   #iou_file.write("%f\n"%IoU)                   
                   res_file.write("%d,%d,%d,%d,%d,%d,%d,%d,%s,%f\n" % (l,t,r,t,r,b,l,b,qword,IoU))                    
                   #plt.plot([l,l,r,r,l],[t,b,b,t,t], color='green', lw=2)
                   #plt.text(l+5,t+5,qword,size=8, color='green' )
                   
                   pts = np.array([[int(voc[nqw].split(',')[0]),int(voc[nqw].split(',')[1])],[int(voc[nqw].split(',')[2]),int(voc[nqw].split(',')[3])],[int(voc[nqw].split(',')[4]),int(voc[nqw].split(',')[5])],[int(voc[nqw].split(',')[6]),int(voc[nqw].split(',')[7])],[int(voc[nqw].split(',')[0]),int(voc[nqw].split(',')[1])]], np.int32)
                   pts = pts.reshape((-1,1,2))
                   cv2.polylines(img,[pts],True,(0,0,255),2)       
                   #cv2.rectangle(img,(gtLeft,gtTop),(gtRight,gtBottom),(0,0,255),2)
                   cv2.putText(img,qword,(gtLeft,gtTop),cv2.FONT_HERSHEY_PLAIN,1,(0,0,255),2)
                   cv2.rectangle(img,(resLeft,resTop),(resRight,resBottom),(0,255,0),2)
                   cv2.putText(img,qword,(resLeft,resTop),cv2.FONT_HERSHEY_PLAIN,1,(0,255,0),2)
                   
                   """
                   cv2.imwrite('/home/fcn/wordSpotting/icdarCh4_results_Localization_fcnEp63/'+fileName+'.png',img)
                   """
                   #print subIOU
                   
            if (idnqw!=0):       
                mainIOU[0][imgNum] = np.divide(np.sum(subIOU),idnqw)
                print "the IoU of is %f"%mainIOU[0][imgNum]
            else:
                mainIOU[0][imgNum] = 1.0000000
                """
                cv2.imwrite('/home/fcn/wordSpotting/icdarCh4_results_Localization_fcnEp63/'+fileName+'.png',img)
                """
        imgNum +=1                      
                   
                   
    print "The total IoU is %f"%np.divide(np.sum(mainIOU),(imgNum))          #imgNum-1 or imgNum+1         
                   
                   
                   

            #plt.savefig('/home/fcn/wordSpotting/icdarCh4_results_Localization_fcnEp63/'+fileName+'.png')
            #plt.show()





            
            

### Write an image with rectangles with openCV
#                   qRect = np.empty([1,8],dtype='int32')
#                   qRect[0][0]= l
#                   qRect[0][0]= t
#                   qRect[0][0]= l
#                   qRect[0][0]= b
#                   qRect[0][0]= r
#                   qRect[0][0]= b
#                   qRect[0][0]= r
#                   qRect[0][0]= t                   
#                   #qRect= [l,l,r,r,l],[t,b,b,t,t]
#                   plt.imshow(img)
#                   plt.show()
#                   plotRectangles(qRect,qword,img,[0,255,0])                                      
#            cv2.imwrite('/home/fcn/wordSpotting/icdarCh4_results_Localization_fcnEp63/'+fileName+'.png',img)









        #matfiles = os.path.join(dname, 'img_*.mat')
        #vocfiles = os.path.join(dname, 'img_*.voc100.txt')
        #sourcefiles = os.path.join(dname, '*.jpg')

        #matNum = len (glob.glob(matfiles))
        #jpgNum = len (glob.glob(sourcefiles))
        #vocNum = len (glob.glob(vocfiles))
        #print 'number of matfiles',matNum
        #for mat_name in glob.glob(matfiles):
            #if mat_name.endswith('.mat'):


















