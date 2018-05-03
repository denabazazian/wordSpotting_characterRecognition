# -*- coding: utf-8 -*-
"""
Created on Fri Jan 20 18:02:35 2017

@author: dena
"""

# cd /home/dena/Projects/CharacterDetection/quantetiveResults
# python CompareGtwithTP.py /home/fcn/wordSpotting/icdarCh4_val_fcnEp113/img_???.mat

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
    else:
        iimg=matData.cumsum(axis=0).cumsum(axis=1)   
        
       
    return iimg

alphabet="#abcdefghijklmnopqrstuvwxyz1234567890@"
def getQueryHistHOC2(qword,minusOneForMissing=False):
    qHist=np.zeros([len(alphabet)*3])
    m=defaultdict(lambda: len(alphabet)-1)
    m.update({alphabet[k]:k for k in range(1,len(alphabet))})
    for c in qword.lower():
        qHist[m[c]]+=1
    for c in qword[:len(qword)/2].lower():
        qHist[38+m[c]]+=1
    for c in qword[len(qword)/2:].lower():
        qHist[2*38+m[c]]+=1
    if (len(qword))%2:
       qHist[38+m[qword[len(qword)/2]]]+=.5
       qHist[2*38+m[qword[len(qword)/2]]]-=.5
    #pos=-2
    #print '\n'.join([str(k)+'\t'+alphabet[k]+'\t'+str(qHist[k,pos])+'\t'+str(qHist[k+38,pos])+'\t'+str(qHist[k+2*38,pos]) for k in range(38)])

#    if minusOneForMissing:
#       qHist[res==0]=-1
    return qHist[:]
    
def textProposals(iimg,iimgTNT,qHist):
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
    BBEnergy=np.empty([len(BBsrt),38*3])
    BBEnergyNorm=np.empty([len(BBsrt),38*3])
    qHistW=np.zeros(38*3)
    qHistNorm=np.zeros(38*3)
    
    for rr in range(0,len(BBsrt)):
        left = BBsrt[rr,0]
        top = BBsrt[rr,1]
        right = left + BBsrt[rr,2] #right = left + width
        bottom = top + BBsrt[rr,3] #bottom = top + height
        surface =  BBsrt[rr,2] * BBsrt[rr,3]       
        #print 'left: %s, top:%s, right:%s, bottom:%s' %(left,top,right,bottom)
        if right>= iimg.shape[1]:
            right = iimg.shape[1]-1
        if top>= iimg.shape[0]:
            top = iimg.shape[0]-1    
            
############################### HOC1 #########################################################################        
#        enrgyVect=iimg[bottom,right,:]+iimg[top,left,:]-(iimg[bottom,left,:]+iimg[top,right,:])
#        enrgyVect/=surface
#        
#        enrgyVectTNT =iimgTNT[bottom,right,:]+iimgTNT[top,left,:]-(iimgTNT[bottom,left,:]+iimgTNT[top,right,:])
#        enrgyVectTNT/=surface
#        
#        
#        enrgyVect[0] *= enrgyVectTNT[0]
#        for envct in range(1,38):
#            enrgyVect[envct] *= enrgyVectTNT[1]
############################### HOC1 ######################################################################### 
      
      
############################### HOC2 #########################################################################      
        cw=(left+right)/2
        energyVect=np.empty([1,38*3])
        energyVect[0,:38]=iimg[bottom,right,:]+iimg[top,left,:]-(iimg[bottom,left,:]+iimg[top,right,:])
        energyVect[0,38:2*38]=iimg[bottom,cw,:]+iimg[top,left,:]-(iimg[bottom,left,:]+iimg[top,cw,:])      
        energyVect[0,38*2:]=iimg[bottom,right,:]+iimg[top,cw,:]-(iimg[bottom,cw,:]+iimg[top,right,:])
        energyVect/=surface
        #energyVect/=energyVect.sum()
            #print 'SHAPES:',energyVect.shape,qHist.shape
            #print (qHist/qHist.sum(axis=0)[None,:]).sum(axis=0)
        #validRange=np.array(range(1,37)+range(38+1,2*38-1)+range(38*2+1,3*38-1),dtype='int32')
        #enVec=energyVect.reshape([1,-1])[:,validRange]
      ############################### add information of text non text ############################### 
        cw=(left+right)/2
        energyVectTNT=np.empty([1,2*3])
        energyVectTNT[0,0:2]=iimgTNT[bottom,right,:]+iimgTNT[top,left,:]-(iimgTNT[bottom,left,:]+iimgTNT[top,right,:])
        energyVectTNT[0,2:4]=iimgTNT[bottom,cw,:]+iimgTNT[top,left,:]-(iimgTNT[bottom,left,:]+iimgTNT[top,cw,:])      
        energyVectTNT[0,4:6]=iimgTNT[bottom,right,:]+iimgTNT[top,cw,:]-(iimgTNT[bottom,cw,:]+iimgTNT[top,right,:])
        energyVectTNT/=surface
        #energyVect/=energyVect.sum()
            #print 'SHAPES:',energyVect.shape,qHist.shape
            #print (qHist/qHist.sum(axis=0)[None,:]).sum(axis=0)
        #validRangeTNT=np.array(range(1,37)+range(38+1,2*38-1)+range(38*2+1,3*38-1),dtype='int32')
        #enVecTNT=energyVectTNT.reshape([1,-1])[:,validRangeTNT]
        energyVect[0,0] *= energyVectTNT[0,0]
        for envct in range(1,38):
            energyVect[0,envct] *= energyVectTNT[0,1]
            
        energyVect[0,38] *= energyVectTNT[0,2]
        for envct in range(39,38*2):
            energyVect[0,envct] *= energyVectTNT[0,3]
            
        energyVect[0,38*2] *= energyVectTNT[0,4]
        for envct in range(77,38*3):
            energyVect[0,envct] *= energyVectTNT[0,5]
       
############################### HOC2 #########################################################################  
       
        BBEnergy[idxTP,:] = energyVect 
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
    

# get the ground truth and the most similar bounding box 
def getBTP(gtB,tpFile):
    
    gtLeft = gtB[0][0]
    gtTop = gtB[0][1]
    gtRight = gtB[0][2]
    gtBottom = gtB[0][3]
    gtWidth = np.absolute(gtRight - gtLeft) 
    gtHeight =  np.absolute(gtTop - gtBottom) 
        
    # compute the intersection over union of all the textProposals and the GT to find the best text proposal as the one which hast he highest IoU
    lines=[l.strip().split(',') for l in open(tpFile).read().split('\n') if len(l)>0]       
    BB = np.empty([len(lines),5], dtype='f')
    for lineNum in range(0,len(lines)):
        for colNum in range(0,4):
            BB[lineNum,colNum] = lines[lineNum][colNum] #converting the list of list to a numpy array
         
    BBsrt = BB[np.argsort(BB[:,-1]),:] #sorting proposals by the scores    
    
    IoUGtTp = np.zeros((1,len(BBsrt)))
    for rr in range(0,len(BBsrt)):
        tpLeft = BBsrt[rr,0]
        tpTop = BBsrt[rr,1]
        tpRight = tpLeft + BBsrt[rr,2] #right = left + width
        tpBottom = tpTop + BBsrt[rr,3] #bottom = top + height
        tpWidth = BBsrt[rr,2]
        tpHeight = BBsrt[rr,3]
        #surface =  BBsrt[rr,2] * BBsrt[rr,3]       
        #print 'left: %s, top:%s, right:%s, bottom:%s' %(left,top,right,bottom)
        if tpRight>= iimg.shape[1]:
            tpRight = iimg.shape[1]-1
        if tpTop>= iimg.shape[0]:
            tpTop = iimg.shape[0]-1  
                        
        intL=np.max([tpLeft,gtLeft],axis=0)
        intT=np.max([tpTop,gtTop],axis=0)
        intR=np.min([tpRight,gtRight],axis=0)
        intB=np.min([tpBottom,gtBottom],axis=0)
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
        U=tpWidth*tpHeight+gtWidth*gtHeight-I
        IoU=I/(U+.0000000001)  
            
       
        IoUGtTp[0][rr] = IoU

    nbtp = np.argmax(IoUGtTp)
    btp = np.zeros((1,4))
    btp[0][0] = BBsrt[nbtp,0] #left
    btp[0][1] = BBsrt[nbtp,1] #top
    btp[0][2] = BBsrt[nbtp,0] + BBsrt[nbtp,2] #right = = left + width
    btp[0][3] = BBsrt[nbtp,1] + BBsrt[nbtp,3] #bottom  = top + height
    print 'the most similar TP with GT is found'  
    return btp
    
def getConfidence(iimg,iimgTNT,qHist,bb):
    
    left = bb[0][0]
    top = bb[0][1]
    right = bb[0][2]
    bottom = bb[0][3]
    surface = (np.absolute(right-left))*(np.absolute(bottom-top))
    
    if right>= iimg.shape[1]:
       right = iimg.shape[1]-1
    if top>= iimg.shape[0]:
       top = iimg.shape[0]-1 
    
    
    cw=(left+right)/2
    energyVect=np.empty([1,38*3])
    energyVect[0,:38]=iimg[bottom,right,:]+iimg[top,left,:]-(iimg[bottom,left,:]+iimg[top,right,:])
    energyVect[0,38:2*38]=iimg[bottom,cw,:]+iimg[top,left,:]-(iimg[bottom,left,:]+iimg[top,cw,:])      
    energyVect[0,38*2:]=iimg[bottom,right,:]+iimg[top,cw,:]-(iimg[bottom,cw,:]+iimg[top,right,:])
    energyVect/=surface
        #energyVect/=energyVect.sum()
            #print 'SHAPES:',energyVect.shape,qHist.shape
            #print (qHist/qHist.sum(axis=0)[None,:]).sum(axis=0)
        #validRange=np.array(range(1,37)+range(38+1,2*38-1)+range(38*2+1,3*38-1),dtype='int32')
        #enVec=energyVect.reshape([1,-1])[:,validRange]
      ############################### add information of text non text ############################### 
    cw=(left+right)/2
    energyVectTNT=np.empty([1,2*3])
    energyVectTNT[0,0:2]=iimgTNT[bottom,right,:]+iimgTNT[top,left,:]-(iimgTNT[bottom,left,:]+iimgTNT[top,right,:])
    energyVectTNT[0,2:4]=iimgTNT[bottom,cw,:]+iimgTNT[top,left,:]-(iimgTNT[bottom,left,:]+iimgTNT[top,cw,:])      
    energyVectTNT[0,4:6]=iimgTNT[bottom,right,:]+iimgTNT[top,cw,:]-(iimgTNT[bottom,cw,:]+iimgTNT[top,right,:])
    energyVectTNT/=surface
        #energyVect/=energyVect.sum()
            #print 'SHAPES:',energyVect.shape,qHist.shape
            #print (qHist/qHist.sum(axis=0)[None,:]).sum(axis=0)
        #validRangeTNT=np.array(range(1,37)+range(38+1,2*38-1)+range(38*2+1,3*38-1),dtype='int32')
        #enVecTNT=energyVectTNT.reshape([1,-1])[:,validRangeTNT]
    energyVect[0,0] *= energyVectTNT[0,0]
    for envct in range(1,38):
        energyVect[0,envct] *= energyVectTNT[0,1]
            
    energyVect[0,38] *= energyVectTNT[0,2]
    for envct in range(39,38*2):
        energyVect[0,envct] *= energyVectTNT[0,3]
            
    energyVect[0,38*2] *= energyVectTNT[0,4]
    for envct in range(77,38*3):
        energyVect[0,envct] *= energyVectTNT[0,5]
        
    # Computing Histogram Intersection          
    BBEnergy = energyVect 

        
    #L1Normalization        
    BBW = np.sum(BBEnergy)
    BBEnergyNorm = BBEnergy/BBW
  
    qHistW = np.sum(qHist)
    qHistNorm = qHist/qHistW
     
    ### Histogram Intersection            
    qHistNorma = np.zeros((1,len(qHistNorm)))
    for ll in range(0, len(qHistNorm)):
        qHistNorma[0,ll] = qHistNorm[ll]
     
    BBEnergyNorma = np.zeros((1,len(qHistNorm)))
    for ll in range(0, len(qHistNorm)):
        BBEnergyNorma[0,ll] = BBEnergyNorm[0][ll]

    dConcatenate = np.concatenate((BBEnergyNorma,qHistNorma), axis =0)
    HistIntersection = np.sum(np.min(dConcatenate, axis =0))/np.sum(dConcatenate)
    conf = HistIntersection
    
    return conf

    
def getnormalize(confArr):
    normConf = []
    #OldRange = (OldMax - OldMin)  
    #NewRange = (NewMax - NewMin)  
    #NewValue = (((OldValue - OldMin) * NewRange) / OldRange) + NewMin
    ConfRange = (np.max(confArr)-np.min(confArr))
   # NormRange = (1-0)
    for i in range(0,len(confArr)):
        normConf.append(((confArr[i])-(np.min(confArr)))/ConfRange)
    return normConf
    

if __name__=='__main__':
    ### get the directory of the sourceImages, matfiles, CSV, vocabulary
    #print "length of the sysargv, numer of images in this experiment is %d"%len(sys.argv[1:])
    imgNum = 0 
    mainIOU = np.zeros((1,(len(sys.argv[1:]))))  
    GTconfs = []
    TPconfs = []
    for mat_name in sys.argv[1:]:
        #sys.argv -> mat_name = '/home/fcn/wordSpotting/icdarCh4_val_fcnEp63/img_804.mat'        
        print mat_name

        ### Read a mat file to get the integral image and the heat map from softmax
        iimg=getIimg(mat_name,standarise=False)
        # get the scores of text/non text classifier
        tnt_matScores = (mat_name.split('.')[0]+'.tnt.mat')
        iimgTNT = getIimg(tnt_matScores,standarise=False)
        
        
        
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

        textRes = '/home/fcn/wordSpotting/GT_TP/'+fileName+'.txt'

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
                   #qHist = getQueryHist(qword)
                   qHist = getQueryHistHOC2(qword)
                   
                   #get the best TP in compare to GT
                   #gtbb = voc[nqw]
                   tpFile = mat_name.split('.')[0]+'.csv'

                   gtB = np.zeros((1,4))
                   gtB[0][0] = np.min([int(voc[nqw].split(',')[0]),int(voc[nqw].split(',')[6])],axis=0) #left                  
                   gtB[0][1] = np.min([int(voc[nqw].split(',')[1]),int(voc[nqw].split(',')[3])],axis=0) #top
                   gtB[0][2] = np.max([int(voc[nqw].split(',')[2]),int(voc[nqw].split(',')[4])],axis=0) #right
                   gtB[0][3] = np.max([int(voc[nqw].split(',')[5]),int(voc[nqw].split(',')[7])],axis=0) #bottom
                   
                   bpt = getBTP(gtB,tpFile)
                   
                   #### find the TextProposal which is align with the heatMap and Query
                   res=textProposals(iimg,iimgTNT,qHist)                 
                   surf=(res[:,0]-res[:,2])*(res[:,1]-res[:,3])
                   res=res[surf>1200,:]
                   idx=np.argsort(res[:,4])
                   [l,t,r,b]=res[idx[-1],:4]  
                   #######
                   
                   #### Compute the confidence of Gt and TP bounding Box
                   confTP = getConfidence(iimg,iimgTNT,qHist,bpt)
                   TPconfs.append(confTP )
                   
                   confGT = getConfidence(iimg,iimgTNT,qHist,gtB)
                   GTconfs.append(confGT )

                   
                   gtLeft=np.min([int(voc[nqw].split(',')[0]),int(voc[nqw].split(',')[6])],axis=0)
                   gtTop=np.min([int(voc[nqw].split(',')[1]),int(voc[nqw].split(',')[3])],axis=0)
                   gtRight=np.max([int(voc[nqw].split(',')[2]),int(voc[nqw].split(',')[4])],axis=0)
                   gtBottom=np.max([int(voc[nqw].split(',')[5]),int(voc[nqw].split(',')[7])],axis=0)
                   gtWidth=np.absolute(gtRight-gtLeft)
                   gtHeight=np.absolute(gtTop-gtBottom)
                   resLeft=int(bpt[0][0])
                   resTop=int(bpt[0][1])
                   resRight=int(bpt[0][2])
                   resBottom=int(bpt[0][3])
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
                   res_file.write("%d,%d,%d,%d,%d,%d,%d,%d,%s,%f,%f,%f\n" % (resLeft,resTop,resRight,resTop,resRight,resBottom,resLeft,resBottom,qword,IoU,confTP,confGT))                    
                   #plt.plot([l,l,r,r,l],[t,b,b,t,t], color='green', lw=2)
                   #plt.text(l+5,t+5,qword,size=8, color='green' )
                   
                   pts = np.array([[int(voc[nqw].split(',')[0]),int(voc[nqw].split(',')[1])],[int(voc[nqw].split(',')[2]),int(voc[nqw].split(',')[3])],[int(voc[nqw].split(',')[4]),int(voc[nqw].split(',')[5])],[int(voc[nqw].split(',')[6]),int(voc[nqw].split(',')[7])],[int(voc[nqw].split(',')[0]),int(voc[nqw].split(',')[1])]], np.int32)
                   pts = pts.reshape((-1,1,2))
                   cv2.polylines(img,[pts],True,(0,0,255),2)       
                   #cv2.rectangle(img,(gtLeft,gtTop),(gtRight,gtBottom),(0,0,255),2)
                   cv2.putText(img,qword,(gtLeft,gtTop),cv2.FONT_HERSHEY_PLAIN,1,(0,0,255),2)
                   cv2.putText(img,str(np.around(confGT,3)),(gtRight,gtTop),cv2.FONT_HERSHEY_PLAIN,1,(0,0,255),2)
                   cv2.rectangle(img,(resLeft,resTop),(resRight,resBottom),(255,0,0),2)
                   cv2.putText(img,qword,(resLeft,resBottom),cv2.FONT_HERSHEY_PLAIN,1,(255,0,0),2)
                   cv2.putText(img,str(np.around(confTP,3)),(resRight,resBottom),cv2.FONT_HERSHEY_PLAIN,1,(255,0,0),2)
                   cv2.rectangle(img,(int(l),int(t)),(int(r),int(b)),(0,255,0),2)
                   cv2.putText(img,qword,(int(l),int(t)),cv2.FONT_HERSHEY_PLAIN,1,(0,255,0),2)
                   
                   #"""
                   #cv2.imwrite('/home/fcn/wordSpotting/icdarCh4_results_Localization_fcnEp63_TNT/'+fileName+'.png',img)
                   cv2.imwrite('/home/fcn/wordSpotting/GT_TP/'+fileName+'.png',img)
                   #"""
                   #print subIOU
                   
            if (idnqw!=0):       
                mainIOU[0][imgNum] = np.divide(np.sum(subIOU),idnqw)
                print "the IoU of is %f"%mainIOU[0][imgNum]
            else:
                mainIOU[0][imgNum] = 1.0000000
                #"""
                #cv2.imwrite('/home/fcn/wordSpotting/icdarCh4_results_Localization_fcnEp63_TNT/'+fileName+'.png',img)
                cv2.imwrite('/home/fcn/wordSpotting/GT_TP/'+fileName+'.png',img)
                #"""
        imgNum +=1                      
                   
                   
    print "The total IoU is %f"%np.divide(np.sum(mainIOU),(imgNum))          #imgNum-1 or imgNum+1         
    resTot = open("/home/fcn/wordSpotting/GT_TP/totalIoU.txt", "w")
    resTot.write ("sum of IoU: %f \nlength: %f \nmean IoU: %f\n" %(np.sum(mainIOU),imgNum, np.divide(np.sum(mainIOU),(imgNum))))
    resTot.close()    

    ConfNorm = open("/home/fcn/wordSpotting/GT_TP/ConfNormTPGT.txt", "w")    
    TPconfsNorm = getnormalize(TPconfs)
    GTconfsNorm = getnormalize(GTconfs)
    for i in range(0,len(TPconfsNorm)):
        ConfNorm.write("%f,%f,%f,%f\n"%(TPconfs[i],GTconfs[i],TPconfsNorm[i],GTconfsNorm[i]))
        
    ConfNorm.close()
    
    
    
    
    
    
    #find the most similar rectangle of text proposal with the GT
    #1. find the center and width and height then compute the multiplication of them then find the minimum one 
    #GTcenterX = (np.max([int(gtbb.split(',')[2]),int(gtbb.split(',')[4])],axis=0)) - (np.min([int(gtbb.split(',')[0]),int(gtbb.split(',')[6])],axis=0))
    #GTcenterY = ( )-( )
        #diffBB = np.zeros((1,len(BBsrt)))
        #BBcenetrX = 
        #BBcenterY = 
        #diffBB[0][rr] = np.absolute(left-int(gtbb.split(',')[0])) + np.absolute(top-int(gtbb.split(',')[1])) + np.absolute(right-int(gtbb.split(',')[2])) + np.absolute(top-int(gtbb.split(',')[3])) + np.absolute(right-int(gtbb.split(',')[4])) + np.absolute(bottom-int(gtbb.split(',')[5])) + np.absolute(left-int(gtbb.split(',')[6])) + np.absolute(bottom-int(gtbb.split(',')[7]))
        #diffBB[0][rr] = np.absolute(left-int(gtbb.split(',')[0])) * np.absolute(top-int(gtbb.split(',')[1])) * np.absolute(right-int(gtbb.split(',')[2])) * np.absolute(top-int(gtbb.split(',')[3])) * np.absolute(right-int(gtbb.split(',')[4])) * np.absolute(bottom-int(gtbb.split(',')[5])) * np.absolute(left-int(gtbb.split(',')[6])) * np.absolute(bottom-int(gtbb.split(',')[7]))
        #diffBB[0][rr] = (np.absolute(left-int(gtbb.split(',')[0])) + np.absolute(left-int(gtbb.split(',')[6])) ) * ( np.absolute(top-int(gtbb.split(',')[1])) + np.absolute(top-int(gtbb.split(',')[3]))) * (np.absolute(right-int(gtbb.split(',')[2])) + np.absolute(right-int(gtbb.split(',')[4]))) *  (np.absolute(bottom-int(gtbb.split(',')[5])) + np.absolute(bottom-int(gtbb.split(',')[7])) )    
        #nbtp = np.argmin(diffBB)