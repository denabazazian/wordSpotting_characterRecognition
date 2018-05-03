# -*- coding: utf-8 -*-
"""
Created on Thu Nov 23 15:49:29 2017

@author: dena
"""

# cd /home/dena/Projects/caffe_CharacterDetection/
# python wordSpottingPHOCEntropy.py /home/fcn/wordSpotting/icdar_ch4_testSet/img_*.jpg
# python wordSpottingPHOCEntropy.py /home/fcn/wordSpotting/icdarCh4_val_fcnEp113/img_*.jpg

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
from pylab import *

sys.path.append('/home/dena/Software/caffe/python')
from PIL import Image
import matplotlib.pyplot as plt
import caffe



def getIimg(im,net):
    
    im = Image.open(im)
    if (im.size.__len__() == 2):
        im_gray = im
        im = Image.new("RGB", im_gray.size)
        im.paste(im_gray)               
            #switch to BGR and substract mean
    in_ = np.array(im, dtype=np.float32)
    in_ = in_[:,:,::-1]
    in_ -= np.array((104.00698793,116.66876762,122.67891434))
    in_ = in_.transpose((2,0,1))
         #shape for input (data blob is N x C x H x W)
    net.blobs['data'].reshape(1, *in_.shape)
    net.blobs['data'].data[...] = in_
       #run net and take scores
    net.forward()    
        # Heatmap computation
    scores_exp = np.exp(net.blobs['score_conv'].data[0][:, :, :])
    sum_exp = np.sum (scores_exp, axis=0)
    heatMap = np.empty((im.size[1], im.size[0], scores_exp.shape[0]))
    for ii in range(0,scores_exp.shape[0]):    
        heatMap[:,:,ii] = scores_exp[ii,:,:]/sum_exp
    
    ##entropy = -1.0*(np.sum(heatMap * np.log(heatMap), axis=2)) 
    ##mx = entropy.max()
    ##mn = entropy.min()
    ##entropy = (entropy - mn) / (mx - mn)
    ##ientropy = entropy.cumsum(axis=0).cumsum(axis=1)
    
    iimg=heatMap.cumsum(axis=0).cumsum(axis=1)
    
    return iimg
    
    
################# HOC Level1 ##########################################
#def getQueryHist(qword,minusOneForMissing=False):
#    qHist=np.zeros([len(alphabet)])
#    m=defaultdict(lambda: len(alphabet)-1)
#    m.update({alphabet[k]:k for k in range(1,len(alphabet))})
#    for c in qword.lower():
#        qHist[m[c]]+=1
#    if minusOneForMissing:
#        qHist[qHist==0]=-1
#    return qHist[:]
######################## HOC Level2 #########################################
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

    if minusOneForMissing:
       qHist[res==0]=-1
    return qHist[:]
    
    

#def textProposals(iimg,iimgTNT,ientropy,ientropyTNT,qHist):
def textProposals(csvName,iimg,iimgTNT,qHist):
    idxTP=0

    #lines=[l.strip().split(',') for l in open((fname+'.csv')).read().split('\n') if len(l)>0]
    lines=[l.strip().split(',') for l in open(csvName).read().split('\n') if len(l)>0]       
    #print "proposals has been read"    
    
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
        left = int(BBsrt[rr,0])
        top = int(BBsrt[rr,1])
        right = int(left + BBsrt[rr,2]) #right = left + width
        bottom = int(top + BBsrt[rr,3]) #bottom = top + height
        surface =  BBsrt[rr,2] * BBsrt[rr,3]       
        #print 'left: %s, top:%s, right:%s, bottom:%s' %(left,top,right,bottom)
        if right>= iimg.shape[1]:
            right = int(iimg.shape[1]-1)
        if top>= iimg.shape[0]:
            top = int(iimg.shape[0]-1)    
            
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
        
        #surfaceEntropy = ientropy[bottom,right]+ientropy[top,left]-(ientropy[bottom,left]+ientropy[top,right])
        energyVect/=surface
        #energyVect/=surfaceEntropy
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
        
        #surfaceEntropyTNT = ientropyTNT[bottom,right]+ientropyTNT[top,left]-(ientropyTNT[bottom,left]+ientropyTNT[top,right])
        energyVectTNT/=surface
        #energyVectTNT/=surfaceEntropyTNT
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
        
    #print "shape of BBenergyNorm is:"
    #print BBEnergyNorm.shape    
  
    qHistW = np.sum(qHist)
    qHistNorm = qHist/qHistW
    
    #print "shape of qHistNorm is:"
    #print qHistNorm.shape 
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
    

if __name__=='__main__':

    caffe.set_device(0)
    caffe.set_mode_gpu()

    net_char = caffe.Net('/home/dena/Projects/caffe_CharacterDetection/synth-fcn8s-atonce/deploy2.prototxt', '/home/dena/Projects/caffe_CharacterDetection/version2/snapshot/train_iter_5101000.caffemodel', caffe.TEST)
    net_text = caffe.Net('/home/dena/Projects/Caffe-FCN-textNontext/deploy.prototxt', '/home/dena/Projects/Caffe-FCN-textNontext/fcn.berkeleyvision/snapshot-ICDAR/train_iter_10500.caffemodel', caffe.TEST)

    print "nets are read"
    imgNum = 0 
    mainIOU = np.zeros((1,(len(sys.argv[1:])))) 
    #mainIOU = np.zeros((1,1)) 
    alphabet="#abcdefghijklmnopqrstuvwxyz1234567890@"
    
    for img_name in sys.argv[1:]:
    #for debugingMode in range(0,1):
        #sys.argv -> img_name = '/home/fcn/wordSpotting/icdar_ch4_testSet/img_873.jpg'        
        #img_name = '/home/fcn/wordSpotting/icdar_ch4_testSet/img_873.jpg'        
        print img_name

        ### Read a mat file to get the integral image and the heat map from softmax
        #iimg, ientropy=getIimg(img_name,net_char)
        iimg = getIimg(img_name,net_char)
        # get the scores of text/non text classifier
        #iimgTNT, ientropyTNT = getIimg(img_name,net_text)
        iimgTNT = getIimg(img_name,net_text)
        
        
        #get the vocabulary for the correspond image
        #voc  = open (mat_name.split('.')[0]+'.voc100.txt').read()
        #Read the vocabulary from the gt
        #voc  = open (img_name.split('.')[0]+'.txt').read()  # for the ch4 testing
        voc  = open (img_name.split('.')[0]+'.gt.txt').read()  #for the ch4 validation
        numQWords =  voc.count('\n')
        voc=re.sub(r'[^\x00-\x7f]',r'',voc)
        voc = voc.split('\r\n')
        csvName = img_name.split('.')[0]+'.csv'
        #print csvName
        fileName = img_name.split('.')[0].split('/')[-1]

        #end2endRes = open ('/home/fcn/wordSpotting/icdar_test_ch4_results/end_to_end_results/res_'+fileName+'.txt', 'w')
        #locRes = open ('/home/fcn/wordSpotting/icdar_test_ch4_results/localization_results/res_'+fileName+'.txt', 'w')
        
        subIOU = np.zeros((1,numQWords))
        #with open(textIOU, "w") as iou_file:
        #with open(textRes, "w") as res_file:  
        img=cv2.imread(img_name.split('.')[0]+'.jpg')    
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
               
               #print qHist
               #res=textProposals(iimg,iimgTNT,ientropy,ientropyTNT,qHist)
               res=textProposals(csvName,iimg,iimgTNT,qHist)

               #print res
               surf=(res[:,0]-res[:,2])*(res[:,1]-res[:,3])
               res=res[surf>1200,:]
               idx=np.argsort(res[:,4])
               
               #print 'idx[-1] devided by surface is %f'%np.multiply((np.divide(idx[-1],surf)),100)
               #if (idx[-1] > 7800.0000):
               [l,t,r,b]=res[idx[-1],:4] 
               #print "Histogram Intersection is : %f"%(res[idx[-1],4])
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
               print "the subIoU is %f"%IoU
               subIOU[0][nqw] = IoU
                           #iou_file.write("%f\n"%IoU)                   
                           #res_file.write("%d,%d,%d,%d,%d,%d,%d,%d,%s,%f\n" % (l,t,r,t,r,b,l,b,qword,IoU))
               #end2endRes.write("%d,%d,%d,%d,%d,%d,%d,%d,%s\r\n" % (l,t,r,t,r,b,l,b,qword))
               #locRes.write("%d,%d,%d,%d,%d,%d,%d,%d\r\n" % (l,t,r,t,r,b,l,b))
                
                                #plt.plot([l,l,r,r,l],[t,b,b,t,t], color='green', lw=2)
                                #plt.text(l+5,t+5,qword,size=8, color='green' )
               
               pts = np.array([[int(voc[nqw].split(',')[0]),int(voc[nqw].split(',')[1])],[int(voc[nqw].split(',')[2]),int(voc[nqw].split(',')[3])],[int(voc[nqw].split(',')[4]),int(voc[nqw].split(',')[5])],[int(voc[nqw].split(',')[6]),int(voc[nqw].split(',')[7])],[int(voc[nqw].split(',')[0]),int(voc[nqw].split(',')[1])]], np.int32)
               pts = pts.reshape((-1,1,2))
               cv2.polylines(img,[pts],True,(0,0,255),2)       
                     #cv2.rectangle(img,(gtLeft,gtTop),(gtRight,gtBottom),(0,0,255),2)
               cv2.putText(img,qword,(gtLeft,gtTop),cv2.FONT_HERSHEY_PLAIN,1,(0,0,255),2)
               cv2.rectangle(img,(resLeft,resTop),(resRight,resBottom),(0,255,0),2)
               cv2.putText(img,qword,(resLeft,resTop),cv2.FONT_HERSHEY_PLAIN,1,(0,255,0),2)
               
               #"""
                       #cv2.imwrite('/home/fcn/wordSpotting/icdarCh4_results_Localization_fcnEp63_TNT/'+fileName+'.png',img)
               #cv2.imwrite('/home/fcn/wordSpotting/icdar_test_ch4_results/visual_results/'+fileName+'.png',img)
               #"""
               #print subIOU
                           
        if (idnqw!=0):       
            mainIOU[0][imgNum] = np.divide(np.sum(subIOU),idnqw)
            print "the IoU of is %f"%mainIOU[0][imgNum]
        else:
            mainIOU[0][imgNum] = 1.0000000
                #"""
                         #cv2.imwrite('/home/fcn/wordSpotting/icdarCh4_results_Localization_fcnEp63_TNT/'+fileName+'.png',img)
                #cv2.imwrite('/home/fcn/wordSpotting/Entropy_PHOC_TNT_results/'+fileName+'.png',img)
                #"""
        imgNum +=1                      
                   
                   
    print "The total IoU is %f"%np.divide(np.sum(mainIOU),(imgNum))          #imgNum-1 or imgNum+1         
    #resTot = open("/home/fcn/wordSpotting/icdar_test_ch4_results/visual_results/totalIoU.txt", "w")
    #resTot.write ("sum of IoU: %f \nlength: %f \nmean IoU: %f\n" %(np.sum(mainIOU),imgNum, np.divide(np.sum(mainIOU),(imgNum))))
    #resTot.close()  


######################################################################

#caffe.set_device(0)
#caffe.set_mode_gpu()
#
#net_char = caffe.Net('/home/dena/Projects/caffe_CharacterDetection/synth-fcn8s-atonce/deploy2.prototxt', '/home/dena/Projects/caffe_CharacterDetection/version2/snapshot/train_iter_5101000.caffemodel', caffe.TEST)
#net_text = caffe.Net('/home/dena/Projects/Caffe-FCN-textNontext/deploy.prototxt', '/home/dena/Projects/Caffe-FCN-textNontext/fcn.berkeleyvision/snapshot-ICDAR/train_iter_10500.caffemodel', caffe.TEST)
#
#print "nets are read"

    # Turn grayscale images to 3 channels
#    if (im.size.__len__() == 2):
#        im_gray = im
#        im = Image.new("RGB", im_gray.size)
#        im.paste(im_gray)
#        
#        
#            #switch to BGR and substract mean
#    in_ = np.array(im, dtype=np.float32)
#    in_ = in_[:,:,::-1]
#    in_ -= np.array((104.00698793,116.66876762,122.67891434))
#    in_ = in_.transpose((2,0,1))
#
#    # shape for input (data blob is N x C x H x W)
#    net.blobs['data'].reshape(1, *in_.shape)
#    net.blobs['data'].data[...] = in_
#
#    # run net and take scores
#    net.forward()
#    
#        # Heatmap computation
#    scores_exp = np.exp(net.blobs['score_conv'].data[0][:, :, :])
#    sum_exp = np.sum (scores_exp, axis=0)
#    heatMap = np.empty((im.size[1], im.size[0], 38))
#    for ii in range(0,38):    
#        heatMap[:,:,ii] = scores_exp[ii,:,:]/sum_exp
        
 


#    matData=scipy.io.loadmat(matFname)['scores']
#    expData=np.exp(matData)
#    expData/=expData.sum(axis=2)[:,:,None]
#    ##entropy = -1.0*(np.sum(expData * np.log(expData), axis=2))
#    ##mx = entropy.max()
#    ##mn = entropy.min()
#    ##entropy = (entropy - mn) / (mx - mn)
#    ##ientropy = entropy.cumsum(axis=0).cumsum(axis=1)
#    
#    matData=expData.copy()
#    if standarise:
#       for layerNum in range(matData.shape[2]):
#           matData[:,:,layerNum] *=entropy
#           matData[:,:,layerNum]-=matData[:,:,layerNum].mean()
#           matData[:,:,layerNum]/=matData[:,:,layerNum].std()
#       iimg=matData.cumsum(axis=0).cumsum(axis=1)
#    else:
#       for layerNum in range(matData.shape[2]):
#           matData[:,:,layerNum] *= (1.0 - entropy)           
#       iimg=matData.cumsum(axis=0).cumsum(axis=1)   
#    return iimg, ientropy       
        

# read the caffe model for character and for text
#caffe model of character dtectioin (38 classes)
#caffe model of text/ non text detction for 2 classes
# compute the softmax heatmap
#write the results a txt file for localization
# a txt file for end-to-end results