#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  6 19:29:28 2016

@author: dena
"""

#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 29 17:41:52 2016

@author: dena
"""
# python /home/dena/Projects/CharacterDetection/quantetiveResults/testQuantetive.py /home/fcn/wordSpotting/icdarCh4_val_fcnEp63/img_*.mat
# cd /home/dena/Projects/CharacterDetection/quantetiveResults
# python testQuantetive.py /home/fcn/wordSpotting/icdarCh4_val_fcnEp63/img_*.mat

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
    print 'MAX PIXEL:',matData.sum(axis=2).max()
    print 'MAT SHAPE:',matData.shape
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
    idx=0

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
        BBEnergy[idx,:] = enrgyVect 
        #resLTRB[idx,:]=[left,top,right,bottom,np.sum(enrgyVect[1:]*qHist[1:])]
        idx+=1
        
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
    resLTRBInter[:,0]=BBsrt[:,0]
    resLTRBInter[:,1]=BBsrt[:,1]
    resLTRBInter[:,2]= BBsrt[:,2]+BBsrt[:,0]
    resLTRBInter[:,3]= BBsrt[:,3]+BBsrt[:,1] 
    resLTRBInter[:,4]= HistIntersection[:,0]
    #resLTRB=[BBsrt[:,0], BBsrt[:,1] , BBsrt[:,2]+BBsrt[:,0] , BBsrt[:,3]+BBsrt[:,1] , HistIntersection[:,0]]
    return resLTRBInter
    





if __name__=='__main__':
    ### get the directory of the sourceImages, matfiles, CSV, vocabulary
    for mat_name in sys.argv[1:]:
        #sys.argv -> matFile = '/home/fcn/wordSpotting/icdarCh4_val_fcnEp63/img_804.mat'        
        print mat_name
        ### Read a mat file to get the integral image and the heat map from softmax
        iimg=getIimg(mat_name,standarise=True)
        #get the vocabulary for the correspond image
        voc  = open (mat_name.split('.')[0]+'.voc100.txt').read()
        numQWords =  voc.count('\n')
        voc=re.sub(r'[^\x00-\x7f]',r'',voc)
        voc = voc.split('\r\n')
        csvName = mat_name.split('.')[0]+'.csv'
        #print csvName
        fileName = mat_name.split('.')[0].split('/')[-1]
        textRes = '/home/fcn/wordSpotting/icdarCh4_results_fcnEp63/'+fileName+'.res.txt'
        with open(textRes, "w") as res_file:                                  
            for nqw in range(0,numQWords): 
                qword = voc[nqw]
                print 'query word is %s'%qword
                qHist = getQueryHist(qword)
                #print qHist
                res=textProposals(iimg,qHist)
                #print res
                surf=(res[:,0]-res[:,2])*(res[:,1]-res[:,3])
                res=res[surf>1200,:]
                idx=np.argsort(res[:,4])

                #if (idx[-1] > 7800.0000):
                [l,t,r,b]=res[idx[-1],:4]
                print 'idx[-1]is %f'%idx[-1]
                print 'surface is %f'%((t-b)*(l-r))
                print 'idx[-1] devided by surface is %f'%((idx[-1]/((t-b)*(l-r)))*100)
                #res_file.write("%f,%f,%f,%f,%f,%f,%f,%f,%s\n" % (l,t,r,t,r,b,l,b,qword))


               



















        #matfiles = os.path.join(dname, 'img_*.mat')
        #vocfiles = os.path.join(dname, 'img_*.voc100.txt')
        #sourcefiles = os.path.join(dname, '*.jpg')

        #matNum = len (glob.glob(matfiles))
        #jpgNum = len (glob.glob(sourcefiles))
        #vocNum = len (glob.glob(vocfiles))
        #print 'number of matfiles',matNum
        #for mat_name in glob.glob(matfiles):
            #if mat_name.endswith('.mat'):


















