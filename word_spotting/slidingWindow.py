# -*- coding: utf-8 -*-
"""
Created on Mon Nov  7 17:41:55 2016

@author: dena
"""
import scipy
import scipy.io
import sys
import numpy as np
from matplotlib import pyplot as plt
import cv2

def getIimg(matFname):
    matData=scipy.io.loadmat(matFname)
    iimg=matData['heatMap'].cumsum(axis=0).cumsum(axis=1)
    return iimg

def slidingWindow(iimg,qHist,dx=8,dy=8,heights=[5,8,11,14,17,22,28,32,48],longalities=[1,2,3,4,6,8,11]):
    resLTRB=np.empty([(iimg.shape[1]/dx)*(iimg.shape[0]/dy)*len(longalities)*len(heights),5])
    idx=0
    for letHeight in heights:
        print letHeight
        for longality in longalities:
            width=letHeight*longality
            surface=width*letHeight
            for left in range(1,(iimg.shape[1]-width)-1,dx):
                for top in range(1,(iimg.shape[0]-letHeight)-1,dy):
                    right=left+width
                    bottom=top+letHeight
                    enrgyVect=iimg[bottom,right,:]+iimg[top,left,:]-(iimg[bottom,left,:]+iimg[top,right,:])
                    enrgyVect/=surface
                    resLTRB[idx,:]=[left,top,right,bottom,np.sum(enrgyVect[1:]*qHist[1:])]
                    idx+=1
    return resLTRB




if __name__=='__main__':
    for fname in sys.argv[1:]:
        fname='.'.join(fname.split('.')[:-1])
        img=cv2.imread(fname+'.jpg')
        qHist=np.zeros(38)
        qHist[[3,1]]=1
        qHist[12]=2
        iimg=getIimg(fname+'.mat')
        res=slidingWindow(iimg,qHist)
        surf=(res[:,0]-res[:,2])*(res[:,1]-res[:,3])
        res=res[surf>1200,:]
        idx=np.argsort(res[:,4])
        plt.imshow(img)
        for k in idx[-10:]:
            print res[k,:]
            [l,t,r,b]=res[k,:4]
            plt.plot([l,l,r,r,l],[t,b,b,t,t])
        plt.show()