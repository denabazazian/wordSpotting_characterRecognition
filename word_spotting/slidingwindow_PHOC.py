# -*- coding: utf-8 -*-
"""
Created on Mon Nov  7 17:41:55 2016

@author: dena
"""

#python /home/dena/Projects/CharacterDetection/HeatMaps/slidingwindow_PHOC.py test ./ICDAR-VAL/img_*mat
#python /home/dena/Projects/CharacterDetection/HeatMaps/slidingwindow_PHOC.py test ./ICDAR-VAL/img_80?.mat 
#python /home/fcn/wordSpotting/slidingwindow_PHOC_18_11_16.py test /home/fcn/wordSpotting/ICDAR-VAL/img_80?.mat



import scipy
import scipy.io
import sys
import numpy as np
from matplotlib import pyplot as plt
import cv2
from collections import defaultdict
from commands import getoutput as go

alphabet="#abcdefghijklmnopqrstuvwxyz1234567890@"

def packHistHOC1(wordList,minusOneForMissing=False):
    res=np.zeros([len(alphabet),len(wordList)])
    m=defaultdict(lambda: len(alphabet)-1)
    m.update({alphabet[k]:k for k in range(1,len(alphabet))})
    idx=0
    for w in wordList:
        for c in w.lower():
            res[m[c],idx]+=1
        idx+=1
    if minusOneForMissing:
        res[res==0]=-1
    return res[:,:]

def packHistHOC2(wordList,minusOneForMissing=False):
    res=np.zeros([len(alphabet)*3,len(wordList)])
    m=defaultdict(lambda: len(alphabet)-1)
    m.update({alphabet[k]:k for k in range(1,len(alphabet))})
    idx=0
    for w in wordList:
        for c in w.lower():
            res[m[c],idx]+=1
        for c in w[:len(w)/2].lower():
            res[38+m[c],idx]+=1
        for c in w[len(w)/2:].lower():
            res[2*38+m[c],idx]+=1
        if (len(w))%2:
            res[38+m[w[len(w)/2]],idx]+=.5
            res[2*38+m[w[len(w)/2]],idx]-=.5
        idx+=1
    #pos=-2
    #print wordList[pos]
    #print '\n'.join([str(k)+'\t'+alphabet[k]+'\t'+str(res[k,pos])+'\t'+str(res[k+38,pos])+'\t'+str(res[k+2*38,pos]) for k in range(38)])

    if minusOneForMissing:
        res[res==0]=-1
    return res[:,:]



def packHistHOC3(wordList,minusOneForMissing=False):
    res=np.zeros([len(alphabet)*4,len(wordList)])
    m=defaultdict(lambda: len(alphabet)-1)
    m.update({alphabet[k]:k for k in range(1,len(alphabet))})
    idx=0
    for w in wordList:
        for c in w.lower():
            res[m[c],idx]+=1
        for c in w[:len(w)/3].lower():
            res[38+m[c],idx]+=1
        for c in w[len(w)/3: 2*(len(w)/3)].lower():
            res[2*38+m[c],idx]+=1
        for c in w[2*(len(w)/3):].lower():
            res[3*38+m[c],idx]+=1

        if (len(w))%3:
#            mdf = np.modf((len(w)/3.0))[0]
#            mdfG = mdf; mdfS=1-mdf
#            if np.greater((1-mdf), mdf): mdfG = 1-mdf; mdfS=mdf

#            res[38+m[w[len(w)/3]],idx]+=.3  #mdfG      #.33333333333333333333333
#            res[2*38+m[w[len(w)/3]],idx]-= 1-0.3 #mdfS      #.33333333333333333333333
#            res[2*38+m[w[len(w)/3]],idx]+=0.3 #mdfG      #.33333333333333333333333
#            res[3*38+m[w[len(w)/3]],idx]-=1-0.3  #mdfS      #.33333333333333333333333

            res[38+m[w[len(w)/3]],idx]+= .33333333333333333333333     #mdfG    #.3  #.33333333333333333333333
            res[2*38+m[w[len(w)/3]],idx]-= .33333333333333333333333   #mdfS   # 1-0.3  #.33333333333333333333333
            res[2*38+m[w[2*(len(w)/3)]],idx]+= .33333333333333333333333  #mdfG # 0.3    #.33333333333333333333333
            res[3*38+m[w[2*(len(w)/3)]],idx]-= .33333333333333333333333 #mdfS # 1-0.3    #.33333333333333333333333           
                          
                          
                          
                          
        idx+=1
    #pos=-2
    #print wordList[pos]
    #print '\n'.join([str(k)+'\t'+alphabet[k]+'\t'+str(res[k,pos])+'\t'+str(res[k+38,pos])+'\t'+str(res[k+2*38,pos])+'\t'+str(res[k+3*38,pos]) for k in range(38)])

    if minusOneForMissing:
        res[res==0]=-1
    return res[:,:]





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

def getDistHOC1(iimg,qHist,l,t,r,b,surface):
    energyVect=iimg[b,r,1:-1]+iimg[t,l,1:-1]-(iimg[b,l,1:-1]+iimg[t,r,1:-1])   
    #energyVect/=surface
    energyVect/=energyVect.sum()
    #print 'SHAPES:',energyVect.shape,qHist.shape
    #print (qHist/qHist.sum(axis=0)[None,:]).sum(axis=0)
    return np.dot(energyVect.reshape([1,-1]),(qHist/qHist.sum(axis=0)[None,:])[1:-1,:])
    #return np.dot(energyVect,qHist)
    #energyVect/=np.sum(surface)
    #return np.dot(energyVect,qHist/np.sum(qHist,axis=1))

def getDistHOC2(iimg,qHist,l,t,r,b,surface):
    cw=(l+r)/2
    energyVect=np.empty([1,38*3])
    energyVect[0,:38]=iimg[b,r,:]+iimg[t,l,:]-(iimg[b,l,:]+iimg[t,r,:])
    energyVect[0,38:2*38]=iimg[b,cw,:]+iimg[t,l,:]-(iimg[b,l,:]+iimg[t,cw,:])      
    energyVect[0,38*2:]=iimg[b,r,:]+iimg[t,cw,:]-(iimg[b,cw,:]+iimg[t,r,:])
    #energyVect/=surface
    energyVect/=energyVect.sum()
    #print 'SHAPES:',energyVect.shape,qHist.shape
    #print (qHist/qHist.sum(axis=0)[None,:]).sum(axis=0)
    validRange=np.array(range(1,37)+range(38+1,2*38-1)+range(38*2+1,3*38-1),dtype='int32')
    enVec=energyVect.reshape([1,-1])[:,validRange]
    qH=(qHist/qHist.sum(axis=0)[None,:])[validRange,:]
    #return np.dot(enVec[:,36:2*36],qH[36:2*36,:])
    return np.dot(enVec[:,:],qH[:,:])
    #return np.dot(energyVect,qHist)
    #energyVect/=np.sum(surface)
    #return np.dot(energyVect,qHist/np.sum(qHist,axis=1))


def getDistHOC3(iimg,qHist,l,t,r,b,surface):
    cw=(r-l)/3
    energyVect=np.empty([1,38*4])
    energyVect[0,:38]=iimg[b,r,:]+iimg[t,l,:]-(iimg[b,l,:]+iimg[t,r,:])
    energyVect[0,38:2*38]=iimg[b,l+cw,:]+iimg[t,l,:]-(iimg[b,l,:]+iimg[t,l+cw,:])      
    energyVect[0,38*2:38*3]=iimg[b,l+(2*cw),:]+iimg[t,l+cw,:]-(iimg[b,l+cw,:]+iimg[t,l+(2*cw),:])
    energyVect[0,38*3:]=iimg[b,r,:]+iimg[t,l+(2*cw),:]-(iimg[b,l+(2*cw),:]+iimg[t,r,:])
    #energyVect/=surface
    energyVect/=energyVect.sum()
    #print 'SHAPES:',energyVect.shape,qHist.shape
    #print (qHist/qHist.sum(axis=0)[None,:]).sum(axis=0)
    validRange=np.array(range(1,37)+range(38+1,2*38-1)+range(38*2+1,3*38-1)+range(38*3+1,4*38-1),dtype='int32')
    enVec=energyVect.reshape([1,-1])[:,validRange]
    qH=(qHist/qHist.sum(axis=0)[None,:])[validRange,:]
    #return np.dot(enVec[:,36:2*36],qH[36:2*36,:])
    return np.dot(enVec[:,:],qH[:,:])
    #return np.dot(energyVect,qHist)
    #energyVect/=np.sum(surface)
    #return np.dot(energyVect,qHist/np.sum(qHist,axis=1))    
    
     
    
getDist=getDistHOC3
packHist=packHistHOC3


def slidingWindow(iimg,qHist,dx=8,dy=8,heights=[5,8,11,14,17,22,28,32,48],longalities=[1,2,3,4,6,8,11]):
    if len(qHist.shape)==1:
        qHist=qHist.reshape([len(alphabet),-1])
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
                    resLTRB[idx,:]=[left,top,right,bottom,getDist(iimg,qHist,left,top,right,bottom,surface)]
                    idx+=1
    return resLTRB


def loadGtFile(fname):
    fileStr=open(fname).read()
    if not any([ord(fileStr[k])<128 for k in range(3)]):
        #if three first bytes are above the standard ascii values
        fileStr=fileStr[3:]
    lines=[l.split(',') for l in open(fname).read()[3:].split('\n') if len(l)>0 and l.split(',')[-1].strip()!='###']
    LTRB=np.empty([len(lines),4])
    captions=[]
    if any([len(line)<9 for line in lines]):#if 2 point rectangles
        for k in range(len(lines)):
            captions.append(','.join(lines[k][4:]).strip())
            LTRB[k,:]=[int(col) for col in lines[k][:4]]
    else:#4 point polygons
        for k in range(len(lines)):
            captions.append(','.join(lines[k][8:]).strip())
            x=np.array([int(lines[k][p]) for p in [0,2,4,6]])
            y=np.array([int(lines[k][p]) for p in [1,3,5,7]])
            LTRB[k,:]=(x.min(),y.min(),x.max(),y.max())
    return (LTRB,np.array(captions))


def plotWords(imgFname,corrMat,LTRB,lexicon):
    img=cv2.imread(imgFname)
    plt.figure()
    plt.imshow(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
    colors="bgrcmykw"*10
    for k in range(corrMat.shape[0]):
        x=[LTRB[k,0],LTRB[k,2],LTRB[k,2],LTRB[k,0],LTRB[k,0]]
        y=[LTRB[k,1],LTRB[k,1],LTRB[k,3],LTRB[k,3],LTRB[k,1]]
        plt.plot(x,y,color=colors[k])
        plt.text(LTRB[k,0],LTRB[k,1],lexicon[np.argmax(corrMat[k,:])],color=colors[k])
    plt.ylim((img.shape[0],0))
    plt.xlim((0,img.shape[1]))

def plotMaps(iimg):
    plt.figure()
    for k in range(iimg.shape[2]):
        plt.subplot(6, 7, k+1)
        img=iimg[1:,:,k]-iimg[:-1,:,k]
        img=img[:,1:]-img[:,:-1]
        plt.imshow(img)
        plt.text(img.shape[0]/2,img.shape[1]/2,alphabet[k],size=32)


def getTestMatrix(matFname,voc=None,draw=False):
    gtFname='.'.join(matFname.split('.')[:-1])+'.txt'
    imgFname='.'.join(matFname.split('.')[:-1])+'.jpg'
    iimg=getIimg(matFname,standarise=True)
    gtLTRB,gtCaptions=loadGtFile(gtFname)
    if voc==None:
        voc=sorted((list(set(gtCaptions.tolist()))+open('/home/fcn/wordSpotting/voc_icdar.txt').read().split('\n'))[:50])
    qHist=packHist(voc)
    surface=(1+ gtLTRB[:,2] -gtLTRB[:,0])*(1+gtLTRB[:,3]-gtLTRB[:,1])
    res=np.zeros([gtCaptions.shape[0],len(voc)])
    for k  in range(gtCaptions.shape[0]):
        d=getDist(iimg,qHist,gtLTRB[k,0],gtLTRB[k,1],gtLTRB[k,2],gtLTRB[k,3],surface[k])
        res[k,:]=d
    if draw:
        plotMaps(iimg)
        plotWords(imgFname,res,gtLTRB,voc)
        plt.show()
    return {'dm':res,'voc':np.array(voc),'gt':gtCaptions}


def drawRectangles(outSize,rectList,useWeightColumn=False):
    """Draws horizontal edges of rectangles and than integrates
    """
    unintegratedImg=np.zeros([outSize[0]+1,outSize[1]+1])
    if useWeightColumn:
        for rect in rectList:
            [l,t,r,b,v]=rect[:5]
            unintegratedImg[t+1,l+1]+=v
            unintegratedImg[b+1,l+1]-=v
            unintegratedImg[t+1,r+1]-=v
            unintegratedImg[t+1,r+1]+=v
    else:
        for rect in rectList:
            [l,t,r,b,v]=rect[:5]
            unintegratedImg[t+1,l+1]+=1
            unintegratedImg[b+1,l+1]-=1
            unintegratedImg[t+1,r+1]-=1
            unintegratedImg[t+1,r+1]+=1
    iimg=unintegratedImg.cumsum(axis=0).cumsum(axis=1)
    return iimg[1:,1:]-iimg[1:,:-1]


if __name__=='__main__':
    if sys.argv[1]=='test':
        res=[]
        for matFname in sys.argv[2:]:
            #dMat = getTestMatrix(matFname, ['RASPRODA', 'REBAJAS', 'SAL','DISTRACTOR'],True)
            dMat = getTestMatrix(matFname,None,False)
            print dMat['voc'][np.argmax(dMat['dm'],axis=1)]
            print dMat['gt']
            resp=dMat['voc'][np.argmax(dMat['dm'],axis=1)]
            gt=dMat['gt']
            res.append(np.array([gt[k]==resp[k] for k in range(len(gt))]))
            print res[-1]
            #plt.show()
        meanRes=[r.mean() for r in res if len(r)>0]
        print meanRes
        print 'TOTAL:',100*(sum(meanRes)/len(meanRes))
    elif sys.argv[1]=='proposals':
        for propfname in sys.argv[2:]:
            imgFname='.'.join(propfname.split('.')[:-1]+['jpg'])
            img=cv2.cvtColor(cv2.imread(imgFname),cv2.COLOR_BGR2RGB)
            proposals=np.array([[float(c) for c in line.split(',')[:5]] for line in open(propfname,'r').read().split('\n') if len(line)>0])
            rects=proposals.astype('int32')
            propHm=drawRectangles(img.shape,)
    else:
        sys.exit(0)
