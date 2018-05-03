#!/usr/bin/env python
# -*- coding: utf-8 -*-

#firefox  /tmp/report/index.html 
### #"!"/usr/bin/env python
#import sys
#
#print '{'+'},{'.join(sys.argv)+'}'
#for fname in sorted(sys.argv):
#	print fname,':',len(open(fname).read()),' bytes'
#python performanceEvaluation.py groundtruthDirectory ResultDirectory


import numpy.matlib
import numpy as np
import re
import sys
import cv2
import time
from commands import getoutput as go
from matplotlib import pyplot as plt

def convLTRB24point(ltbr):
    L=ltbr[:,0]
    T=ltbr[:,1]
    B=ltbr[:,2]
    R=ltbr[:,3]
    res = np.concatenate([L.reshape([-1,1]),T.reshape([-1,1]),R.reshape([-1,1]),T.reshape([-1,1]),R.reshape([-1,1]),B.reshape([-1,1]),L.reshape([-1,1]),B.reshape([-1,1])],axis=1)
    if ltbr.shape[1]>4:
        res=np.concatenate([res,ltbr[:,4:]])
    return res


def conv4pointToLTBR(pointMat):
    L=pointMat[:,[0,2,4,6]].min(axis=1)
    R=pointMat[:,[0,2,4,6]].max(axis=1)
    T=pointMat[:,[1,3,5,7]].min(axis=1)
    B=pointMat[:,[1,3,5,7]].max(axis=1)
    res = np.concatenate([L.reshape([-1,1]),T.reshape([-1,1]),R.reshape([-1,1]),B.reshape([-1,1])],axis=1)
    if pointMat.shape[1]>8:
        res=np.concatenate([res,pointMat[:,8:]])
    return res


def convLTWH2LTBR(ltwh):
    L=ltwh[:,0]
    R=ltwh[:,2]+1+L
    T=ltwh[:,1]
    B=ltwh[:,3]+1+T
    res = np.concatenate([L.reshape([-1,1]),T.reshape([-1,1]),R.reshape([-1,1]),B.reshape([-1,1])],axis=1)
    if ltwh.shape[1]>4:
        res=np.concatenate([res,ltwh[:,8:]])
    return res


def convLTRB2LTWH(ltrb):
    L=ltrb[:,0]
    T=ltrb[:,1]
    W=1+(ltrb[:,2]-ltrb[:,0])
    H=1+(ltrb[:,3]-ltrb[:,1])
    res = np.concatenate([L.reshape([-1,1]),T.reshape([-1,1]),W.reshape([-1,1]),H.reshape([-1,1])],axis=1)
    if ltrb.shape[1]>4:
        res=np.concatenate([res,ltrb[:,4:]])
    return res


def get2PointIoU(gtMat,resMat):
    gtMat=convLTRB2LTWH(gtMat)
    resMat=convLTRB2LTWH(resMat)
#    maxProposalsIoU=int(switches['maxProposalsIoU'])
#    if maxProposalsIoU>0:
#        resMat=resMat[:maxProposalsIoU,:]
    #matSz=(gtMat.shape[0],resMat.shape[0])
    gtLeft=numpy.matlib.repmat(gtMat[:,0],resMat.shape[0],1)
    gtTop=numpy.matlib.repmat(gtMat[:,1],resMat.shape[0],1)
    gtRight=numpy.matlib.repmat(gtMat[:,0]+gtMat[:,2]-1,resMat.shape[0],1)
    gtBottom=numpy.matlib.repmat(gtMat[:,1]+gtMat[:,3]-1,resMat.shape[0],1)
    gtWidth=numpy.matlib.repmat(gtMat[:,2],resMat.shape[0],1)
    gtHeight=numpy.matlib.repmat(gtMat[:,3],resMat.shape[0],1)
    resLeft=numpy.matlib.repmat(resMat[:,0],gtMat.shape[0],1).T
    resTop=numpy.matlib.repmat(resMat[:,1],gtMat.shape[0],1).T
    resRight=numpy.matlib.repmat(resMat[:,0]+resMat[:,2]-1,gtMat.shape[0],1).T
    resBottom=numpy.matlib.repmat(resMat[:,1]+resMat[:,3]-1,gtMat.shape[0],1).T
    resWidth=numpy.matlib.repmat(resMat[:,2],gtMat.shape[0],1).T
    resHeight=numpy.matlib.repmat(resMat[:,3],gtMat.shape[0],1).T
    intL=np.max([resLeft,gtLeft],axis=0)
    intT=np.max([resTop,gtTop],axis=0)
    intR=np.min([resRight,gtRight],axis=0)
    intB=np.min([resBottom,gtBottom],axis=0)
    intW=(intR-intL)+1
    intW[intW<0]=0
    intH=(intB-intT)+1
    intH[intH<0]=0
    I=intH*intW
    U=resWidth*resHeight+gtWidth*gtHeight-I
    IoU=I/(U+.0000000001)
    return (IoU,I,U)


def loadBBoxTranscription(fname,**kwargs):
    txt=open(fname).read()
    txt=re.sub(r'[^\x00-\x7f]',r'',txt)#removing the magical bytes crap
    lines=[l.strip().split(',') for l in txt.split('\n') if (len(l.strip())>0)]
    colFound=min([len(l) for l in lines])-1
    if colFound==4:
        resBoxes=np.empty([len(lines),4],dtype='int32')
        resTranscriptions=np.empty(len(lines), dtype=object)
        for k in range(len(lines)):
            resBoxes[k,:]=[int(c) for c in lines[k][:4]]
            resTranscriptions[k]=','.join(lines[k][4:])
    elif colFound==8:
        resBoxes=np.empty([len(lines),8],dtype='int32')
        resTranscriptions=np.empty(len(lines), dtype=object)
        for k in range(len(lines)):
            resBoxes[k,:]=[int(float(c)) for c in lines[k][:8]]
            resTranscriptions[k]=','.join(lines[k][8:])
    else:
        sys.stderr.write('Cols found '+str(colFound)+'\n')
        sys.stderr.flush()
        raise Exception('Wrong columns found')
    return (resBoxes,resTranscriptions)


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


def getReport(imgGtSubmFnames,**kwargs):
    #print imgGtSubmFnames
    """imgGtSubmFnames is a list of tuples with three strings:
       The first one is the path to the input image
       The second is the path to the 4point+transcription Gt
       The third is the path to the 4point+transcription Solution
    """
    p={'IoUthreshold':.5,'dontCare':'###','visualise':True,'outReportDir':'/tmp/report/'}
    p.update(kwargs)
    allIoU=[]
    allEqual=[]
    allCare=[]
    accDict={}
    go('mkdir -p '+p['outReportDir'])
    startTime=time.time()
    for inImgFname,gtFname,submFname in imgGtSubmFnames:
        sampleFname=p['outReportDir']+'/'+inImgFname.split('/')[-1].split('.')[0]
        rectSubm,transcrSubm=loadBBoxTranscription(submFname)
        rectGt,transcrGt=loadBBoxTranscription(gtFname)
        rectGt=conv4pointToLTBR(rectGt)
        rectSubm=conv4pointToLTBR(rectSubm)
        print rectSubm
        #rectSubm,transcrSubm,rectGt,transcrGt
        IoU=get2PointIoU(rectGt,rectSubm)[0]
        exactIoU=IoU.copy()
        strEqual=np.zeros([len(transcrSubm),len(transcrGt)])
        strCare=np.zeros([len(transcrSubm),len(transcrGt)])
        for gt in range(transcrGt.shape[0]):
            strCare[:,gt]=(transcrGt[gt]!=p['dontCare'])
            for subm in range(transcrSubm.shape[0]):
                strEqual[subm,gt]=(transcrGt[gt]==transcrSubm[subm])
        #IoU[IoU!=IoU.max(axis=0)[None,:]]=0
        IoU=(IoU>p['IoUthreshold']).astype('float')
        allIoU.append(IoU)
        allEqual.append(strEqual)
        allCare.append(strCare)
        img=cv2.imread(inImgFname)
        if p['visualise']:
            plt.imshow(img)
            plt.show()
            plotRectangles(rectGt,transcrGt,img,[0,255,0])
            plotRectangles(rectSubm,transcrSubm,img,[255,0,0])
            cv2.imwrite(sampleFname+'.png',img)
        else:
            cv2.imwrite(sampleFname+'.png',img)
        resTbl='<table border=1>\n<tr><td></td><td>'
        resTbl+='</td> <td>'.join([s for s in transcrGt])+'</tb></tr>\n'
        for k in range(IoU.shape[0]):
            resTbl+='<tr><td>'+transcrSubm[k]+'</td><td>'
            resTbl+='</td><td>'.join([str(int(k*10000)/100.0) for k in IoU[k,:]*strEqual[k,:]])+'</td></tr>\n'
        resTbl+='</table>\n'
        resHtml='<html><body>\n<h3>'+inImgFname.split('/')[-1].split('.')[0]+'</h3>\n'
        
        acc=((IoU*strEqual).sum()/float(IoU.shape[1]))
        if p['dontCare']!='':
            precision=(IoU*strEqual).max(axis=1)[(strCare*IoU).sum(axis=1)>0].mean()
            if np.isnan(precision):
                precision=0
            recall=(IoU*strEqual).max(axis=0)[(strCare).sum(axis=0)>0].mean()
            if np.isnan(recall):
                recall=0
        else:
            precision=(IoU*strEqual).max(axis=1).mean()
            recall=(IoU*strEqual).max(axis=0).mean()
        fm=(2.0*precision*recall)/(.0000001+precision+recall)
        accDict[sampleFname+'.html']=[acc,precision,recall,fm]
        resHtml+='<hr>\n<table><tr>'
        resHtml+='<td>Accuracy : '+str(int(acc*10000)/100.0)+'% </td>'
        resHtml+='<td>Precision : '+str(int(precision*10000)/100.0)+'% </td>'
        resHtml+='<td>Recall : '+str(int(recall*10000)/100.0)+'% </td>'
        resHtml+='<td> FM : '+str(int(fm*10000)/100.0)+'% </td>'
        resHtml+='</tr></table>\n<hr>\n'
        resHtml+='<img src="'+sampleFname+'.png"/>\n<hr>\n'+resTbl
        resHtml+='</body></html>'
        open(sampleFname+'.html','w').write(resHtml)
    gtSize=sum([iou.shape[1] for iou in allIoU])
    submSize=sum([iou.shape[0] for iou in allIoU])
    IoU=np.zeros([submSize,gtSize])
    strEqual=np.zeros([submSize,gtSize])
    strCare=np.zeros([submSize,gtSize])
    gtIdx=0
    submIdx=0
    for k in range(len(allIoU)):
        submSize,gtSize=allIoU[k].shape
        IoU[submIdx:submIdx+submSize,gtIdx:gtIdx+gtSize]=allIoU[k]
        strEqual[submIdx:submIdx+submSize,gtIdx:gtIdx+gtSize]=allEqual[k]
        strCare[submIdx:submIdx+submSize,gtIdx:gtIdx+gtSize]=allCare[k]
        gtIdx+=gtSize
        submIdx+=submSize
    acc=((IoU*strEqual).sum()/float(IoU.shape[1]))
    if p['dontCare']!='':
        print 'CARE:',(IoU*strCare).sum(axis=1)>0
        precision=(IoU*strEqual).max(axis=1)[(IoU*strCare).sum(axis=1)>0].mean()
        recall=(IoU*strEqual).max(axis=0)[(strCare).sum(axis=0)>0].mean()
    else:
        precision=(IoU*strEqual).max(axis=1).mean()
        recall=(IoU*strEqual).max(axis=0).mean()

#    precision=(IoU*strEqual).max(axis=1).mean()
#    recall=(IoU*strEqual).max(axis=0).mean()
    fm=(2.0*precision*recall)/(.0000001+precision+recall)
    resHtml='<body><html>\n<h3>Report on end 2 end</h3>\n'
    resHtml+='<hr>\n<table border=1>'
    resHtml+='<tr><td> Total Samples: </td><td>'+str(IoU.shape[1])+'</td></tr>'
    resHtml+='<tr><td> Detected Samples : </td><td>'+str(IoU.shape[0])+' </td></tr>'
    resHtml+='<tr><td>Correct Samples : </td><td>'+str(int((IoU*strEqual).sum()))+' </td></tr>'
    resHtml+='<tr><td>Computation Time : </td><td>'+str(int(1000*(time.time()-startTime))/1000.0)+' sec. </td></tr>'
    resHtml+='<tr><td></td><td></td></tr>\n'
    resHtml+='<tr><td>Accuracy : </td><td>'+str(int(acc*10000)/100.0)+'\% </td></tr>'
    resHtml+='<tr><td>Precision :</td><td> '+str(int(precision*10000)/100.0)+'\% </td></tr>'
    resHtml+='<tr><td>Recall : </td><td>'+str(int(recall*10000)/100.0)+'\% </td></tr>'
    resHtml+='<tr><td> FM : </td><td>'+str(int(fm*10000)/100.0)+'\% </td></tr>'
    resHtml+='</table>\n<hr>\n'
    resHtml+='<table><tr><td>sample</td><td>Acc</td><td>Precision</td><td>Recall</td><td>FMeasure</td><tr>\n'
    for sampleFname in accDict.keys():
        fname=sampleFname.split('/')[-1]
        acc,pr,rec,fm=accDict[sampleFname]
        resHtml+='<tr><td><a href="'+fname+'">'+fname.split('.')[0]+'</a></td><td>'+str(int(10000*(acc))/100.0)+'%</td><td>'
        resHtml+=str(int(10000*(pr))/100.0)+'%</td><td>'+str(int(10000*(rec))/100.0)+'%</td><td>'+str(int(10000*(fm))/100.0)+'%</td></tr>\n'
    resHtml+='</table></body></html>'
    open(p['outReportDir']+'index.html','w').write(resHtml)

    
        #careIdx=strCare.sum(axis=0)>0


if __name__=='__main__':
    gtDir=sys.argv[1]
    submFiles=sys.argv[2:]
    imgGtSubmFnames=[(gtDir+f.split('/')[-1].split('.')[0]+'.jpg',gtDir+f.split('/')[-1].split('.')[0]+'.gt.txt',f) for f in submFiles]
    getReport(imgGtSubmFnames,dontCare='')
