# -*- coding: utf-8 -*-
"""
Created on Wed Nov  9 17:12:36 2016

@author: dena
"""

#python ~/Projects/CharacterDetection/HeatMaps/wordSpottingTP.py "SEPHORA"  /home/fcn/wordSpotting/ICDAR-VAL/img_817.mat
#python ~/Projects/CharacterDetection/HeatMaps/wordSpottingTP.py "VACHERON"  /home/fcn/wordSpotting/ICDAR-VAL/img_820.mat
#python /home/fcn/wordSpotting/wordSpottingTP.py "CONSTANTIN"  /home/fcn/wordSpotting/ICDAR-VAL/img_820.mat


import scipy
import scipy.io
import sys
import numpy as np
from matplotlib import pyplot as plt
import cv2
import csv
from plotly import tools
import plotly.plotly as py

def getIimg(matFname):
    matData=scipy.io.loadmat(matFname)
    #matData['heatMap'] = matData.keys()[1]
    iimg=matData['heatMap'].cumsum(axis=0).cumsum(axis=1)
    return matData, iimg
    
    
#def slidingWindow(iimg,qHist,dx=8,dy=8,heights=[5,8,11,14,17,22,28,32,48],longalities=[1,2,3,4,6,8,11]):
#    resLTRB=np.empty([(iimg.shape[1]/dx)*(iimg.shape[0]/dy)*len(longalities)*len(heights),5])
#    idx=0
#    for letHeight in heights:
#        print letHeight
#        for longality in longalities:
#            width=letHeight*longality
#            surface=width*letHeight
#            for left in range(1,(iimg.shape[1]-width)-1,dx):
#                for top in range(1,(iimg.shape[0]-letHeight)-1,dy):
#                    right=left+width
#                    bottom=top+letHeight
#                    enrgyVect=iimg[bottom,right,:]+iimg[top,left,:]-(iimg[bottom,left,:]+iimg[top,right,:])
#                    enrgyVect/=surface
#                    resLTRB[idx,:]=[left,top,right,bottom,np.sum(enrgyVect[1:]*qHist[1:])]
#                    idx+=1
#    return resLTRB
#'/home/fcn/dena/icdar_ch4_val/conf_hm_ICDAR_FCN_400_epoch/img_801.csv'
    
def textProposals(iimg,qHist):
    idx=0
    #TPcsv = open((fname+'.csv'),'rb')
    #TP = csv.reader(TPcsv)
    #TP = TP[TP[:,4].argsort()]
    lines=[l.strip().split(',') for l in open((fname+'.csv')).read().split('\n') if len(l)>0]
    BB = np.empty([len(lines),5], dtype='f')
    for ii in range(0,len(lines)):
        for jj in range(0,4):
            BB[ii,jj] = lines[ii][jj] #converting the list of list to a numpy array
     

       
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
        
#        for dd in range(0,38):
#            matData['heatMap'][:][:][dd]

        enrgyVect=iimg[bottom,right,:]+iimg[top,left,:]-(iimg[bottom,left,:]+iimg[top,right,:])
#        print 'energy vector %s'%enrgyVect
#        print 'size of energy vector %s'%(enrgyVect.shape)
        enrgyVect/=surface
        BBEnergy[idx,:] = enrgyVect 
        #resLTRB[idx,:]=[left,top,right,bottom,np.sum(enrgyVect[1:]*qHist[1:])]
        idx+=1
    
    #L2Normalization    
    BBW = np.sqrt(np.sum(BBEnergy**2))
    BBEnergyNorm = BBEnergy/BBW
    qHistW = np.sqrt(np.sum(qHist**2))
    qHistNorm = qHist/qHistW
    #dotProduct between energy vector of each bounding box and histogram of query
    resLTRB=[BBsrt[:,0], BBsrt[:,1] , BBsrt[:,2]+BBsrt[:,0] , BBsrt[:,3]+BBsrt[:,1] , np.dot(BBEnergyNorm,qHistNorm)]
    
    return resLTRB   
    


if __name__=='__main__':
    
    #for query in sys.argv[1:]:
    query = sys.argv[1]
    #query = "SEPHORA"
    char = list(query)
    queryLength = len(char)
    qHist=np.zeros(38)
    for cc in range(0,queryLength):
        Character = char[cc]
        if (Character == 'a') or  (Character == 'A'):
            qHist[1] += 1 
        elif (Character == 'b') or  (Character == 'B'):
            qHist[2] += 1
        elif (Character == 'c') or  (Character == 'C'):
            qHist[3] += 1
        elif (Character == 'd') or  (Character == 'D'):
            qHist[4] += 1
        elif (Character == 'e') or  (Character == 'E'):
            qHist[5] += 1
        elif (Character == 'f') or  (Character == 'F'):
            qHist[6] += 1
        elif (Character == 'g') or  (Character == 'G'):
            qHist[7] += 1
        elif (Character == 'h') or  (Character == 'H'):
            qHist[8] += 1
        elif (Character == 'i') or  (Character == 'I'):
            qHist[9] += 1
        elif (Character == 'j') or  (Character == 'J'):
            qHist[10] += 1
        elif (Character == 'k') or  (Character == 'K'):
            qHist[11] += 1
        elif (Character == 'l') or  (Character == 'L'):
            qHist[12] += 1
        elif (Character == 'm') or  (Character == 'M'):
            qHist[13] += 1
        elif (Character == 'n') or  (Character == 'N'):
            qHist[14] += 1
        elif (Character == 'o') or  (Character == 'O'):
            qHist[15] += 1
        elif (Character == 'p') or  (Character == 'P'):
            qHist[16] += 1
        elif (Character == 'q') or  (Character == 'Q'):
            qHist[17] += 1
        elif (Character == 'r') or  (Character == 'R'):
            qHist[18] += 1
        elif (Character == 's') or  (Character == 'S'):
            qHist[19] += 1
        elif (Character == 't') or  (Character == 'T'):
            qHist[20] += 1
        elif (Character == 'u') or  (Character == 'U'):
            qHist[21] += 1
        elif (Character == 'v') or  (Character == 'V'):
            qHist[22] += 1
        elif (Character == 'w') or  (Character == 'W'):
            qHist[23] += 1
        elif (Character == 'x') or  (Character == 'X'):
            qHist[24] += 1
        elif (Character == 'y') or  (Character == 'Y'):
            qHist[25] += 1
        elif (Character == 'z') or  (Character == 'Z'):
            qHist[26] += 1
        elif (Character == '1'): 
            qHist[27] += 1
        elif (Character == '2'): 
            qHist[28] += 1
        elif (Character == '3'): 
            qHist[29] += 1
        elif (Character == '4'): 
            qHist[30] += 1
        elif (Character == '5'): 
            qHist[31] += 1
        elif (Character == '6'): 
            qHist[32] += 1
        elif (Character == '7'):
            qHist[33] += 1
        elif (Character == '8'): 
            qHist[34] += 1
        elif (Character == '9'): 
            qHist[35] += 1
        elif (Character == '0'): 
            qHist[36] += 1          
        elif (Character == '!') or (Character == '$') or (Character == '>') or (Character == '<') or (Character == '.') or (Character == ':') or (Character == '-') or (Character == '_') or (Character == '(') or (Character == ')') or (Character == '[') or (Character == ',') or (Character == ';') or (Character == '"') or (Character == '#') or (Character == '?') or (Character == '%') or (Character == '*') or (Character == '/') or (Character == '@') or (Character == '|') or (Character == '&') or (Character == '=') or (Character == '+') or (Character == '{') or (Character == '}') or (Character == '`')  or (Character == '^') or (Character == '€'):   #||(Character == '\') || (Character == ''')  
            qHist[37] += 1
        else:
            qHist[37] += 1
    
        
    for fname in sys.argv[2:]:
    #for fname in'/home/fcn/dena/icdar_ch4_val/conf_hm_ICDAR_FCN_400_epoch/img_801.mat':
        fname='.'.join(fname.split('.')[:-1])
        img=cv2.imread(fname+'.jpg')
#        qHist=np.zeros(38)
#        qHist[[3,1]]=1
#        qHist[12]=2
        matData, iimg=getIimg(fname+'.mat')
        #res=slidingWindow(iimg,qHist)
        res=textProposals(iimg,qHist)

        resArr = np.empty([len(res[0]),len(res)])
        for ii in range(0,len(res[0])):
            for jj in range(0,len(res)):
                resArr[ii,jj] = res[jj][ii] #converting the transport of list of list to a numpy array
        
        
        surf=(resArr[:,0]-resArr[:,2])*(resArr[:,1]-resArr[:,3])
        res=resArr[surf>1200,:]
        idx=np.argsort(res[:,4])
        print res.shape
        fig = plt.figure()
        queryBB = fig.add_subplot(2,1,1)
        plt.imshow(img)
        for k in idx[-1:]:
            print k
            #print res[k,:]
            [l,t,r,b]=res[k,:4]
            #queryBB.plt.plot([l,l,r,r,l],[t,b,b,t,t])
            queryBB.plot([l,l,r,r,l],[t,b,b,t,t])           
            #TextBoxes = ([l,l,r,r,l],[t,b,b,t,t])
            #fig.append_trace(TextBoxes,1)
        
        qHM = fig.add_subplot(2,1,2)
        queryHM = np.ones(((matData['heatMap'].shape[0]), (matData['heatMap'].shape[1])))
        for dd in range (len(qHist)):
            if qHist[dd] != 0:
               queryHM = np.multiply ( queryHM, (matData['heatMap'][:,:,dd]))
        #qHM.plot(queryHM)
        hmap_0 = matData['heatMap'][:,:,0]
        hmap_0 = np.exp(hmap_0)
        queryHM = np.exp(queryHM)
        queryHM_softmax = queryHM / (hmap_0 + queryHM)
        plt.imshow(queryHM_softmax)        
        plt.show()
        
        print 'query is %s and queryLengthis %s'%(query,queryLength)
        
        
        
        
        
        
        
        
        
        
        
#        # subplot of all the heatmaps and proposed text boxes
#        #fig = tools.make_subplots(rows=1, cols=1+queryLength, subplot_titles=('TextProposals', '%s'%char[0],'%s'%char[1],' %s'%char[2], '%s'%char[3], '%s'%char[4], '%s'%char[5], '%s'%char[6]))
#        fig = tools.make_subplots(rows=1, cols=queryLength, subplot_titles=('%s'%char[0],'%s'%char[1],' %s'%char[2], '%s'%char[3], '%s'%char[4], '%s'%char[5], '%s'%char[6]))
#        #fig.append_trace(TextBoxes,1)    
#        fig.append_trace((matData['heatMap'][:][:][19]),1)#s
#        fig.append_trace((matData['heatMap'][:][:][5]),1)#E
#        fig.append_trace((matData['heatMap'][:][:][16]),1)#p
#        fig.append_trace((matData['heatMap'][:][:][8]),1)#h
#        fig.append_trace((matData['heatMap'][:][:][15]),1)#o
#        fig.append_trace((matData['heatMap'][:][:][18]),1)#r
#        fig.append_trace((matData['heatMap'][:][:][1]),1)#a      
#
#    fig['layout'].update(height=600, width=600, title='Multiple Subplots' + ' with Titles')
#
#    plot_url = py.plot(fig, filename='make-subplots-multiple-with-title')

