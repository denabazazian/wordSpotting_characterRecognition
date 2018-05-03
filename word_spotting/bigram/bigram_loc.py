# -*- coding: utf-8 -*-
"""
Created on Thu Nov 23 15:49:29 2017

@author: dena
"""

# cd /home/dena/Projects/caffe_CharacterDetection/
# python bigram_loc.py /home/fcn/wordSpotting/icdar_ch4_testSet/img_*.jpg
# python bigram_loc.py /home/fcn/wordSpotting/icdarCh4_val_fcnEp113/img_*.jpg

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
    
    return iimg,heatMap


def getQueryHist(qword,minusOneForMissing=False):
    qHist=np.zeros([len(alphabet)])
    m=defaultdict(lambda: len(alphabet)-1)
    m.update({alphabet[k]:k for k in range(1,len(alphabet))})
    for c in qword.lower():
        qHist[m[c]]+=1
    if minusOneForMissing:
        qHist[qHist==0]=-1
    return qHist[:]

#def bigram(hmMix, c1, c2):


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
    char2int = {'a':1,'A':1,'b':2,'B':2,'c':3,'C':3,'d':4,'D':4,'e':5,'E':5,'f':6,'F':6,'g':7,'G':7,'h':8,'H':8,'i':9,'I':9,'j':10,'J':10,'k':11,'K':11,'l':12,'L':12,'m':13,'M':13,'n':14,'N':14,'o':15,'O':15,'p':16,'P':16,'q':17,'Q':17,'r':18,'R':18,'s':19,'S':19,'t':20,'T':20,'u':21,'U':21,'v':22,'V':22,'w':23,'W':23,'x':24,'X':24,'y':25,'Y':25,'z':26,'Z':26,'1':27,'2':28,'3':29,'4':30,'5':31,'6':32,'7':33,'8':34,'9':35,'0':36,'!':37,'$':37,'>':37,'<':37,'.':37,':':37,'-':37,'_':37,'(':37,')':37,'[':37,']':37,'{':37,'}':37,',':37,';':37,'#':37,'?':37,'%':37,'*':37,'/':37,'@':37,'^':37,'&':37,'=':37,'+':37,'€':37,"'":37,'`':37,'"':37,'\\':37,'\xc2':37,'\xb4':37,' ':37,'\xc3':37,'\x89':37} #'\':37
    
    #for img_name in sys.argv[1:]:
    for debugingMode in range(0,1):
        #sys.argv -> img_name = '/home/fcn/wordSpotting/icdar_ch4_testSet/img_873.jpg'        
        #img_name = '/home/fcn/wordSpotting/icdar_ch4_testSet/img_499.jpg' #turning domain
        #img_name = '/home/fcn/wordSpotting/icdar_ch4_testSet/img_332.jpg'  # swatch
        #img_name = '/home/fcn/wordSpotting/icdar_ch4_testSet/img_327.jpg'   #knowledge
        #img_name = '/home/fcn/wordSpotting/icdar_ch4_testSet/img_341.jpg'   #the soup soom
        img_name = '/home/dena/Desktop/crop.jpg'

        img = cv2.imread(img_name)       
        print img_name

        cv_size = lambda img: tuple(img.shape[1::-1])         
        width,height = cv_size(img)

        ### Read a mat file to get the integral image and the heat map from softmax
        #iimg, ientropy=getIimg(img_name,net_char)
        iimg,hm = getIimg(img_name,net_char)
        # get the scores of text/non text classifier
        #iimgTNT, ientropyTNT = getIimg(img_name,net_text)
        iimgTNT,hmTNT = getIimg(img_name,net_text)
        
        hmMix = np.zeros((height,width,38))        
        hmMix[:,:,0] = hm[:,:,0] * hmTNT[:,:,0]
        for envct in range(1,38):
            hmMix[:,:,envct] = hmTNT[:,:,1]* hm[:,:,envct]
        
        #for each query word
        voc  = open (img_name.split('.')[0]+'.txt').read()  #for the ch4 validation
        numQWords =  voc.count('\n')
        voc=re.sub(r'[^\x00-\x7f]',r'',voc)
        voc = voc.split('\r\n')
        idnqw = 0                              
        for nqw in range(0,numQWords): 
            #qword = voc[nqw]
            if (len(voc[nqw])>0 and voc[nqw].split(',')[-1].strip()!='###'):
               idnqw +=1
               queryword = voc[nqw].split(',')[-1] 
               #queryword = 'traffic'
               print queryword
               hmQ = np.zeros((height,width))
               for c in range(0,(len(queryword)-1)):
                    #hmQ += (hmMix[:,:,char2int[queryword[c]]] * hmMix[:,:,char2int[queryword[c+1]]])
                    hmQ += (hm[:,:,char2int[queryword[c]]] * hm[:,:,char2int[queryword[c+1]]])
                    #hmQ += (hm[:,:,char2int[queryword[c]]] + hm[:,:,char2int[queryword[c+1]]])
                    #plt.imshow(hm[:,:,char2int[queryword[c]]] * hm[:,:,char2int[queryword[c+1]]]);plt.imshow(img,alpha=.5);plt.title(queryword[c]+','+queryword[c+1]); plt.show()


               plt.imshow(hmQ);plt.imshow(img,alpha=.5);plt.title(queryword); plt.show()


   #plt.imshow(hm[:,:,char2int['c']] * hm[:,:,char2int['a']]);plt.imshow(img,alpha=.5);plt.title(queryword); plt.show()


    # hmW = np.sum(hmMix[:,:,1:38], axis=2)
    #plt.imshow(-1*hm[:,:,0]);plt.imshow(img,alpha=.5);plt.title('text'); plt.show()
    #plt.imshow(-1*hmMix[:,:,0]);plt.imshow(img,alpha=.5);plt.title('text'); plt.show()
    # plt.imshow(hmMix[:,:,0]);plt.imshow(img,alpha=.5);plt.title('Background'); plt.show()
    # plt.imshow(hmW);plt.imshow(img,alpha=.5);plt.title('Text'); plt.show()



    # plt.imshow(hmTNT[:,:,1])
    # plt.show()

    fig = plt.figure()
    #fig.suptitle(str(m['imm_path_string']).split(',')[0].split('u')[-1].strip(']').replace("'",'')+"="+transcription, fontsize=14)  
    #fig.subplots_adjust(hspace=.1)
    #fig.subplots_adjust(wspace=.1)

    for i in range (0,38):
        fig.add_subplot(8,5,(i+1)).imshow(hm[:,:,i])
        plt.imshow(img,alpha=.5)
        fig.add_subplot(8,5,(i+1)).set_title(('%d,%s')%(i,alphabet[i]), fontsize=9)
        plt.xticks([])
        plt.yticks([]) 
        #fig.add_subplot(8,5,(i+1)).set_title(i)                  
        fig.savefig('/home/dena/Desktop/hm_allChar.png', dpi = 800)   
        # Turn off tick labels
        plt.xticks([])
        plt.yticks([])         
    plt.show()


    # fig = plt.figure()
    # for i in range (0,38):
    #     fig.add_subplot(8,5,(i+1)).imshow(hmMix[:,:,i])
    #     fig.add_subplot(8,5,(i+1)).set_title(('%d,%s')%(i,alphabet[i]))
    #     #fig.add_subplot(8,5,(i+1)).set_title(i)                  
    #     #fig.savefig('/home/dena/Projects/softPHOC_tensorflow/debugingCodes/batch_debug/annot.png', dpi = 300)               
    # plt.show()