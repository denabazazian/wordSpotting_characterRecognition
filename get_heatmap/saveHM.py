# -*- coding: utf-8 -*-
"""
Created on Wed Apr  5 11:39:03 2017

@author: dena
"""

#python saveHM.py /icdar_ch4_test/input/*.jpg



import sys
sys.path.append('/Software/caffe/python')
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import caffe
from scipy.misc import imresize, imsave, toimage
import time
import cv2

### Run in GPU
caffe.set_device(0)
caffe.set_mode_gpu()


    
    
def getHM(im,net):
    
    ### Turn Grayscale Images to 3 Channels
    if (im.size.__len__() == 2):
        im_gray = im
        im = Image.new("RGB", im_gray.size)
        im.paste(im_gray)
    ### Switch to BGR and Substract Mean
    in_ = np.array(im, dtype=np.float32)
    in_ = in_[:,:,::-1]
    in_ -= np.array((104.00698793,116.66876762,122.67891434))
    in_ = in_.transpose((2,0,1))   
     
    ### Shape for Input (data blob is N x C x H x W)
    net.blobs['data'].reshape(1, *in_.shape)
    net.blobs['data'].data[...] = in_
    
    
    ### Run Net and Take Scores
    net.forward()
    
    # Heatmap Computation
    scores = net.blobs['score_conv'].data[0][:, :, :]
    scores_exp = np.exp(scores)
    sum_exp = np.sum (scores_exp, axis=0)
    heatMap = np.empty((im.size[1], im.size[0], 2))
    for ii in range(0,2):    
        heatMap[:,:,ii] = scores_exp[ii,:,:]/sum_exp
        
    return 1-heatMap[:,:,0]




if __name__=='__main__':
    
    #net = caffe.Net('/home/fcn/fcnTpWs/nets/coco_2c/deploy.prototxt', '/home/fcn/fcnTpWs/nets/coco_2c/train_iter_406000.caffemodel', caffe.TEST) 
    net = caffe.Net('/home/fcn/fcnTpWs/nets/icdar_2c/deploy.prototxt', '/home/fcn/fcnTpWs/nets/icdar_2c/train_iter_10500.caffemodel', caffe.TEST) 
    #net = caffe.Net('/home/fcn/fcnTpWs/nets/synth_38c/deploy.prototxt', '/home/fcn/fcnTpWs/nets/synth_38c/train_iter_1360000.caffemodel', caffe.TEST) 


    for inpImg in sys.argv[1:]:
        #sys.argv -> inpImg = '/home/fcn/fcnTpWs/ch4Val/input/img_804.jpg'
        #inpImg = '/home/fcn/fcnTpWs/ch4Val/input/img_804.jpg'

        imgName = inpImg.split('.')[0].split('/')[-1]
        #imgDir = inpImg.split('/')[0]+'/'+inpImg.split('/')[1]+'/'+inpImg.split('/')[2]+'/'+inpImg.split('/')[3]+'/'+inpImg.split('/')[4]

        ### Read Input Image
        im = Image.open(inpImg)        
        
        ### Get the HeatMap
        hm = getHM(im,net)
        
        ### Convert to Grayscale
        hm_png = (255.0 * hm).astype(np.uint8)
        hm_png = Image.fromarray(hm_png)
        
        
        ### Write the HeatMap
        
        hmPath = '/home/fcn/'+imgName+'.png'


        #hmPath = imgDir+'/hm2Coco/'+imgName+'.png'
        #hmPath = imgDir+'/hm2Icdar/'+imgName+'.png'
        #hmPath = imgDir+'/hm38Synth/'+imgName+'.png'
             
        hm_png.save(hmPath)       
        print hmPath
        
        
        
        
