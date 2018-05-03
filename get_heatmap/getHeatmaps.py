# cd /Projects/caffe_CharacterDetection/synth-fcn8s-atonce/
# python getHeatmapsCharacter.py 


import sys
sys.path.append('/home/dena/Software/caffe/python')
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import caffe
import surgery, score
from scipy.misc import imresize, imsave, toimage
import time


# Run in GPU
caffe.set_device(0)
caffe.set_mode_gpu()

#Compute heatmaps from images in txt
#val = np.loadtxt('/Projects/caffe_CharacterDetection/SynthImgNamesVal_100samples.txt', dtype=str)
val = np.loadtxt('/datasets/ICDAR/ICDAR_val_names.txt', dtype=str)
#val ='138/punting_42_41' #,'70/hedge_42_26'
# load net
#net = caffe.Net('/Projects/caffe_CharacterDetection/synth-fcn8s-atonce/deploy.prototxt', '/Projects/caffe_CharacterDetection/snapshot/train_iter_411000.caffemodel', caffe.TEST)
#net = caffe.Net('/Projects/caffe_CharacterDetection/synth-fcn8s-atonce/deploy.prototxt', '/Projects/caffe_CharacterDetection/snapshot/train_iter_3250000.caffemodel', caffe.TEST)
#net = caffe.Net('/Projects/caffe_CharacterDetection/synth-fcn8s-atonce/deploy.prototxt', '/Projects/caffe_CharacterDetection/snapshot/train_iter_4330000.caffemodel', caffe.TEST)
#net = caffe.Net('/Projects/caffe_CharacterDetection/synth-fcn8s-atonce/deploy.prototxt', '/Projects/caffe_CharacterDetection/snapshot/train_iter_2180000.caffemodel', caffe.TEST) # good results
net = caffe.Net('/Projects/caffe_CharacterDetection/synth-fcn8s-atonce/deploy.prototxt', '/Projects/caffe_CharacterDetection/snapshot/train_iter_1360000.caffemodel', caffe.TEST) # good results

print 'Computing heatmaps ...'

count = 0
start = time.time()

#for idx in range(0,len(val)):
for idx in range(33,35):
#for idx in range(120,121):
    count = count + 1
    if count % 100 == 0:
        print count

    # load image
    #im = Image.open('/datasets/synthtext/input_images/' + val[idx]+'.jpg')
    im = Image.open('/datasets/ICDAR/ICDAR_VAL/' + val[idx]+'.jpg')
    print idx

    # Turn grayscale images to 3 channels
    if (im.size.__len__() == 2):
        im_gray = im
        im = Image.new("RGB", im_gray.size)
        im.paste(im_gray)

    #switch to BGR and substract mean
    in_ = np.array(im, dtype=np.float32)
    in_ = in_[:,:,::-1]
    in_ -= np.array((104.00698793,116.66876762,122.67891434))
    in_ = in_.transpose((2,0,1))

    # shape for input (data blob is N x C x H x W)
    net.blobs['data'].reshape(1, *in_.shape)
    net.blobs['data'].data[...] = in_

    # run net and take scores
    net.forward()


    # Heatmap computation
    scores_exp = np.exp(net.blobs['score_conv'].data[0][:, :, :])
    sum_exp = np.sum (scores_exp, axis=0)
    heatMap = np.empty((im.size[1], im.size[0], 38))
    for ii in range(0,38):    
        heatMap[:,:,ii] = scores_exp[ii,:,:]/sum_exp



    #Show the heatmap for some of the characters
    #charVal = 1+ ord('a')-ord('a')
    #plt.imshow(heatMap[:,:,charVal]);plt.imshow(inpImg,alpha=.5);plt.show()
    #in case of looking for numbers
    #plt.imshow(heatMap[:,:,29]);plt.imshow(inpImg,alpha=.5);plt.show()
    #heatmap of the background
    plt.imshow(heatMap[:,:,0]);plt.imshow(im,alpha=.5);plt.title('Background'); plt.show()
    
    plt.imshow(heatMap[:,:,1+ ord('a')-ord('a')]);plt.imshow(im,alpha=.5);plt.title('a-%d'% (1+ ord('a')-ord('a'))); plt.show()  #plt.colorbar();
    plt.imshow(heatMap[:,:,1+ ord('b')-ord('a')]);plt.imshow(im,alpha=.5);plt.title('b-%d'% (1+ ord('b')-ord('a'))); plt.show()
    plt.imshow(heatMap[:,:,1+ ord('c')-ord('a')]);plt.imshow(im,alpha=.5);plt.title('c-%d'% (1+ ord('c')-ord('a'))); plt.show()
    plt.imshow(heatMap[:,:,1+ ord('d')-ord('a')]);plt.imshow(im,alpha=.5);plt.title('d-%d'% (1+ ord('d')-ord('a'))); plt.show()
    plt.imshow(heatMap[:,:,1+ ord('e')-ord('a')]);plt.imshow(im,alpha=.5);plt.title('e-%d'% (1+ ord('e')-ord('a'))); plt.show()
    plt.imshow(heatMap[:,:,1+ ord('f')-ord('a')]);plt.imshow(im,alpha=.5);plt.title('f-%d'% (1+ ord('f')-ord('a'))); plt.show()
    plt.imshow(heatMap[:,:,1+ ord('g')-ord('a')]);plt.imshow(im,alpha=.5);plt.title('g-%d'% (1+ ord('g')-ord('a'))); plt.show()
    plt.imshow(heatMap[:,:,1+ ord('h')-ord('a')]);plt.imshow(im,alpha=.5);plt.title('h-%d'% (1+ ord('h')-ord('a'))); plt.show()
    plt.imshow(heatMap[:,:,1+ ord('i')-ord('a')]);plt.imshow(im,alpha=.5);plt.title('i-%d'% (1+ ord('i')-ord('a'))); plt.show()
    plt.imshow(heatMap[:,:,1+ ord('j')-ord('a')]);plt.imshow(im,alpha=.5);plt.title('j-%d'% (1+ ord('j')-ord('a'))); plt.show()
    plt.imshow(heatMap[:,:,1+ ord('k')-ord('a')]);plt.imshow(im,alpha=.5);plt.title('k-%d'% (1+ ord('k')-ord('a'))); plt.show()
    plt.imshow(heatMap[:,:,1+ ord('l')-ord('a')]);plt.imshow(im,alpha=.5);plt.title('l-%d'% (1+ ord('l')-ord('a'))); plt.show()
    plt.imshow(heatMap[:,:,1+ ord('m')-ord('a')]);plt.imshow(im,alpha=.5);plt.title('m-%d'% (1+ ord('m')-ord('a'))); plt.show()
    plt.imshow(heatMap[:,:,1+ ord('n')-ord('a')]);plt.imshow(im,alpha=.5);plt.title('n-%d'% (1+ ord('n')-ord('a'))); plt.show()
    plt.imshow(heatMap[:,:,1+ ord('o')-ord('a')]);plt.imshow(im,alpha=.5);plt.title('o-%d'% (1+ ord('o')-ord('a'))); plt.show()
    plt.imshow(heatMap[:,:,1+ ord('p')-ord('a')]);plt.imshow(im,alpha=.5);plt.title('p-%d'% (1+ ord('p')-ord('a'))); plt.show()
    plt.imshow(heatMap[:,:,1+ ord('q')-ord('a')]);plt.imshow(im,alpha=.5);plt.title('q-%d'% (1+ ord('q')-ord('a'))); plt.show()
    plt.imshow(heatMap[:,:,1+ ord('r')-ord('a')]);plt.imshow(im,alpha=.5);plt.title('r-%d'% (1+ ord('r')-ord('a'))); plt.show()
    plt.imshow(heatMap[:,:,1+ ord('s')-ord('a')]);plt.imshow(im,alpha=.5);plt.title('s-%d'% (1+ ord('s')-ord('a'))); plt.show()
    plt.imshow(heatMap[:,:,1+ ord('t')-ord('a')]);plt.imshow(im,alpha=.5);plt.title('t-%d'% (1+ ord('t')-ord('a'))); plt.show()
    plt.imshow(heatMap[:,:,1+ ord('u')-ord('a')]);plt.imshow(im,alpha=.5);plt.title('u-%d'% (1+ ord('u')-ord('a'))); plt.show()
    plt.imshow(heatMap[:,:,1+ ord('v')-ord('a')]);plt.imshow(im,alpha=.5);plt.title('v-%d'% (1+ ord('v')-ord('a'))); plt.show()
    plt.imshow(heatMap[:,:,1+ ord('w')-ord('a')]);plt.imshow(im,alpha=.5);plt.title('w-%d'% (1+ ord('w')-ord('a'))); plt.show()
    plt.imshow(heatMap[:,:,1+ ord('x')-ord('a')]);plt.imshow(im,alpha=.5);plt.title('x-%d'% (1+ ord('x')-ord('a'))); plt.show()
    plt.imshow(heatMap[:,:,1+ ord('y')-ord('a')]);plt.imshow(im,alpha=.5);plt.title('y-%d'% (1+ ord('y')-ord('a'))); plt.show()
    plt.imshow(heatMap[:,:,1+ ord('z')-ord('a')]);plt.imshow(im,alpha=.5);plt.title('z-%d'% (1+ ord('z')-ord('a'))); plt.show()    
#
#    plt.imshow(heatMap[:,:,1+ ord('f')-ord('a')])


    #Save CSV heatmap
    # pixels = np.asarray(hmap_softmax)
    # np.savetxt('/csv/' + idx + '.csv', pixels, delimiter=",")

    #Save PNG softmax heatmap
    #hmap_softmax_2save = (255.0 * hmap_softmax).astype(np.uint8)
    #hmap_softmax_2save = Image.fromarray(hmap_softmax_2save)
    #hmap_softmax_2save.save('/Projects/caffe_CharacterDetection/heatmaps/Synth/' + idx[:-4] + '.png')


    # Save color softmax heatmap
    # fig = plt.figure(frameon=False)
    # fig.set_size_inches(5.12,5.12)
    # ax = plt.Axes(fig, [0., 0., 1., 1.])
    # ax.set_axis_off()
    # fig.add_axes(ax)
    # ax.imshow(hmap_softmax, aspect='auto', cmap="jet")
    # fig.savefig('/heatmaps/' + idx + '-ht.jpg')
    # plt.close(fig)


    #print 'Heatmap saved for image: ' +idx
    #print 'Heatmap saved for image: ' +val[idx]
end = time.time()
print 'Total time elapsed in heatmap computations'
print(end - start)
print 'Time per image'
print(end - start)/val.__len__()
