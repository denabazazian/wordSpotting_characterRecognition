#shell command:
#cd /home/dena/Projects/caffe_CharacterDetection/version2/synth-fcn8s-atonce/
#nice -20 python solve.py 0
#nice -20 python solve.py 1

import sys
#sys.path.append('/home/dena/Projects/Caffe-FCN-textNontext/fcn')

sys.path.append('/home/dena/Software/caffe/python')
import caffe

import surgery, score

import numpy as np
import os

from pylab import *
import setproctitle
setproctitle.setproctitle(os.path.basename(os.getcwd()))

#just write it for the first time
#outPut = open('/home/dena/Projects/caffe_CharacterDetection/version2/snapshot/train/vAc_vIU_vLs_tLs.csv', 'w')


outPut = open('/home/dena/Projects/caffe_CharacterDetection/version2/snapshot/train/vAc_vIU_vLs_tLs.csv', 'w')

#outPut_old = open('/home/dena/Projects/caffe_CharacterDetection/version2/snapshot/train/vAc_vIU_vLs_tLs_1_1994000.csv').read()


#for line in outPut_old:
#    outPut.write(line)
#    outPut.flush()

outPut.write('\n')
outPut.flush()


#weights = '../ilsvrc-nets/vgg16-fcn.caffemodel'
weights = '/home/dena/Projects/caffe_CharacterDetection/data/model/VGG_ILSVRC_16_layers.caffemodel'
# init
caffe.set_device(int(sys.argv[1]))
caffe.set_mode_gpu()

solver = caffe.SGDSolver('solver.prototxt')
#solver.net.copy_from(weights) # replace it with the three line afterward because of the ERROR : Cannot copy param 0 weights from layer 'fc6'; shape mismatch.  Source param shape is 1 1 4096 25088 (102760448); target param shape is 4096 512 7 7 (102760448). To learn this layer's parameters from scratch rather than copying from a saved net, rename the layer.
# the answer of Shelhamer : https://groups.google.com/forum/#!topic/caffe-users/rYihuHF4LFs

#base_net = caffe.Net('/home/dena/Projects/caffe_CharacterDetection/data/model/vgg16.prototxt', '/home/dena/Projects/caffe_CharacterDetection/data/model/VGG_ILSVRC_16_layers.caffemodel', caffe.TEST)

base_net = caffe.Net('/home/dena/Projects/caffe_CharacterDetection/data/model/VGG_ILSVRC_16_layers_deploy.prototxt', '/home/dena/Projects/caffe_CharacterDetection/data/model/VGG_ILSVRC_16_layers.caffemodel', caffe.TEST)

surgery.transplant(solver.net, base_net)
del base_net

# surgeries
interp_layers = [k for k in solver.net.params.keys() if 'up' in k]
surgery.interp(solver.net, interp_layers)

# scoring
#val = np.loadtxt('../data/segvalid11.txt', dtype=str)
#val = np.loadtxt('/home/dena/Projects/caffe_CharacterDetection/SynthImgNamesVal.txt', dtype=str)
#dena@CVC216:~/datasets/synthtext$ cat SynthImgNamesVal.txt|shuf|head -n 500 > SynthImgNamesVal500.txt

val = np.loadtxt('/home/dena/Projects/caffe_CharacterDetection/SynthImgNamesVal_100samples.txt', dtype=str)


#for _ in range(75):
#    #solver.step(4000)
#    solver.step(10)
#    score.seg_tests(solver, False, val, layer='score')

# change batch-size
# print(solver.net.blobs['data'].data.shape)
# solver.net.blobs['data'].reshape(1, 3, 720, 1280)

#load snapshot
#solver.restore('/home/dena/Projects/caffe_CharacterDetection/version2/snapshot/train_iter_642000.solverstate')
#solver.restore('/home/dena/Projects/caffe_CharacterDetection/version2/snapshot/train_iter_1994000.solverstate')
#solver.restore('/home/dena/Projects/caffe_CharacterDetection/version2/snapshot/train_iter_4310000.solverstate')
solver.restore('/home/dena/Projects/caffe_CharacterDetection/version2/snapshot/train_iter_4773000.solverstate')

# init vars to train and  store results
size_intervals = 1000 #500 #4000 No of iterations between each validation and plot
num_intervals = 250000  #2500 No of times to validate and plot
total_iterations = size_intervals * num_intervals # 2500*4000 = 10.000.000 total iterations

# set plots data
train_loss = np.zeros(num_intervals)
val_loss = np.zeros(num_intervals)
val_acc = np.zeros(num_intervals)
val_iu = np.zeros(num_intervals)
it_axes = (arange(num_intervals) * size_intervals) + size_intervals


#copy the previouse
#lines=[l.strip().split(',') for l in open(outPut_old).read().split('\n') if len(l)>0] 
#for lineNum in range(0,len(lines)):
#     for colNum in range(0,4):
#         val_acc[lineNum,colNum] = lines[lineNum][colNum] #converting the list of list to a numpy array




_, ax1 = subplots()
ax2 = ax1.twinx()
ax1.set_xlabel('iteration')
ax1.set_ylabel('train loss (b) - val loss (r)')
ax2.set_ylabel('val accuracy (y) - val iu (g)')
ax2.set_autoscaley_on(False)
ax2.set_ylim([0, 1])

print "one step"
solver.step(1)
print "before for loop"

for it in range(num_intervals):

    solver.step(size_intervals)
    # solver.net.forward()

    # Test with validation set every 'size_intervals' iterations
    [loss, acc, iu] = score.seg_tests(solver, False, val, layer='score')
    val_acc[it] = acc
    val_iu[it] = iu
    val_loss[it] = loss
    train_loss[it] = solver.net.blobs['loss'].data

    outPut.write('%f;%f;%f;%f\n'%(val_acc[it],val_iu[it],val_loss[it],train_loss[it]))
    outPut.flush()
    # Plot results
    if it > 0:
        ax1.lines.pop(1)
        ax1.lines.pop(0)
        ax2.lines.pop(1)
        ax2.lines.pop(0)

    ax1.plot(it_axes[0:it+1], train_loss[0:it+1], 'b') #Training loss averaged last 20 iterations
    ax1.plot(it_axes[0:it+1], val_loss[0:it+1], 'r')    #Average validation loss
    ax2.plot(it_axes[0:it+1], val_acc[0:it+1], 'y') #Average validation accuracy (mean accuracy of text and background)
    ax2.plot(it_axes[0:it+1], val_iu[0:it+1], 'g')  #Average intersecction over union of score-groundtruth masks

    #ax1.plot(it_axes[0:solver.iter], train_loss[0:solver.iter], 'b') #Training loss averaged last 20 iterations
    #ax1.plot(it_axes[0:solver.iter], val_loss[0:solver.iter], 'r')    #Average validation loss
    #ax2.plot(it_axes[0:solver.iter], val_acc[0:solver.iter], 'y') #Average validation accuracy (mean accuracy of text and background)
    #ax2.plot(it_axes[0:solver.iter], val_iu[0:solver.iter], 'g')  #Average intersecction over union of score-groundtruth masks

    plt.ion()
    plt.show()
    plt.pause(0.001)
    plt_dir = '/home/dena/Projects/caffe_CharacterDetection/version2/snapshot/train/training-' + str(solver.iter) + '.png' #Save graph to disk every "size intervals"
    savefig(plt_dir, bbox_inches='tight')

outPut.close()

