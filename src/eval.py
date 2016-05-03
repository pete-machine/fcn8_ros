import os

#cafferootss=os.environ["CAFFE_ROOTSS"]
#os.environ["CAFFE_ROOT"] = cafferootss
#os.environ["PYTHONPATH"]=os.path.join(cafferootss,'python')

import numpy as np
from PIL import Image
import matplotlib
import matplotlib.pyplot as plt
from random import random
import time
import caffe
import glob


dirImages = "/home/repete/DeepLearningStuff/ModelsCaffe/fsm-xs/32s/"
dirsImages = glob.glob(dirImages + "*.jpg");
dirImagesOutput = dirImages + "/segmented";

if not os.path.exists(dirImagesOutput):
    os.makedirs(dirImagesOutput)

for dirImage in dirsImages:
    print "Name %s " % dirImage
    dirParts = dirImage.split('/');
    print "DirPart %s " % dirParts[len(dirParts)-1]
    print dirImagesOutput + dirParts[len(dirParts)-1].split('.')[0] + '.png';
    
# load net
caffe.set_mode_cpu()
#net = caffe.Net('deploy.prototxt', 'fcn-8s-pascalcontext.caffemodel', caffe.TEST)
net = caffe.Net('deploy.prototxt', 'fcn-8s-pascalcontext.caffemodel', caffe.TEST)
#caffe.set_mode_gpu()
#caffe.set_device(0)

# load image, switch to BGR, subtract mean, and make dims C x H x W for Caffe
im = Image.open('Street2.jpg')
in_ = np.array(im, dtype=np.float32)
in_ = in_[:,:,::-1]
in_ -= np.array((104.00698793,116.66876762,122.67891434))
in_ = in_.transpose((2,0,1))


# shape for input (data blob is N x C x H x W), set data
net.blobs['data'].reshape(1, *in_.shape)
net.blobs['data'].data[...] = in_
# run net and take argmax for prediction

t1 = time.time();
net.forward()
print time.time()-t1
out = net.blobs['score-final'].data[0].argmax(axis=0)

#Create colormap
colors = [(1,1,1)] + [(random(),random(),random()) for i in xrange(255)]
new_map = matplotlib.colors.LinearSegmentedColormap.from_list('new_map', colors, N=256)

plt.matshow(out,cmap=new_map)
#plt.matshow()

plt.savefig('segmented.png', format='png')

pass
