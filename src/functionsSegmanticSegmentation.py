import os
import sys
#cafferootss=os.environ["CAFFE_ROOTSS"]
#os.environ["CAFFE_ROOT"] = cafferootss
#os.environ["PYTHONPATH"]=os.path.join(cafferootss,'python')

import numpy as np
import scipy.io as sio
from PIL import Image
import matplotlib.pyplot as plt
import time
import caffe


def initCaffeSS(dirArchi,dirModel,dirRemapping):
    test = sio.loadmat(dirRemapping)
    classRemapping = np.concatenate((np.array([0]),np.array(test['newSorting'][:,1])),axis=0)
    
    # load net
    caffe.set_mode_cpu()
    
    net = caffe.Net(dirArchi, dirModel, caffe.TEST)
    #net = caffe.Net('fcn-32s-pascal-deploy.prototxt', 'fcn-32s-pascalcontext.caffemodel', caffe.TEST)
    
    # load image, switch to BGR, subtract mean, and make dims C x H x W for Caffe
    return net,classRemapping
    
def predictImageSS(net,im,classRemapping):
    #t1 = time.clock();
    in_ = np.array(im, dtype=np.float32)
    in_ = in_[:,:,::-1]
    in_ -= np.array((104.00698793,116.66876762,122.67891434))
    in_ = in_.transpose((2,0,1))


    # shape for input (data blob is N x C x H x W), set data
    net.blobs['data'].reshape(1, *in_.shape)
    net.blobs['data'].data[...] = in_
    #print "predictImageSS: Prior forward pass time:", time.clock()-t1, "seconds"
    
    # run net and take argmax for prediction
    caffe.set_mode_gpu()
    caffe.set_device(2)
    #t1 = time.clock();
    net.forward()
    #print "predictImageSS: Forward pass time:", time.clock()-t1, "seconds"


    #t1 = time.clock();
    out = net.blobs['score-final'].data[0].argmax(axis=0)
    maxValues = net.blobs['score-final'].data[0].max(axis=0)
    return out,maxValues
#    maxValues = net.blobs['score-final'].data[0].max(axis=0)
#    for iType in range(0,len(objectType)):
#        if(objectType[iType]==True):
#            predictionRemappedProbability = np.zeros(out.shape)
#            test = np.in1d(out, np.array(np.argwhere(classRemapping==objectType)))
#            predictionRemapped = np.reshape(test,(out.shape)) # True for valid classes
#            predictionRemappedProbability[predictionRemapped] = maxValues[predictionRemapped]
#    #print "predictImageSS: Post forward pass time", time.clock()-t1, "seconds"
#    return predictionRemappedProbability

#im = Image.open('Street2.jpg')
#dirArchi = '/home/repete/blank_ws/src/semantic_segmentation/models/fcn-8s-pascal-deploy.prototxt'
#dirModel = '/home/repete/blank_ws/src/semantic_segmentation/models/fcn-8s-pascalcontext.caffemodel'
#dirRemapping = "/home/repete/Code/PascalContext/MATLAB/remappingObjectTypes.mat"
#objectType = 8
#net,classRemapping = initCaffeSS(dirArchi,dirModel,dirRemapping)
#
#predictionRemapped, predictionRemappedProbability = predictImageSS(net,im,objectType,classRemapping)
#plt.matshow(predictionRemapped)
#plt.matshow(predictionRemappedProbability)
