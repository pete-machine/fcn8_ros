import os
import sys
#cafferootss=os.environ["CAFFE_ROOTSS"]
#os.environ["CAFFE_ROOT"] = cafferootss
#os.environ["PYTHONPATH"]=os.path.join(cafferootss,'python')
sys.path.append("/usr/lib/python2.7/dist-packages")
import numpy as np
import scipy.io as sio
import caffe
 
#<<<<<<< HEAD
#=======
#import numpy as np
##import matplotlib.pyplot as plt
#from functionsSegmanticSegmentation import initCaffeSS, predictImageSS
#
#
## 0.aeroplane, 1. bicycle, 2 bird, 3.boat, 4. bottle, 5. bus, 6. car, 7. cat, 8. chair, 9. cow, 10.diningtable
## 11. dog, 12. horse, 13. motorbike, 14. person, 15. pottedplant, 16. sheep, 17. sofa, 18. train, 19. tvmonitor, 20. pedestrian
## REMAPPING (agriculture classes)
#
#def numbers_to_strings(argument):
#    switcher = {
##        0: "SS_unknown",        # 0
##        1: "SS_animal",         # 0
##        2: "SS_building",       # 0  
##        3: "SS_field",          # 1
##        4: "SS_ground",         # 2
##        5: "SS_obstacle",       # 0
##        6: "SS_person",         # 3
##        7: "SS_shelterbelt",    # 4
##        8: "SS_sky",            # 0
##        9: "SS_vehicle",        # 5
##        10: "SS_water",         # 6
#        0: "unknown",
#        1: "other",
#        2: "building",
#        3: "grass",
#        4: "ground",
#        5: "human",
#        6: "shelterbelt",
#        7: "water",
#    }
#    return switcher.get(argument, "Unknown")
##secondRemapping = np.array([0, 0, 0, 1, 2, 0, 3, 4, 0, 5, 6])
#secondRemapping = np.array([1, 1, 2, 3, 4, 1, 5, 6, 0, 1, 8])
#rospy.init_node('SemanticSegmentation', anonymous=True)
#nodeName = rospy.get_name()
#topicInName  = rospy.get_param(nodeName+'/topicInName', '/imageUnknown')
#topicOutNameShowResult = rospy.get_param(nodeName+'/topicOutNameShowResult', nodeName+'/imageSS')
#dirModelDescription = rospy.get_param(nodeName+'/dirModelDescription', '/detImageUnknown')
#gpuDevice = rospy.get_param(nodeName+'/gpuDevice', -1) # -1 is cpu, 0-3 is gpu 1-4
#dirModelVaules = rospy.get_param(nodeName+'/dirModelVaules', '/notDefined')
#dirTestImage = rospy.get_param(nodeName+'/dirTestImage', '/notDefined')
#dirRemapping = rospy.get_param(nodeName+'/dirRemapping', '/notDefined')
#imgDimWidth   = rospy.get_param(nodeName+'/imgDimWidth', 800)
#imgDimHeight  = rospy.get_param(nodeName+'/imgDimHeight', 600)
#
#
#pubImage = rospy.Publisher(topicOutNameShowResult, msgImage , queue_size=1)
## RETURNS launch parameters specifying if an object is set as an output 
#objectType =  list()
#for iObj in range(0,len(np.unique(secondRemapping))):
#    objectType.append(rospy.get_param(nodeName+'/objectType_'+numbers_to_strings(iObj), False))
#
#print "dirRemapping:", dirRemapping
#strParts = topicInName.split('/')
#pubImageObjs = list() 
#for iType in range(0,len(objectType)):
#    topicOutName = '/det/' + strParts[1] + nodeName + '/' + numbers_to_strings(iType)
#    pubImageObjs.append(rospy.Publisher(topicOutName, msgImage , queue_size=10))
#
##print topicOutName
#
#bridge = CvBridge()
#>>>>>>> da81e4bfedc49d71ba895f023b2ec434683ab925


def initCaffeSS(dirArchi,dirModel,dirRemapping):
    test = sio.loadmat(dirRemapping)
    classRemapping = np.concatenate((np.array([0]),np.array(test['newSorting'][:,1])),axis=0)
    
    # load net
    caffe.set_mode_cpu()
    
    net = caffe.Net(dirArchi, dirModel, caffe.TEST)
    #net = caffe.Net('fcn-32s-pascal-deploy.prototxt', 'fcn-32s-pascalcontext.caffemodel', caffe.TEST)
    
    # load image, switch to BGR, subtract mean, and make dims C x H x W for Caffe
    return net,classRemapping
    
def predictImageSS(net,im,gpuDevice):
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
    if(gpuDevice>=0):
        caffe.set_mode_gpu()
        caffe.set_device(gpuDevice)
    else:
        # load net
        caffe.set_mode_cpu()
        
    #t1 = time.clock();
    net.forward()
    #print "predictImageSS: Forward pass time:", time.clock()-t1, "seconds"


    #t1 = time.clock();
    outputMatrix = net.blobs['score'].data[0]
    out = outputMatrix.argmax(axis=0)
    #maxValues = 
    outExp = np.exp(outputMatrix)
    maxValues = outExp.max(axis=0)/outExp.sum(axis=0) 
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


#im = Image.open('/home/repete/Code/ros_workspace/src/fcn8_ros/src/Street2.jpg')
##im_gray = cv2.imread("/home/repete/Code/ros_workspace/src/fcn8_ros/src/Street2.jpg", cv2.IMREAD_GRAYSCALE)
##im_color = cv2.applyColorMap(im_gray, cv2.COLORMAP_JET)
##dirArchi = '/home/repete/Code/ros_workspace/src/fcn8_ros/models/fcn-8s-pascal-deploy.prototxt'
#dirArchi = '/home/repete/Code/ros_workspace/src/fcn8_ros/models/deploy.prototxt'
##dirModel = '/home/repete/Code/ros_workspace/src/fcn8_ros/models/fcn-8s-pascalcontext.caffemodel'
#dirModel = '/home/repete/Code/ros_workspace/src/fcn8_ros/models/pascalcontext-fcn8s-heavy.caffemodel'
#dirRemapping = "/home/repete/Code/ros_workspace/src/fcn8_ros/remappingObjectTypes.mat"
#objectType = 8
#net,classRemapping = initCaffeSS(dirArchi,dirModel,dirRemapping)
#
##secondRemapping = np.array([0, 0, 0, 1, 2, 0, 3, 4, 0, 5, 6])
##classRemappingNew = -1*np.ones(classRemapping.shape)
##for iObj in range(0,len(np.unique(secondRemapping))):
##    test = np.in1d(classRemapping, np.array(np.argwhere(secondRemapping==iObj)))
##    classRemappingNew[test] = iObj
#predictionRemapped, predictionRemappedProbability = predictImageSS(net,im,-1)
#plt.matshow(predictionRemapped)
#plt.matshow(predictionRemappedProbability)
##plt.matshow(np.array(im_color))

