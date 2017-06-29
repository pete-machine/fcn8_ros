#!/usr/bin/env python

import rospy
#import time
#from collections import namedtuple
#from std_msgs.msg import Float64MultiArray
#from std_msgs.msg import UInt16
from sensor_msgs.msg import Image as msgImage
import os
import sys
sys.path.append("/usr/lib/python2.7/dist-packages")
from cv_bridge import CvBridge, CvBridgeError
import time
import cv2 
import numpy as np
from scipy.interpolate import interp1d
from boundingbox_msgs.msg import ImageDetections
#import matplotlib.pyplot as plt
from semantic_segmentation import initCaffeSS, predictImageSS

#secondRemapping = np.array([0, 0, 0, 1, 2, 0, 3, 4, 0, 5, 6])
#0: "unknown",
#1: "grass",
#2: "ground",
#3: "human",
#4: "shelterbelt",
#5: "vehicle",
#6: "water",
secondRemapping = np.array([1,1,3,4,5,1,0,6,2,1,7])

def numbers_to_strings(argument):
    switcher = {
        0: "human",
        1: "other",
        2: "unknown",
        3: "building",
        4: "grass",
        5: "ground",
        6: "shelterbelt",
        7: "water"
    }
    return switcher.get(argument, "undefined")

# Define the number of classes. 
classids = np.unique(secondRemapping)
nClasses = len(classids)

# Must contain as many values as classes
stuffType = np.array([False, False, True, True, True, True, True, True])

classids = np.unique(secondRemapping)
nClasses = len(classids)

# 0: "SS_unknown",        # 1
# 1: "SS_animal",         # 1
# 2: "SS_building",       # 3 
# 3: "SS_field",          # 4
# 4: "SS_ground",         # 5
# 5: "SS_obstacle",       # 1
# 6: "SS_person",         # 0
# 7: "SS_shelterbelt",    # 6
# 8: "SS_sky",            # 2
# 9: "SS_vehicle",        # 1
# 10: "SS_water",         # 7


rospy.init_node('SemanticSegmentation', anonymous=True)
nodeName = rospy.get_name()
topicInName  = rospy.get_param(nodeName+'/topicInName', '/imageUnknown')
outputTopicPrefix = rospy.get_param(nodeName+'/outputTopicPrefix', nodeName+'/imageSS')

visualize = rospy.get_param(nodeName+'/visualize', False)
topicOutVisualize = rospy.get_param(nodeName+'/topicOutVisualize', '/visualizeTopic')

dirModelDescription = rospy.get_param(nodeName+'/dirModelDescription', '/detImageUnknown')
gpuDevice = rospy.get_param(nodeName+'/gpuDevice', -1) # -1 is cpu, 0-3 is gpu 1-4
dirModelVaules = rospy.get_param(nodeName+'/dirModelVaules', '/notDefined')

dirRemapping = rospy.get_param(nodeName+'/dirRemapping', '/notDefined')
imgDimWidth   = rospy.get_param(nodeName+'/imgDimWidth', 800)
imgDimHeight  = rospy.get_param(nodeName+'/imgDimHeight', 600)

if visualize:
    pubImage = rospy.Publisher(topicOutVisualize, msgImage , queue_size=1)
pubDetectionImage = rospy.Publisher(os.path.join(outputTopicPrefix,'detection_image'), ImageDetections , queue_size=10)

print "dirRemapping:", dirRemapping
strParts = topicInName.split('/')
pubImageObjs = {}
for idx, iType in enumerate(classids):
    if stuffType[idx] == True:
        topicOutName = os.path.join(outputTopicPrefix,numbers_to_strings(iType))
        pubImageObjs[numbers_to_strings(iType)] = rospy.Publisher(topicOutName, msgImage , queue_size=10)

#print topicOutName

bridge = CvBridge()


dirModel = dirModelVaules
dirArchi = dirModelDescription
#dirRemapping = "../remappingObjectTypes.mat"
net,classRemapping = initCaffeSS(dirArchi,dirModel,dirRemapping)
inProcessing = False
# MAKE NEW REMAPING
classRemappingNew = -1*np.ones(classRemapping.shape)
for iObj in range(0,len(np.unique(secondRemapping))):
    test = np.in1d(classRemapping, np.array(np.argwhere(secondRemapping==iObj)))
    classRemappingNew[test] = iObj
    
#image_message = bridge.cv2_to_imgmsg(predictionRemappedProbability, encoding="mono8")
#plt.matshow(predictionRemapped)
#plt.matshow(predictionRemappedProbability)

def callbackImage_received(data):
    global inProcessing
    if(inProcessing==False): 
        t1 = time.time()
        inProcessing = True
        cv_image = bridge.imgmsg_to_cv2(data, "rgb8")
    
#    cv_image = np.array(im)
        cv_image = cv2.resize(cv_image,(imgDimWidth, imgDimHeight))
        print "ImageReceived! Image dim: ", cv_image.shape
    
        out,maxValues = predictImageSS(net,cv_image,gpuDevice)  
#        print "Image predicted: out.shape", out.shape, "maxValues.shape",maxValues.shape

        if visualize:
            msgImage_SSResult = bridge.cv2_to_imgmsg(cv2.applyColorMap(np.uint8(out*255.0/59.0), cv2.COLORMAP_JET), encoding="rgb8")
            pubImage.publish(msgImage_SSResult)
        
        maxValues = np.uint8(maxValues*255)
        # Define remapping using a interp1d function. NICE :D
        f = interp1d(np.arange(0,len(classRemappingNew)),classRemappingNew,kind='nearest')
        outRemapped = f(out).astype(np.int)
        
        # For all stuff classes: Publish detection images used in image2ism. 
        for idx,iType in enumerate(classids):    
            if stuffType[idx] == True:
                mask = outRemapped==iType
                tmp = np.zeros(maxValues.shape,dtype=np.uint8)
                tmp[mask] = maxValues[mask]
                
                className = numbers_to_strings(iType)
                image_message = bridge.cv2_to_imgmsg(tmp, encoding="mono8")
                image_message.header = data.header
                image_message.header.frame_id = os.path.join(outputTopicPrefix,className)
                #pubImageObjs[iType].publish(image_message)
                pubImageObjs[className].publish(image_message)
        
        # Detections to be converted for to first 2D bounding boxes and later 3D.
        msg = ImageDetections()
        msg.header = data.header
        msg.imgConfidence = bridge.cv2_to_imgmsg(maxValues, encoding="mono8")
        msg.imgClass = bridge.cv2_to_imgmsg(outRemapped.astype(np.uint8), encoding="mono8")
        msg.crop = [0.0, 1.0, 0.0, 1.0]
        
        pubDetectionImage.publish(msg)
        
#        for iType in range(0,len(objectType)):
#            
#            predictionRemappedProbability = np.zeros(out.shape)
#            test = np.in1d(out, np.array(np.argwhere(classRemappingNew==iType)))
#            predictionRemapped = np.reshape(test,(out.shape)) # True for valid classes
#            predictionRemappedProbability[predictionRemapped] = maxValues[predictionRemapped]
#            occMap = np.uint8(predictionRemappedProbability*255)
#            
#            image_message = bridge.cv2_to_imgmsg(occMap, encoding="mono8")
#            image_message.header = data.header
#            image_message.header.frame_id = '/det/' + strParts[1] + nodeName + '/' + numbers_to_strings(iType)
#            pubImageObjs[iType].publish(image_message)
        print "ImageProcessed in: ", time.time()-t1, "s"
        inProcessing = False




# main
def main():
    print ''
    for idx,iType in enumerate(classids):
        if stuffType[idx] == True:
            print 'SemanticSegmentation (forwarded to image2ism) publishing:"', os.path.join(outputTopicPrefix, numbers_to_strings(iType)), ', receiving:"', topicInName
        else:
            print 'SemanticSegmentation (forwarded to eventually bb2ism) publishing (' + numbers_to_strings(iType) + '):', os.path.join(outputTopicPrefix,'detectionImage'), ', receiving:"', topicInName
    
    #print(topicInName)
    #global soundhandle
    
    
    rospy.Subscriber(topicInName, msgImage, callbackImage_received,queue_size=None)    
    #rospy.Timer(rospy.Duration(timeBetweenEvaluation), EvaluateHumanAwareness)
    rospy.spin()
    rospy.Subscriber()

if __name__ == '__main__':
    main()
