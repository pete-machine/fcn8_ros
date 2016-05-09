#!/usr/bin/env python

import rospy
#import time
#from collections import namedtuple
#from std_msgs.msg import Float64MultiArray
#from std_msgs.msg import UInt16
from sensor_msgs.msg import Image as msgImage
from PIL import Image as ImagePil
import sys
sys.path.append("/usr/lib/python2.7/dist-packages")
from cv_bridge import CvBridge, CvBridgeError
import cv2 
import numpy as np
#import matplotlib.pyplot as plt
from functionsSegmanticSegmentation import initCaffeSS, predictImageSS


# 0.aeroplane, 1. bicycle, 2 bird, 3.boat, 4. bottle, 5. bus, 6. car, 7. cat, 8. chair, 9. cow, 10.diningtable
# 11. dog, 12. horse, 13. motorbike, 14. person, 15. pottedplant, 16. sheep, 17. sofa, 18. train, 19. tvmonitor, 20. pedestrian
# REMAPPING (agriculture classes)

def numbers_to_strings(argument):
    switcher = {
#        0: "SS_unknown",        # 0
#        1: "SS_animal",         # 0
#        2: "SS_building",       # 0  
#        3: "SS_field",          # 1
#        4: "SS_ground",         # 2
#        5: "SS_obstacle",       # 0
#        6: "SS_person",         # 3
#        7: "SS_shelterbelt",    # 4
#        8: "SS_sky",            # 0
#        9: "SS_vehicle",        # 5
#        10: "SS_water",         # 6
        0: "unknown",
        1: "grass",
        2: "ground",
        3: "human",
        4: "shelterbelt",
        5: "vehicle",
        6: "water",
    }
    return switcher.get(argument, "Unknown")
secondRemapping = np.array([0, 0, 0, 1, 2, 0, 3, 4, 0, 5, 6])
rospy.init_node('SemanticSegmentation', anonymous=True)
nodeName = rospy.get_name()
topicInName  = rospy.get_param(nodeName+'/topicInName', '/imageUnknown')
topicOutNameShowResult = rospy.get_param(nodeName+'/topicOutNameShowResult', nodeName+'/imageSS')
dirModelDescription = rospy.get_param(nodeName+'/dirModelDescription', '/detImageUnknown')
gpuDevice = rospy.get_param(nodeName+'/gpuDevice', -1) # -1 is cpu, 0-3 is gpu 1-4
dirModelVaules = rospy.get_param(nodeName+'/dirModelVaules', '/notDefined')
dirTestImage = rospy.get_param(nodeName+'/dirTestImage', '/notDefined')
dirRemapping = rospy.get_param(nodeName+'/dirRemapping', '/notDefined')
imgDimWidth   = rospy.get_param(nodeName+'/imgDimWidth', 800)
imgDimHeight  = rospy.get_param(nodeName+'/imgDimHeight', 600)


pubImage = rospy.Publisher(topicOutNameShowResult, msgImage , queue_size=1)
# RETURNS launch parameters specifying if an object is set as an output 
objectType =  list()
for iObj in range(0,len(np.unique(secondRemapping))):
    objectType.append(rospy.get_param(nodeName+'/objectType_'+numbers_to_strings(iObj), False))

print "dirRemapping:", dirRemapping
strParts = topicInName.split('/')
pubImageObjs = list() 
for iType in range(0,len(objectType)):
    topicOutName = '/det/' + strParts[1] + '/' + nodeName + '/' + numbers_to_strings(iType)
    pubImageObjs.append(rospy.Publisher(topicOutName, msgImage , queue_size=10))

#print topicOutName

bridge = CvBridge()


im = ImagePil.open(dirTestImage)
dirModel = dirModelVaules
dirArchi = dirModelDescription
#dirRemapping = "../remappingObjectTypes.mat"
net,classRemapping = initCaffeSS(dirArchi,dirModel,dirRemapping)

# MAKE NEW REMAPING
classRemappingNew = -1*np.ones(classRemapping.shape)
for iObj in range(0,len(np.unique(secondRemapping))):
    test = np.in1d(classRemapping, np.array(np.argwhere(secondRemapping==iObj)))
    classRemappingNew[test] = iObj
    
#image_message = bridge.cv2_to_imgmsg(predictionRemappedProbability, encoding="mono8")
#plt.matshow(predictionRemapped)
#plt.matshow(predictionRemappedProbability)

def callbackImage_received(data):
    cv_image = bridge.imgmsg_to_cv2(data, "rgb8")
    
#    cv_image = np.array(im)
    cv_image = cv2.resize(cv_image,(imgDimWidth, imgDimHeight))
#    print "ImageReceived! Image dim: ", cv_image.shape
    
    out,maxValues = predictImageSS(net,cv_image,gpuDevice)    
    msgImage_SSResult = bridge.cv2_to_imgmsg(cv2.applyColorMap(np.uint8(out*255/59), cv2.COLORMAP_JET), encoding="rgb8")
    
    pubImage.publish(msgImage_SSResult)
    for iType in range(0,len(objectType)):
        if(objectType[iType]==True):
            
            predictionRemappedProbability = np.zeros(out.shape)
            test = np.in1d(out, np.array(np.argwhere(classRemappingNew==iType)))
            predictionRemapped = np.reshape(test,(out.shape)) # True for valid classes
            predictionRemappedProbability[predictionRemapped] = maxValues[predictionRemapped]
            image_message = bridge.cv2_to_imgmsg(np.uint8(predictionRemappedProbability*255), encoding="mono8")
            image_message.header.frame_id = '/det/' + strParts[1] + nodeName + '/' + numbers_to_strings(iType)
            pubImageObjs[iType].publish(image_message)
            #print image_message.header.frame_id




# main
def main():
    print ''
    for iType in range(0,len(objectType)):
        if(objectType[iType]==True):
            print 'SemanticSegmentation  publishing:"', '/det/' + strParts[1]  + nodeName + '/' + numbers_to_strings(iType), ', receiving:"', topicInName
    
    #print(topicInName)
    #global soundhandle
    
    
    rospy.Subscriber(topicInName, msgImage, callbackImage_received,queue_size=None)    
    #rospy.Timer(rospy.Duration(timeBetweenEvaluation), EvaluateHumanAwareness)
    rospy.spin()
    rospy.Subscriber()

if __name__ == '__main__':
    main()
