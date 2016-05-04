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
        0: "YoloAeroplane",
        1: "YoloBicycle",
        2: "YoloBird",
        3: "YoloBoat",
        4: "YoloBottle",
        5: "YoloBus",
        6: "YoloCar",
        7: "YoloCat",
        8: "YoloChair",
        9: "YoloCow",
        10: "YoloDiningtable",
        11: "YoloDog",
        12: "YoloHorse",
        13: "YoloMotorbike",
        14: "YoloPerson",
        15: "YoloPottedplant",
        16: "YoloSheep",
        17: "YoloSofa",
        18: "YoloTrain",
        19: "YoloTvMonitor",
        20: "SS_unknown",
        21: "SS_animal",
        22: "SS_building",
        23: "SS_field",
        24: "SS_ground",
        25: "SS_obstacle",
        26: "SS_person",
        27: "SS_shelterbelt",
        28: "SS_sky",
        29: "SS_vehicle",
        30: "SS_water",
        31: "PD_Pedestrian",
        32: "thermal",
    }
    return switcher.get(argument, "Unknown")


rospy.init_node('SemanticSegmentation', anonymous=True)
nodeName = rospy.get_name()
topicInName  = rospy.get_param(nodeName+'/topicInName', '/imageUnknown')
topicOutName = rospy.get_param(nodeName+'/topicOutName', '/detImageUnknown')
dirModelDescription = rospy.get_param(nodeName+'/dirModelDescription', '/detImageUnknown')
dirModelVaules = rospy.get_param(nodeName+'/dirModelVaules', '/notDefined')
dirTestImage = rospy.get_param(nodeName+'/dirTestImage', '/notDefined')
dirRemapping = rospy.get_param(nodeName+'/dirRemapping', '/notDefined')
imgDimWidth   = rospy.get_param(nodeName+'/imgDimWidth', 800)
imgDimHeight  = rospy.get_param(nodeName+'/imgDimHeight', 600)

# RETURNS launch parameters specifying if an object is set as an output 
objectType =  np.array([False, False, False, False, False, False, False, False, False, False, False])
for iObj in range(0,len(objectType)):
    objectType[iObj] = rospy.get_param(nodeName+'/objectType_'+numbers_to_strings(iObj+20), False)     

#objectType[1] = rospy.get_param(nodeName+'/objectTypeAnimal', False) # 
#objectType[2] = rospy.get_param(nodeName+'/objectTypeBuilding', False) # 
#objectType[3] = rospy.get_param(nodeName+'/objectTypeField', False) # 
#objectType[4] = rospy.get_param(nodeName+'/objectTypeGround', False) # 
#objectType[5] = rospy.get_param(nodeName+'/objectTypeObstacle', False) #
#objectType[6] = rospy.get_param(nodeName+'/objectTypePerson', False) # 
#objectType[7] = rospy.get_param(nodeName+'/objectTypeShelterbelt', False) #
#objectType[8] = rospy.get_param(nodeName+'/objectTypeSky', False) # 
#objectType[9] = rospy.get_param(nodeName+'/objectTypeVehicle', False) # 
#objectType[10] = rospy.get_param(nodeName+'/objectTypeWater', False) # 

print "dirRemapping:", dirRemapping

pubImageObjs = list() 
for iType in range(0,len(objectType)):
    topicOutName = '/detectionImage' + numbers_to_strings(iType+20)
    pubImageObjs.append(rospy.Publisher(topicOutName, msgImage , queue_size=1))

#print topicOutName

bridge = CvBridge()


im = ImagePil.open(dirTestImage)
dirModel = dirModelVaules
dirArchi = dirModelDescription
#dirRemapping = "../remappingObjectTypes.mat"
net,classRemapping = initCaffeSS(dirArchi,dirModel,dirRemapping)

#image_message = bridge.cv2_to_imgmsg(predictionRemappedProbability, encoding="mono8")
#plt.matshow(predictionRemapped)
#plt.matshow(predictionRemappedProbability)


def callbackImage_received(data):
    #blank_image = np.zeros((imgDimHeight,imgDimWidth,1), np.uint8)
    
    #t1 = time.clock()
    cv_image = bridge.imgmsg_to_cv2(data, "rgb8")
    
    #t2 = time.clock()
    cv_image = cv2.resize(cv_image,(imgDimWidth, imgDimHeight))
    print "ImageReceived! Image dim: ", cv_image.shape
    #t3 = time.clock()
    #print "ImageShape: ",  cv_image.shape
    out,maxValues = predictImageSS(net,cv_image,classRemapping)
    
    
    for iType in range(0,len(objectType)):
        if(objectType[iType]==True):
            predictionRemappedProbability = np.zeros(out.shape)
            test = np.in1d(out, np.array(np.argwhere(classRemapping==objectType)))
            predictionRemapped = np.reshape(test,(out.shape)) # True for valid classes
            predictionRemappedProbability[predictionRemapped] = maxValues[predictionRemapped]
            #t4 = time.clock()
            image_message = bridge.cv2_to_imgmsg(np.uint8(predictionRemappedProbability*255), encoding="mono8")
            #t5 = time.clock()
            pubImageObjs[iType].publish(image_message)
            #t6 = time.clock()
    #print "predictImageSS: Post forward pass time", time.clock()-t1, "seconds"
    
    #redictionRemappedProbability = cv_image 
    #print predictionRemappedProbability
    
#    print "Image processed in: ", time.clock()-t1, "s"
#    print "     bridge  in: ", t2-t1, "s"
#    print "     resize image: ", t3-t2, "s"
#    print "     forward pass (in predictImageSS): ", t4-t3, "s"
#    print "     bridge out: ", t5-t4, "s"
#    print "     publish: ", t6-t5, "s"



# main
def main():
    print ''
    for iType in range(0,len(objectType)):
        if(objectType[iType]==True):
            print 'SemanticSegmentation  publishing:"', '/detectionImage' + numbers_to_strings(iType+20), ', receiving:"', topicInName
    
    #print(topicInName)
    #global soundhandle
    
    
    rospy.Subscriber(topicInName, msgImage, callbackImage_received)    
    #rospy.Timer(rospy.Duration(timeBetweenEvaluation), EvaluateHumanAwareness)
    rospy.spin()


if __name__ == '__main__':
    main()
