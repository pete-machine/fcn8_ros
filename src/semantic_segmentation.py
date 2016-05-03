#!/usr/bin/env python

import rospy
import time
from collections import namedtuple
from std_msgs.msg import Float64MultiArray
from std_msgs.msg import UInt16
from sensor_msgs.msg import Image as msgImage
from PIL import Image as ImagePil
import sys
sys.path.append("/usr/lib/python2.7/dist-packages")
from cv_bridge import CvBridge, CvBridgeError
import cv2 
import numpy as np
import matplotlib.pyplot as plt
from functionsSegmanticSegmentation import initCaffeSS, predictImageSS


# 0.aeroplane, 1. bicycle, 2 bird, 3.boat, 4. bottle, 5. bus, 6. car, 7. cat, 8. chair, 9. cow, 10.diningtable
# 11. dog, 12. horse, 13. motorbike, 14. person, 15. pottedplant, 16. sheep, 17. sofa, 18. train, 19. tvmonitor, 20. pedestrian
# REMAPPING (agriculture classes)

topicInName  = rospy.get_param('/semantic_segmentation/topicInName', '/imageUnknown')
topicOutName = rospy.get_param('/semantic_segmentation/topicOutName', '/detImageUnknown')
dirModelDescription = rospy.get_param('/semantic_segmentation/dirModelDescription', '/detImageUnknown')
dirModelVaules = rospy.get_param('/semantic_segmentation/dirModelVaules', '/notDefined')
dirTestImage = rospy.get_param('/semantic_segmentation/dirTestImage', '/notDefined')
dirRemapping = rospy.get_param('/semantic_segmentation/dirRemapping', '/notDefined')
objectTypeInt = rospy.get_param('/semantic_segmentation/objectTypeInt', 1000) # 1000 is not specified. 0-19 is pascal classes. 20 is the pedestrian detector
imgDimWidth   = rospy.get_param('/semantic_segmentation/imgDimWidth', 800)
imgDimHeight  = rospy.get_param('/semantic_segmentation/imgDimHeight', 600)

print "dirRemapping:", dirRemapping

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
    }
    return switcher.get(argument, "Unknown")

topicOutName = '/detectionImage' + numbers_to_strings(objectTypeInt)

#print topicOutName
pubImage = rospy.Publisher(topicOutName, msgImage , queue_size=1)
bridge = CvBridge()


im = ImagePil.open(dirTestImage)
dirModel = dirModelVaules
dirArchi = dirModelDescription
#dirRemapping = "../remappingObjectTypes.mat"
objectType = objectTypeInt-20 # -20 to shift to the object classes in the SS functions.
net,classRemapping = initCaffeSS(dirArchi,dirModel,dirRemapping)

#image_message = bridge.cv2_to_imgmsg(predictionRemappedProbability, encoding="mono8")
#plt.matshow(predictionRemapped)
#plt.matshow(predictionRemappedProbability)


def callbackImage_received(data):
    #blank_image = np.zeros((imgDimHeight,imgDimWidth,1), np.uint8)
    
    t1 = time.clock()
    cv_image = bridge.imgmsg_to_cv2(data, "rgb8")
    
    t2 = time.clock()
    cv_image = cv2.resize(cv_image,(imgDimWidth, imgDimHeight))
    print "ImageReceived! Image dim: ", cv_image.shape
    t3 = time.clock()
    #print "ImageShape: ",  cv_image.shape
    predictionRemappedProbability = predictImageSS(net,cv_image,objectType,classRemapping)
    
    
    #redictionRemappedProbability = cv_image 
    #print predictionRemappedProbability
    t4 = time.clock()
    image_message = bridge.cv2_to_imgmsg(np.uint8(predictionRemappedProbability*255), encoding="mono8")
    t5 = time.clock()
    pubImage.publish(image_message)
    t6 = time.clock()
    print "Image processed in: ", time.clock()-t1, "s"
    print "     bridge  in: ", t2-t1, "s"
    print "     resize image: ", t3-t2, "s"
    print "     forward pass (in predictImageSS): ", t4-t3, "s"
    print "     bridge out: ", t5-t4, "s"
    print "     publish: ", t6-t5, "s"



# main
def main():
    print ''
    print 'SemanticSegmentation  publishing:"', topicOutName, ', receiving:"', topicInName
    #print(topicInName)
    #global soundhandle
    rospy.init_node('SemanticSegmentation', anonymous=True)
    
    rospy.Subscriber(topicInName, msgImage, callbackImage_received)    
    #rospy.Timer(rospy.Duration(timeBetweenEvaluation), EvaluateHumanAwareness)
    rospy.spin()


if __name__ == '__main__':
    main()
