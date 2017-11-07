#!/usr/bin/env python

import rospy
#import time
#from collections import namedtuple
#from std_msgs.msg import Float64MultiArray
#from std_msgs.msg import UInt16
from sensor_msgs.msg import Image as msgImage
#import sys
#sys.path.append("/usr/lib/python2.7/dist-packages")
from cv_bridge import CvBridge

import cv2 
import numpy as np

from boundingbox_msgs.msg import ImageDetections

COLORS = cv2.applyColorMap(np.uint8(np.array([0,1,2,3,4,5,6,7])*255.0/10.0), cv2.COLORMAP_JET)

rospy.init_node('ros_visualize_fcn', anonymous=True)
nodeName = rospy.get_name()
topicInName  = rospy.get_param(nodeName+'/topicInName', '/imageUnknown')
outputTopic = rospy.get_param(nodeName+'/outputTopic', nodeName+'/imageSS')

pubDetectionImage = rospy.Publisher(outputTopic, msgImage , queue_size=10)

bridge = CvBridge()


def callbackImage_received(data):
    
    out = bridge.imgmsg_to_cv2(data.imgClass, desired_encoding="passthrough")
    print out.shape
    out_visualized = cv2.applyColorMap(np.uint8(out*255.0/10.0), cv2.COLORMAP_JET)
    print out_visualized.shape
    msgImage_SSResult = bridge.cv2_to_imgmsg(out_visualized, encoding="rgb8")
    pubDetectionImage.publish(msgImage_SSResult)
    


# main
def main():
    print ''    
    rospy.Subscriber(topicInName, ImageDetections, callbackImage_received,queue_size=None)    
    
    rospy.spin()
#    rospy.Subscriber()

if __name__ == '__main__':
    main()
