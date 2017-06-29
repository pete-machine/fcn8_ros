#!/usr/bin/env python


#import time
#from collections import namedtuple
#from std_msgs.msg import Float64MultiArray
#from std_msgs.msg import UInt16

import sys
sys.path.append("/usr/lib/python2.7/dist-packages")
import cv2 
import numpy as np
import time
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
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

secondRemapping = np.array([1,1,3,4,5,1,0,6,2,1,7])
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

dirRemapping = "../remappingObjectTypes.mat"#rospy.get_param(nodeName+'/dirRemapping', '/notDefined')
dirModelDescription = "../models/deploy.prototxt"#rospy.get_param(nodeName+'/dirModelDescription', '/detImageUnknown')
gpuDevice = -1#rospy.get_param(nodeName+'/gpuDevice', -1) # -1 is cpu, 0-3 is gpu 1-4
dirModelVaules = "../models/pascalcontext-fcn8s-heavy.caffemodel" #rospy.get_param(nodeName+'/dirModelVaules', '/notDefined')
imgDimWidth = 512
imgDimHeight = 217
dirTestImage = "Street2.jpg"


# RETURNS launch parameters specifying if an object is set as an output 
#objectType = [True,True,True,True,True,True,True]
stuffType = np.array([False, False, True, True, True, True, True, True])

print "dirRemapping:", dirRemapping


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

cv_image = cv2.cvtColor(cv2.imread(dirTestImage),cv2.COLOR_BGR2RGB)


cv_image = cv2.resize(cv_image,(imgDimWidth, imgDimHeight))

t1 = time.time()
out,maxValues = predictImageSS(net,cv_image,gpuDevice)

maxValues = np.uint8(maxValues*255)


# Define remapping using a interp1d function. NICE :D
f = interp1d(np.arange(0,len(classRemappingNew)),classRemappingNew,kind='nearest')
outRemapped = f(out).astype(np.int)
for idx,iType in enumerate(classids):    
    if stuffType[idx] == True:
        mask = outRemapped==iType
        tmp = np.zeros(maxValues.shape,dtype=np.uint8)
        tmp[mask] = maxValues[mask]
    
print "Time: ", time.time()-t1, "s"

plt.figure()
plt.imshow(cv_image)

plt.figure()
plt.imshow(out)

plt.figure()
plt.imshow(outRemapped)

plt.figure()
plt.imshow(maxValues)

# Continue to image2ism
# msg.image = tmp

# Continue to 2D detections to 3D.
#msg.header = image.header
#msg.imgConfidence = maxValues
#msg.imgClass = outRemapped
#msg.crop = [0.0, 1.0, 0.0, 1.0]




