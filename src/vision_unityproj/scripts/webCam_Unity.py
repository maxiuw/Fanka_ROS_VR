#!/usr/bin/env python
from __future__ import print_function

# other imports
import time
import matplotlib.pylab as plt 
import numpy as np 
import cv2
import json
# ros
import roslib
import sys
import rospy
from std_msgs.msg import String, Float32
# from diagnostic_msgs.msg import KeyValue

from sensor_msgs.msg import Image, CompressedImage
from cv_bridge import CvBridge, CvBridgeError


def publisher():
    webCamPub = rospy.Publisher(
            "plan_publisher",
            CompressedImage,
            queue_size=30,
        )
    rospy.init_node('webCam', anonymous=True)
    rate = rospy.Rate(30)
    bridge = CvBridge()
    # bridge.encoding_as_cvtype2('8UC3')

    cam = cv2.VideoCapture(0)
    while not rospy.is_shutdown():
        _, img = cam.read()
        # cv_image = bridge.imgmsg_to_cv2(img, "bgr8")
        # br.(im)
        webCamPub.publish(bridge.cv2_to_compressed_imgmsg(img)) # out.get_image()[:, :, ::-1] ,'bgr8'
        rate.sleep()

if __name__ == '__main__':
    
    try:
        publisher()
    except rospy.ROSInterruptException:
        pass

