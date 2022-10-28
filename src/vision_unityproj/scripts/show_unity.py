#!/usr/bin/env python
from __future__ import print_function

# other imports
import time

from cv2 import ROTATE_180
from trainer import Trainer
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

from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
# dl stuff
import torch, detectron2
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog


class Image_converter(Trainer):

  def __init__(self):
    self.model = None
    self.cfg = None
    self.setup_model()
    self.image_pub = rospy.Publisher("/myresult", Image, queue_size=2)
    self.prediction_pub = rospy.Publisher("/predictedObjects", String, queue_size=2)
    self.metadata = MetadataCatalog.get(self.trainer.cfg.DATASETS.TEST[0])
    self.bridge = CvBridge()
    self.image_sub = rospy.Subscriber("/camera_top/image_raw", Image, self.callback)
    # self.trainer = Trainer()
    # self.metadata.set(**{'person':'bla'})
    self.rename()
    



  def setup_model(self):
    self.trainer = Trainer()
    self.trainer.dataname = "unityDF"
    self.trainer.n_classes = 4   
    self.trainer.pretrained_model_weights = "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml" 
    # self.trainer.setup() // if special model
    # self.trainer.load_testdataset() // if special model
    # self.trainer.load_model() // if special model
    self.trainer.load_buildin_model()
    self.dictopub = {'bb': [], 'classes':None}
    self.current_time = time.time() 
  

  def rename(self):
    # labels 
    # bannana 52
    # apple 53
    # orange 55 
    # baseball bat 39 - pen 
    # vase 86 - bana
    # cheating with renaming, change for the real project 
    print(self.metadata)
    self.metadata.thing_classes[self.metadata.thing_classes.index("baseball bat")] = "pen"
    self.metadata.thing_classes[self.metadata.thing_classes.index("vase")] = "banana"
    self.metadata.thing_classes[self.metadata.thing_classes.index("toothbrush")] = "banana"

  
  def publish_predictions(self, predictions):
    # basically if smt changed in the scene     
    if len(self.dictopub["bb"]) != len(predictions["instances"].pred_boxes.to("cpu")):
      self.dictopub["classes"] = predictions["instances"].pred_classes.to("cpu").numpy().tolist()
      self.dictopub["bb"] = predictions["instances"].pred_boxes.to("cpu").tensor.numpy().tolist()
      # print(self.dictopub["bb"].tensor)
      self.prediction_pub.publish(json.dumps(self.dictopub))

  def callback(self, data):
    # print(self.current_time - time.time())
    if (time.time() - self.current_time  < 0.1):
      return
    else:
      self.current_time = time.time()
      
    # if self.frames < 10:

    # Convert the image from OpenCV to ROS format
    try:
      cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
    except CvBridgeError as e:
      print(f"this is error 1 {e}")   
    
    try:
      # !!!!!!!!!!!!!!! check if the detections will be moved now (probably they will) since the probelm was mirrored image 
        # grab the dimensions to calculate the center
      (h, w) = cv_image.shape[:2]
      center = (w / 2, h / 2)
      # rotate the image by 180 degrees
      M = cv2.getRotationMatrix2D(center, 180, 1.0)
      rotated = cv2.warpAffine(cv_image, M, (w, h))
      rotated = rotated[:,::-1]
      # print(self.trainer.predictor)
      prediction = self.trainer.predictor(rotated)  # format is documented at https://detectron2.readthedocs.io/tutorials/models.html#model-output-format
      # # print(prediction)
      viz = Visualizer(rotated[:, :,::-1], self.metadata, scale=1.2)
      out = viz.draw_instance_predictions(prediction["instances"].to("cpu"))
      self.publish_predictions(prediction)
      # print(prediction)
      # print(f"{h} x {w}")
      # print(out)
      # cv2.imshow("im",out.get_image())
      # # plt.show()
    except TypeError as e:  
      print(f"this is error 2 {e}")

    # Publish the image
    try:
      # out = out.get_image()[:, :, ::-1]
      self.image_pub.publish(self.bridge.cv2_to_imgmsg(
                        cv2.rotate(cv2.resize(out.get_image()[:, :, ::-1], (256,256), interpolation = cv2.INTER_AREA),ROTATE_180), "rgb8"))
    except CvBridgeError as e:
      print(f"this is error 3 {e}")

  
def main(args):
  rospy.init_node('ImageAnalizer', anonymous=True)

  ic = Image_converter()
  print("running...")
  try:
    rospy.spin()
  except KeyboardInterrupt:
    print("Shutting down")

  cv2.destroyAllWindows()

if __name__ == '__main__':
    main(sys.argv)