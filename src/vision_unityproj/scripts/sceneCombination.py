import numpy as np
import random
import rospy
from std_msgs.msg import String, Int16
import json

class Scene_Maker():

    def __init__(self) -> None:
        self.detected_classes_pub = rospy.Publisher("/detected_classes", String, queue_size=2)
        self.missing_class_pub = rospy.Publisher("/missing_class", String, queue_size=2)
        self.image_sub_rs = rospy.Subscriber("/scene_idx", Int16, self.change_idx)

        self.missing_class = []
        self.detected_classes = []
        self.make_scene()
        self.scene_idx = 0

    def make_scene(self):
        # path = "/media/raghav/m2/VRHRI_Rebecca/Assets/Scripts/rebecca_tests"
        # try: 
        #     os.mkdir(path + "/setup_folder")  
        # except FileExistsError:
        #     shutil.rmtree(path + "/setup_folder")
        #     os.mkdir(path + "/setup_folder")  
        # path += "/setup_folder"
        poses = [[0.2, 0.45], [0.04, 0.45], [-0.125, 0.45]]
        classes = ["banana", "CubeDetected", "Food_Apple_Red"]
        # start with the order in which they won't be detected
        # random.shuffle(classes)
        for i in range(len(classes)):
            # shuffle poses 
            d_classes = {}
            m_classes = {}
            # random.shuffle(poses)
            for j in range(len(classes)):
                # remove object i or rather not include it 
                if (j == i):
                    m_classes[f"{classes[j]}"] = f"{poses[j][0]}, {poses[j][1]}" # save the bb for each missing class             
                #     continue
                # save the bb for each detected class 
                d_classes[f"{classes[j]}"] = f"{poses[j][0]}, {poses[j][1]}"
            # append d_classes of scene i in the list of detected classes and missing classes
            self.detected_classes.append(d_classes)
            self.missing_class.append(m_classes)


    def publisher(self):
        # publish detected classes (dictionary) and missing classes (dictionary) json formatpri
        print("publishing...")
        while not rospy.is_shutdown():
            hello_str = "hello world %s" % rospy.get_time()
            # rospy.loginfo(hello_str)
            print(self.missing_class[self.scene_idx])
            self.detected_classes_pub.publish(json.dumps(self.detected_classes[self.scene_idx]))
            self.missing_class_pub.publish(json.dumps(self.missing_class[self.scene_idx]))
            rate = rospy.Rate(10) # 10hz
            rate.sleep()

    def change_idx(self, data):
        # change the scene index
        self.scene_idx = data.data

def main():

  rospy.init_node('SceneInitializer', anonymous=True)

  maker = Scene_Maker()
  print("running...")
  try:
    # rospy.spin()
    maker.publisher()
  except KeyboardInterrupt:
    print("Shutting down")

if __name__ == "__main__":
    main()