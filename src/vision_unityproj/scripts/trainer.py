import torch, detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
import os, json, cv2, random, sys

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2.engine import DefaultTrainer
from detectron2.utils.visualizer import ColorMode
import matplotlib.pylab as plt 


torch.cuda.empty_cache()

class Trainer:

    def __init__(self) -> None:
        self.dataunity = None
        self.unity_metadata = None
        self.pretrained_model_weights = None 
        self.dataname = "" # data set name 
        self.n_classes = 1
        self.testdataname = ""
        self.test_data = ""
        self.test_metadata = ""
        self.predictor = None
        self.cfg = None 
        
    def prepare_data(self, path:str):
        register_coco_instances(self.dataname, {}, f"{path}/labels.json", path)
        dataunity = DatasetCatalog.get(self.dataname)
        unity_metadata = MetadataCatalog.get(self.dataname)

    def show_sample_data(self, train = True):
        # detectron style outputs 
        # dataset_dicts = dataunity
        for d in random.sample(self.dataunity, 5):
            img = cv2.imread(d["file_name"])
            visualizer = Visualizer(img[:, :, ::-1], metadata=self.unity_metadata, scale=0.5)
            out = visualizer.draw_dataset_dict(d)
            cv2.imshow(out.get_image()[:, :, ::-1])
            cv2.waitKey(0)

    def setup(self):
        self.cfg = get_cfg()
        if self.pretrained_model_weights is not None:
            
            self.cfg.merge_from_file(model_zoo.get_config_file(self.pretrained_model_weights))
            self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(self.pretrained_model_weights)  # Let training initialize from model zoo

        self.cfg.DATASETS.TRAIN = (self.dataname,)
        self.cfg.DATASETS.TEST = ()
        self.cfg.DATALOADER.NUM_WORKERS = 1
        self.cfg.SOLVER.IMS_PER_BATCH = 2  # This is the real "batch size" commonly known to deep learning people
        self.cfg.SOLVER.BASE_LR = 0.0000105  # pick a good LR
        self.cfg.SOLVER.MAX_ITER = 100000   # 300 iterations seems good enough for this toy dataset; you will need to train longer for a practical dataset
        self.cfg.SOLVER.STEPS = []        # do not decay learning rate
        self.cfg.SOLVER.MOMENTUM = 0.92
        self.cfg.MODEL.ROI_HEADS.NUM_CLASSES = self.n_classes
        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.8 # set a custom testing threshold



    def train(self, continue_training:bool = False):
        os.makedirs(self.cfg.OUTPUT_DIR, exist_ok=True)
        trainer = DefaultTrainer(self.cfg) 
        trainer.resume_or_load(resume=continue_training)
        trainer.train()

        # BATCH_SIZE = 32
        # self.cfg.BATCH_SIZE_PER_IMAGE = BATCH_SIZE
        # self.cfg.MODEL.BATCH_SIZE_PER_IMAGE = BATCH_SIZE
        # self.cfg.MODEL.FPN.BATCH_SIZE_PER_IMAGE = BATCH_SIZE
        # self.cfg.MODEL.RPN.BATCH_SIZE_PER_IMAGE = BATCH_SIZE
        # self.cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = BATCH_SIZE
        # self.cfg.DATALOADER.BATCH_SIZE_PER_IMAGE = BATCH_SIZE
        # self.cfg.SOLVER.BATCH_SIZE_PER_IMAGE = BATCH_SIZE
        # # 
        # self.cfg.MODEL.RPN.PRE_NMS_TOPK_TRAIN = 10
        # self.cfg.MODEL.RPN.POST_NMS_TOPK_TRAIN = 10
        # self.cfg.DATASETS.PRECOMPUTED_PROPOSAL_TOPK_TRAIN = 10
        # self.cfg.INPUT.MAX_SIZE_TRAIN = 32
        # self.cfg.INPUT.MAX_SIZE_TEST = 32
        # self.cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 2   # The "RoIHead batch size". 128 is faster, and good enough for this toy dataset (default: 512)
        #### some additional model settings 
        # self.cfg.MODEL.FPN.FPN_ON = True
        # self.cfg.MODEL.FPN.MULTILEVEL_RPN = True
        # self.cfg.MODEL.FPN.MULTILEVEL_ROIS = True
        # self.cfg.MODEL.FPN.ROI_MIN_LEVEL = 2
        # self.cfg.MODEL.FPN.ROI_MAX_LEVEL = 5
        # self.cfg.MODEL.FPN.RPN_MIN_LEVEL = 2
        # self.cfg.MODEL.FPN.RPN_MAX_LEVEL = 8
        # self.cfg.MODEL.FPN.COARSEST_STRIDE = 256
        # self.cfg.MODEL.FPN.SCALES_PER_OCTAVE = 3
        # self.cfg.MODEL.FPN.ANCHOR_SCALE = 2
        # self.cfg.MODEL.FPN.RPN_ANCHOR_START_SIZE = 5
        # self.cfg.MODEL.FPN.RPN_ASPECT_RATIOS = (0.5, 1, 2)

    def load_model(self):
        # MODEL_NAME = "model_0014999.pth" 
        MODEL_NAME = "model_0099999.pth"
        self.cfg.MODEL.WEIGHTS = os.path.join(self.cfg.OUTPUT_DIR, MODEL_NAME)  # path to the model we just trained
        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.97 # set a custom testing threshold
        self.predictor = DefaultPredictor(self.cfg)

    def load_testdataset(self):
        path = "/home/raghav/code/UnityRobotics/Unity-Robotics-Hub/tutorials/pick_and_place/captures/show"
        self.testdataname = f"{random.randint(0,10000)}"
        register_coco_instances(f"{self.testdataname}", {}, f"{path}/labels.json", f"{path}")
        self.test_data = DatasetCatalog.get(f"{self.testdataname}")
        self.test_metadata = MetadataCatalog.get(f"{self.testdataname}")
        self.cfg.DATASETS.TEST = (self.testdataname,)

    def check_predictions(self):
        k = random.sample(self.test_data, 6)
        print([i['file_name'] for i in k])
        fig,ax = plt.subplots(nrows=2, ncols=3)
        fig.set_size_inches(20,12)
        rows = 0 
        cols = 0
        for d in k:    
            img = cv2.imread(d["file_name"])
            prediction = self.predictor(img)  # format is documented at https://detectron2.readthedocs.io/tutorials/models.html#model-output-format
            viz = Visualizer(img[:, :, ::-1], MetadataCatalog.get(self.cfg.DATASETS.TEST[0]), scale=1.2)
            out = viz.draw_instance_predictions(prediction["instances"].to("cpu"))
            print(prediction)
            ax[rows, cols].imshow(out.get_image())
            cols+=1
            if cols > 2:
                rows+=1
                cols = 0
        plt.show()

    def load_buildin_model(self):
        self.cfg = get_cfg()
        # add project-specific config (e.g., TensorMask) here if you're not running a model in detectron2's core library
        self.cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_DC5_1x.yaml"))
        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
        # Find a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well
        self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_DC5_1x.yaml")
        self.predictor = DefaultPredictor(self.cfg)
          


def main():
    if len(sys.argv) <= 1:
        print("idk what to do")
        return 
    trainer = Trainer()
    trainer.dataname = "unityDF"
    trainer.n_classes = 4   
    trainer.pretrained_model_weights = "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml" 
    trainer.setup()
    if sys.argv[1] == "train":
        trainer.prepare_data("/home/raghav/code/UnityRobotics/Unity-Robotics-Hub/tutorials/pick_and_place/captures/training")
        trainer.train()
    elif sys.argv[1] == "test":
        trainer.load_model()
        trainer.load_testdataset()
        trainer.check_predictions()
    

if __name__ == "__main__":
    main()