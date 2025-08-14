import cv2
import sys
from pathlib import Path
import numpy as np
import argparse
import os
from ultralytics import YOLO

base_dir = Path("/Tensorflow_Vision/YOLOv8_semantic_segmentation")
sys.path.append(base_dir.as_posix())

class YOLOv8SingleShot:
    def __init__(self, model_path, images_path, output_directory):
        self.model_path = model_path
        self.images_path = images_path
        self.output_directory = output_directory

        if not os.path.exists(self.output_directory):
            os.makedirs(self.output_directory, exist_ok=True)


    #class method init
    def inference(self):
        for images in os.listdir(self.images_path):
            img = os.path.join(self.images_path, images)
            img = cv2.imread(img)

            self.model = YOLO("/Tensorflow_Vision/YOLOv8_semantic_segmentation/algorithms/yolov8m-seg.onnx")

            results = self.model.predict(
                source = str(self.images_path),         #take all images from this dir
                conf = 0.25, 
                save = True, 
                project = str(self.output_directory),
                name = "batch_results",                 # save predicted images to this dir
                exist_ok=True
                #name=self.output_directory
                )
           



#class INIT
def main():
    parser = argparse.ArgumentParser(description='YOLOv8 Single Shot Inference')
    parser.add_argument('--model_path', type=str, default=str(base_dir/ "algorithms/yolov8m-seg.onnx"))
    parser.add_argument('--images_path', type=str, required=True)
    parser.add_argument('--output_directory', type=str, required=True)

    args = parser.parse_args()
    yolov8 = YOLOv8SingleShot(args.model_path, args.images_path, args.output_directory)
    yolov8.inference()

if __name__ == '__main__':
    main()