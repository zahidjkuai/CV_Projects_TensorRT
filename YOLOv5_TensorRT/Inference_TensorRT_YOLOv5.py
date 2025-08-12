'''
Preprocessing libraries
Target: Inference on Images with YOLOv5 TensorRT

'''
import tensorrt as trt
import pycuda.autoinit              # pycuda gives NVIDIA's CUDA parallel computation API
import cv2
import pycuda.driver as cuda

import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt 
import yaml                         #for yaml classes file
import time                         # Time Synchronize
import logging

class TRTInferenceYOLOv5:
    def __init__(self, engine_file_path, input_shape, output_shape, classes_label_file, conf_threshold, score_threshold, nms_threshold):
        # Init TensorRT Logger
        self.logger = trt.Logger(trt.Logger.WARNING)
        # engine file
        self.engine_file_path = engine_file_path
        # load engine
        self.engine = self.load_engine(self.engine_file_path)
        # context
        self.context = self.engine.create_execution_context()
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.classes_label_file = classes_label_file
        self.conf_threshold = conf_threshold
        self.score_threshold = score_threshold
        self.nms_threshold = nms_threshold
        # load and read class file
        with open(classes_label_file, 'r') as class_read:
            data =yaml.safe_load(class_read)
            self.class_labels = [name for name in data['names'].values()]
    '''
    Load the Engine
    '''
    def load_engine(self, engine_file_path):
        with open(engine_file_path, 'rb') as f:
            runtime = trt.Runtime(self.logger)
            engine_deserialized = runtime.deserialize_cuda_engine(f.read())
            return engine_deserialized
        
    '''
    target:
    name:
    param:
    out:
    '''
    def preprocess_image(self, image_path):
        img_list = []
        img_path = []
        count = 0
        for img_original in os.listdir(image_path):
            if img_original.endswith('jpg') or img_original.endswith('jpeg') or img_original.endswith('.png'):
                img_full_path = os.path.join(image_path, img_original)
                # read Image
                self.img = cv2.imread(img_full_path)
                # retrieve original image width, height
                self.org_h, self.org_w = self.img[:2]     #640 x 640
                self.img_resized = cv2.resize(self.img, (self.input_shape[2], self.input_shape[3]), interpolation=cv2.INTER_AREA)
                img_np = np.array(self.img_resized).astype(np.float32) / 255.0
                img_np = np.transpose(img_np, (2,0,1))
                img_np = np.expand_dims(img_np, axis=0)
                #retrieve resized image width and height
                self.resized_img_h, self.resized_img_w = self.img_resized.shape[:2]
                #count images
                count += 1
                img_list.append(img_np)
                img_path.append(img_full_path)
                if count>= 12:
                    continue
        return img_list, img_path
    
    '''
    target: 
    name:
    output:
    '''
    def inference_detection(self, image_path):
        input_list, full_imgs_path = self.preprocess_image(image_path)

        self.total_time = 0
        self.num_frames = len(input_list)
        for inputs, full_imgs_path in zip(input_list, full_imgs_path):
            #start time
            self.start = time.time()
            inputs = np.ascontiguousarray(inputs)
            outputs = np.empty(self.output_shape, dtype=np.float32)
            #allocate memory on GPU
            d_inputs = cuda.mem_alloc(1*inputs.nbytes)
            d_outputs = cuda.mem_alloc(1*outputs.nbytes)
            bindings = [d_inputs, d_outputs] 
            #Transfer the data to the GPU
            cuda.memcpy_htod(d_inputs, inputs)
            #synchronize images inference
            self.context.execute_v2(bindings)
            # Transfer the data from the GPU to CPU
            cuda.memcpy_dtoh(outputs, d_outputs)
            # free memory
            d_inputs.free()
            d_outputs.free()

            #end time
            self.end = time.time()
            self.total_time = (self.end - self.start)
            self.FPS = self.num_frames / self.total_time
            #call preprocess function
            self.postprocessing_recognized_image(full_imgs_path, outputs, self.FPS)

        return outputs
    

    def postprocessing_recognized_image(self, image_path, yolov5_output, FPS):
        image = cv2.imread(image_path)
        detections = yolov5_output[0].shape[0]
        height, width = image.shape[:2]
        #rescaling
        x_scale = width / self.resized_img_w
        y_scale = height / self.resized_img_h

        conf_threshold = self.conf_threshold
        score_threshold = self.score_threshold
        nms_threshold = self.nms_threshold

        # class IDs
        class_ids = []
        #confidences 
        confidences = []
        #bounding boxes
        bboxes = []

        # YOLOv5 OUTPUT: [cx, cy, w, h, confidence, score, detection]
        # loop through detection
        for i in range(detections):
            detect = yolov5_output[0][i]
            getConf = detect[4]
            if getConf >= conf_threshold:
                class_score = detect[5:]
                class_idx = np.argmax(class_score)
                if (class_score[class_idx] > score_threshold):
                    # append confidences
                    confidences.append(getConf)
                    # class ids appends
                    class_ids.append(class_idx)
                    # get cx, cy, w, h, conf, score, class, detection
                    cx, cy, w, h = detect[0], detect[1], detect[2], detect[3]
                    #scale cx, cy, w, h to original image size
                    #cx *= x_scale
                    #cy *= y_scale
                    #w *= x_scale
                    #h *= y_scale
                    left = int((cx - w/2) * x_scale)
                    top = int((cy - h/2) * y_scale)
                    width = int(w * x_scale)
                    height = int(h * y_scale)

                    box = np.array([left, top, width, height])
                    bboxes.append(box)
        indices_nonmax = cv2.dnn.NMSBoxes(bboxes, confidences, conf_threshold, nms_threshold)
        for i in indices_nonmax:
            box = bboxes[i]
            left = box[0]
            top = box[1]
            width = box[2]
            height = box[3]

            label = "{}:{:.2f}: {:.2f}".format(self.class_labels[class_ids[i]], confidences[i], FPS)
            cv2.rectangle(image, (left, top), (left + width, top + height), (0,255,0), 3)
            cv2.putText(image, label, (left, top -20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2, cv2.LINE_AA)
        cv2.namedWindow('Detection.jpg', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Detection.jpg', 900, 800)
        cv2.imshow('Detection.jpg', image)
        cv2.waitKey(5000)
        cv2.destroyAllWindows()

def main():
    engine_file_path = "/Tensorflow_Vision/TensorRT_YOLOv5/yolov5_tensorrt_engine.engine"
    input_shape = [1, 3, 640, 640]
    '''
    1 --> batch size, 25200 --> total number of anchor predictions from all 3 layers 
    85 --> per anchor values.  0 -3 --> x_center, y_center, width, height, normalized 0-1
    '''
    output_shape = [1, 25200, 85]
    image_path = "/Tensorflow_Vision/TensorRT_YOLOv5/Images"
    path_to_classes = "/Tensorflow_Vision/TensorRT_YOLOv5/coco.yaml"

    # call the class
    inference = TRTInferenceYOLOv5(engine_file_path, input_shape, output_shape, path_to_classes, 0.4, 0.45, 0.35 )
    inference.inference_detection(image_path)

if __name__=="__main__":
    main()




            







        
        
        