import tensorrt as trt 
import pycuda.autoinit
import cv2
import pycuda.driver as cuda
import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as  plt 
import yaml
import time

class YOLOv5TrTVideo:
    def __init__(self, engine_file_path, input_shape, output_shape, classes_label_file, conf_threshold, score_threshold, nms_threshold):
        # error warning while engine loading
        self.logger = trt.Logger(trt.Logger.WARNING)
        
        #initialize parameters
        self.engine_file_path = engine_file_path
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.classes_label_files = classes_label_file
        self.conf_threshold = conf_threshold
        self.score_threshold = score_threshold
        self.nms_threshold = nms_threshold
        self.prev_time = time.time()
        self.FPS = 0.0 

        #load engine
        self.engine = self.load_engine(self.engine_file_path)
        self.context = self.engine.create_execution_context()

        # read the yaml class labels
        with open(classes_label_file, 'r') as class_read:
            data = yaml.safe_load(class_read)
            self.class_labels = [name for name in data['names'].values()]

    '''
    Loading the engine file and deserialize for inference
    '''
    def load_engine(self, engine_file_path):
        with open(engine_file_path, 'rb') as f:
            runtime = trt.Runtime(self.logger)
            engine_deserialized = runtime.deserialize_cuda_engine(f.read())
        return engine_deserialized
    
    '''
    Target: pre-processing the video frames
    '''
    def preprocess_video(self, video_path):
        video = cv2.VideoCapture(video_path)
        if not video.isOpened():
            print("Failed to Open Video")

        #while opens do pre-processing
        while video.isOpened():
            ret, frame = video.read()
            frame = cv2.resize(frame, (1000, 800))
            if not ret:
                break
            self.org_frame_h, self.org_frame_w = frame.shape[:2]

            # normalization for inference
            img_resized = cv2.resize(frame, (self.input_shape[2], self.input_shape[3]), interpolation=cv2.INTER_AREA)
            self.resized_frame_h, self.resized_frame_w = img_resized.shape[:2]
            #convert to numpy and divide it by float32 and 255.0
            img_np = np.array(img_resized).astype(np.float32) / 255.0
            img_np = np.transpose(img_np, (2,0,1))
            yield img_np, frame
        video.release()

    '''
    video path, frame, outputs
    '''

    def inference_detection(self, video_path):
        self.total_time = 0
        self.num_frames = 0
        outputs = None
        for inputs, frame in self.preprocess_video(video_path):
            self.num_frames += 1
            self.start = time.time()
            inputs = np.ascontiguousarray(inputs)
            outputs = np.empty(self.output_shape, dtype=np.float32)

            d_inputs = cuda.mem_alloc(1*inputs.nbytes)
            d_outputs = cuda.mem_alloc(1 * outputs.nbytes)
            bindings = [d_inputs, d_outputs]
            cuda.memcpy_htod(d_inputs, inputs)
            self.context.execute_v2(bindings)
            cuda.memcpy_dtoh(outputs, d_outputs)

            d_inputs.free()
            d_outputs.free()
            
            #end time
            self.end = time.time()
            self.total_time += (self.end - self.start)
            self.FPS = self.num_frames / self.total_time
            # post processing gpu results
            self.postprocessing_recognized_frames(frame, outputs)
        return outputs
    
    '''
    postprocessing frame and output gpu results
    '''
    def inference_on_frame(self, frame):
        start = time.time()
        # Store original frame dimensions
        self.org_frame_h, self.org_frame_w = frame.shape[:2]

        # Resize for model input
        img_resized = cv2.resize(frame, (self.input_shape[2], self.input_shape[3]))
        self.resized_frame_h, self.resized_frame_w = img_resized.shape[:2]

        # Normalize and reshape
        img_np = np.array(img_resized).astype(np.float32) / 255.0
        img_np = np.transpose(img_np, (2, 0, 1))   # CHW
        img_np = np.expand_dims(img_np, axis=0)    # NCHW
        inputs = np.ascontiguousarray(img_np)
        outputs = np.empty(self.output_shape, dtype=np.float32)

        # Allocate device memory
        d_inputs = cuda.mem_alloc(inputs.nbytes)
        d_outputs = cuda.mem_alloc(outputs.nbytes)
        bindings = [int(d_inputs), int(d_outputs)]

        # Copy to device
        cuda.memcpy_htod(d_inputs, inputs)
        self.context.execute_v2(bindings)
        cuda.memcpy_dtoh(outputs, d_outputs)

        # Free device memory
        d_inputs.free()
        d_outputs.free()

        # Postprocess detections on original frame
        frame_out = self.postprocessing_recognized_frames(frame, outputs)
        end = time.time()
        self.FPS = 1.0 / (end -start + 1e-06)
        return np.asarray(frame_out)

    def postprocessing_recognized_frames(self, frame, yolov5_output):
        detections = yolov5_output[0].shape[0]
        height, width = frame.shape[:2]
        x_scale = self.org_frame_w / self.resized_frame_w
        y_scale = self.org_frame_h / self.resized_frame_h
        
        conf_threshold= self.conf_threshold
        score_threshold = self.score_threshold
        nms_threshold = self.nms_threshold
        class_ids = []
        confidences = []
        bboxes = []

        for i in range(detections):
            detect = yolov5_output[0][i]
            getConf = detect[4]
            if getConf >= conf_threshold:
                class_score = detect[5:]
                class_idx = np.argmax(class_score)
                if (class_score[class_idx] > score_threshold):
                    confidences.append(getConf)
                    class_ids.append(class_idx)

                    #yolov5 output frames
                    cx, cy, w, h = detect[0], detect[1], detect[2], detect[3]
                    left = int((cx - w / 2) * x_scale)
                    top = int((cy - h / 2) * y_scale)                  
                    width =  int(w * x_scale)
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
              
            label = "{}:{:.2f}, FPS: {:.2f}".format(self.class_labels[class_ids[i]], confidences[i], self.FPS)
            cv2.rectangle(frame, (left, top), (left + width, top+height), (0, 255, 0), 3)
            '''
            (0, 255, 0)       # Green
            (0, 0, 255)       # Red
            (255, 0, 0)       # Blue
            (0, 255, 255)     # Yellow
            (255, 0, 255)     # Pink/Magenta
            (255, 255, 0)     # Cyan
            (0, 128, 255)     # Orange
            (147, 20, 255)    # Purple
            '''
            cv2.putText(frame, label, (left, top-20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (147, 20, 255), 2, cv2.LINE_AA)
        return frame

            

def main():
    engine_file_path = "/Tensorflow_Vision/TensorRT_YOLOv5/yolov5_tensorrt_engine.engine"
    input_shape = (1, 3, 640, 640)
    output_shape = (1, 25200, 85)
    video_path = "/Tensorflow_Vision/TensorRT_YOLOv5/Videos/vdinf4.mp4"
    path_to_classes = "/Tensorflow_Vision/TensorRT_YOLOv5/coco.yaml"
    inference = YOLOv5TrTVideo(engine_file_path, input_shape, output_shape, path_to_classes, 0.4, 0.45, 0.35)
    inference.inference_detection(video_path)

if __name__ == "__main__":
    main()

