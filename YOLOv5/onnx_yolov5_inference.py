import os
import numpy as np
import cv2
import time

'''
target: Making Inference with YOLOv5 onnx model on set of images
'''
class DetectYOLOv5Onnx:
    def __init__(self, imgs_path, model_path, imgs_width, imgs_height, conf_threshold, score_threshold, nms_threshold, classes_path):
        self.imgs_path = imgs_path
        self.model_path = model_path
        self.imgs_width = imgs_width
        self.imgs_height = imgs_height
        self.conf_threshold = conf_threshold
        self.score_threshold = score_threshold
        self.nms_threshold = nms_threshold
        self.classes_path = classes_path
    
    #pre-built call function 
    #target: load images and onnx model
    def __call__(self):
        for img in os.listdir(self.imgs_path):
            if img.endswith('.jpg') or img.endswith('.png') or img.endswith('.jpeg'):
                #full path of images
                imgs_path = os.path.join(self.imgs_path, img)
                # read images
                image = cv2.imread(imgs_path)
                #load onnx model
                network = cv2.dnn.readNetFromONNX(self.model_path)
                #GPU
                network.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
                #Confit GPU
                network.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
                #read classes
                classes = self.class_name()
                #detection function called
                self.detection(image, network, classes)
            
        pass

    # read class labels ... output == [class list]
    def class_name(self):
        #list of classes
        classes = []
        file = open(self.classes_path, 'r') 
        while True:
            name = file.readline().strip('\n') #read line by line

            classes.append(name)
            if not name:
                break

        return classes
    
    '''
    target: To make Inference and Show Image
    name: detection
    param[1]: image
    param[2]: network
    param[3]: classes
    '''
    def detection(self,img,net, classes):
        #blob to apply in input image
        #out[]: Return 4-dim Matrix with NCHW dimension order 
        blob = cv2.dnn.blobFromImage(img, 1/255.0, (640,640), swapRB=True, mean=(0,0), crop =False)
        #set to input model
        net.setInput(blob)
        #start time
        t1 = time.time()
        # unconnected layers by index
        output_layers = net.getUnconnectedOutLayersNames()
        outputs = net.forward(output_layers)
        t2 = time.time()
        print('OpenCV DNN YOLOv5 Inference Time: ', t2-t1)

        #number of detections
        n_detections = outputs[0].shape[1]
        height,width = img.shape[:2]
        #scale
        x_scale = width / self.imgs_width
        y_scale = height / self.imgs_height

        confidence_threshold = self.conf_threshold
        score_threshold = self.score_threshold
        nms_threshold = self.nms_threshold
        # list of class ids
        class_ids = []
        confidences = []
        bboxes = []

        # Loop through detections
        for i in range(n_detections):
            detect = outputs[0][0][i]
            confidence = detect[4]
            if confidence >= confidence_threshold:
                class_score = detect[5:]
                class_id = np.argmax(class_score)
                if (class_score[class_id] > score_threshold):
                    confidences.append(confidence)
                    class_ids.append(class_id)
                    cx, cy, w, h = detect[0], detect[1], detect[2], detect[3]
                    #calculate bounding box coordinates
                    left = int((cx -w/2)*x_scale)
                    top = int((cy -h/2) *y_scale)
                    width = int(w * x_scale)
                    height = int(h *y_scale)
                    box = np.array([left, top, width, height])
                    bboxes.append(box)
            # non max suppression
        indices = cv2.dnn.NMSBoxes(bboxes, confidences, confidence_threshold, nms_threshold)
        for i in indices:
            box = bboxes[i]
            left = box[0]
            top = box[1]
            width = box[2]
            height = box[3]

            # Label
            label = "{}:{:.2f}".format(classes[class_ids[i]], confidences[i])
            #rectangle
            cv2.rectangle(img, (left, top), (left+width, top+height), (0,255,0), 3)
            cv2.putText(img, label, (left, top+20), cv2.FONT_HERSHEY_COMPLEX, 0.7, (0,255,255), 2, cv2.LINE_AA)
        cv2.namedWindow('detection.jpg', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('detection.jpg', 900, 800)
        cv2.imshow('detection.jpg', img)
        cv2.waitKey(10000) 
         
         
        cv2.destroyAllWindows()

def main():
    imgs_path = "/Tensorflow_Vision/onnx_yolov4/Images"
    onnx_path = "/Tensorflow_Vision/onnx_yolov5/yolov5s.onnx"
    classes_path = "/Tensorflow_Vision/onnx_yolov5/coco-classes.txt"
    instance = DetectYOLOv5Onnx(imgs_path, onnx_path, 640, 640, 0.34, 0.38, 0.3, classes_path)
    instance()


if __name__=="__main__":
    main()

  
    

