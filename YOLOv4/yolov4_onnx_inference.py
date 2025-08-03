'''
Python packages:
os
cv2
time
target: YOLOV4 DNN inference on images
'''
import os
import cv2
import time

'''
class components
class name: Yolov4DNN
class init components: nms_threshold, confidence_threshold, image_path, yolov4: [path_to_cfg_yolo, path_to_weights]
'''
class Yolov4DNN:       #0.3            #0.38
    def __init__(self, nms_threshold, conf_threshold, class_labels, image_path, path_to_cfg_yolo, path_to_weights):
        self.nms_threshold = nms_threshold #non maximum suppression threshold
        self.conf_threshold = conf_threshold
        self.class_labels = class_labels
        self.image_path = image_path
        self.path_to_cfg_yolo = path_to_cfg_yolo    #path to configuration
        self.path_to_weights = path_to_weights      #path to weights

        # Load the class labels from txt file
        with open(class_labels, 'r') as read_class:
            self.class_labels = [classes.strip() for classes in read_class.readlines()]

        #frame image
        self.frames = self.load_images(self.image_path)

        #preprocess images and resize
        for self.frame in self.frames:
            self.image = cv2.imread(self.frame)
            # get height and width of the images
            self.original_h, self.original_w = self.image.shape[:2]
            dimension = (640, 640)
            self.resize_image = cv2.resize(self.image, dimension, interpolation=cv2.INTER_AREA)

            #get new height and width of the resized images
            self.new_h, self.new_w = self.resize_image.shape[:2]
            #call function
            self.inference_run(self.resize_image)

            

        '''
        Function Target: Load Images
        param[1]: self
        param[2]: image_path
        '''
    def load_images(self, image_path):
        #list of images
        img_list = []
        for img_original in os.listdir(image_path):
            if img_original.endswith('.jpg') or img_original.endswith('.png') or img_original.endswith('.jpeg'):
                img_full_path = os.path.join(image_path, img_original)
                img_list.append(img_full_path)
        return img_list
    
    '''
    target: inference DNN OpenCV with onnx
    param[1]: path to yolov4 configuration
    param[2]: path to yolov4 weights
    '''
    def inference_dnn(self, path_to_cfg_yolo, path_to_weights):
        # read dnn of yolov4
        network = cv2.dnn.readNet(path_to_cfg_yolo, path_to_weights)
        # Try to use GPU (CUDA)
        network.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        network.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)


        #creates net from file with trained weights and configuration
        model = cv2.dnn_DetectionModel(network)

        # set model parameters#
        model.setInputParams(size=(640, 640), scale=1/255.0, swapRB=True)

        '''
        classIDs: class index in n result detection
        [out] confidences A set of corresponding confidences.
        [out] boxes A set of corresponding bounding boxes.
        '''
        classes, scores, boxes = model.detect(self.image, self.conf_threshold, self.nms_threshold)
        return classes, scores, boxes
    '''
    target: Inference Run and Draw Bounding Boxes
    param[1]: image
    '''
    def inference_run(self,image):
        #start
        start = time.time()
        # get classes, scores, boxes
        # inference for every frame
        getClasses, getScores, getBoxes = self.inference_dnn(self.path_to_cfg_yolo, self.path_to_weights)  
        end = time.time()

        #Frame time
        frame_time = (end - start) * 1000
        #frame per second
        FPS = 1.0 * (end - start)
        '''
        CALCULATE new scale of image which is image formed between original and resized image
        '''
        ratio_h = self.new_h / self.original_h
        ratio_w = self.new_w / self.original_w
        for (class_id, score, box) in zip(getClasses, getScores, getBoxes):
            #normalize the box
            box[0] = int(box[0] * ratio_w) # x
            box[1] = int(box[1] * ratio_h) # y
            box[2] = int(box[2] * ratio_w) #width
            box[3] = int(box[3] * ratio_h) # height
            cv2.rectangle(image, box, color=(0, 255, 0), thickness=2) #rgb code color
            #string label
            #label = "Frame time: %.2f ms, FPS: %.2f, ID: %s, Score: %.2f" % (frame_time, FPS, self.class_labels[class_id], score)
            label = "%s: %.2f" % (self.class_labels[class_id], score) #only class id and score
            cv2.putText(image, label, (box[0] -30, box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
        cv2.imshow("Image Detected", image)
        cv2.waitKey(10000) # Display the image for 10 seconds.... 1000 = 1 second
        cv2.destroyAllWindows()

def main():
    #open the coco classes txt file
    path_to_classes ="/Tensorflow_Vision/onnx_yolov4/coco-classes.txt"
    image_path = "/Tensorflow_Vision/onnx_yolov4/Images"
    path_to_cfg_yolo = "/Tensorflow_Vision/onnx_yolov4/yolov4.cfg"
    path_to_weights = "/Tensorflow_Vision/onnx_yolov4/yolov4.weights"
    
    Yolov4DNN(0.3, 0.38, path_to_classes, image_path, path_to_cfg_yolo, path_to_weights)
if __name__ == "__main__":
    main()



        

            



