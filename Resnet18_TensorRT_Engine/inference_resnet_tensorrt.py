import tensorrt as trt
import pycuda.autoinit
import pycuda.driver as cuda
import cv2
import numpy as np 
import os
from PIL import Image
from matplotlib import pyplot as plt

'''
Class name: TensorRTInference
INIT: Self.logger
params [1]: engine_path, context
params [2]: input_shape, output_shape, class_labels
'''
class TRTInference:
    def __init__(self, engine_file_path, class_labels_file):
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.engine_file_path = engine_file_path
        #load engine
        self.runtime, self.engine = self.load_engine(self.engine_file_path)

        #init context
        self.context = self.engine.create_execution_context()
        
        #self.input_shape = input_shape
        #self.output_shape = output_shape
        self.class_labels_file = class_labels_file
        #open class file
        with open(self.class_labels_file, 'r') as class_read:
            self.class_labels = [line.strip() for line in class_read.readlines()]

    def load_engine(self, engine_file_path):
        with open(self.engine_file_path, 'rb') as f:
            engine_data = f.read()
            runtime = trt.Runtime(self.logger)
            engine_deserialized = runtime.deserialize_cuda_engine(engine_data)
        return runtime, engine_deserialized

    '''
    param [1]: image_path
    results (return): img_list, img_path
    '''
    def preprocess_img(self, image_path):
        img_list = []

        img_path = []
        target_height = 224
        target_width = 224

        for img_original in os.listdir(image_path):
            if img_original.endswith('.jpg') or img_original.endswith('.png') or img_original.endswith('.jpeg'):
                img_full_path = os.path.join(image_path, img_original)

                #open image and convert to RGB
                image = Image.open(img_full_path).convert('RGB')
                image = image.resize((target_width, target_height), Image.BILINEAR)
                #convert to numpy array
                img_np = np.array(image).astype(np.float32) /255.0
                #change from HWC to CHW format
                img_np = np.transpose(img_np, (2, 0, 1))
                img_np = np.expand_dims(img_np, axis=0)
                img_list.append(img_np)
                img_path.append(img_full_path)
        
        return img_list, img_path

    def postprocess_img(self, outputs):
        #classes
        classes_indices = []

        for output in outputs:
            class_idx = output.argmax()
            print("Class Detected: ", self.class_labels[class_idx])
            classes_indices.append(self.class_labels[class_idx])
        return classes_indices

    #Inference Detection
    def inference_detection(self, image_path):
        # list
        input_list, full_img_paths = self.preprocess_img(image_path)
        results = []

        for inputs, full_img_path in zip(input_list, full_img_paths):
            #allocate memory
            inputs = np.ascontiguousarray(inputs)
            inpute_name = self.engine.get_tensor_name(0)
            self.context.set_input_shape(inpute_name, inputs.shape)

            batch_size = inputs.shape[0]
            output_shape = (batch_size, 1000) #Resnet18 has 1000 classes
            outputs = np.empty(output_shape, dtype=np.float32)

            d_inputs = cuda.mem_alloc( inputs.nbytes)
            d_outputs = cuda.mem_alloc(outputs.nbytes)
            bindings = [int(d_inputs), int(d_outputs)]
            
            #transfer input to GPU
            cuda.memcpy_htod(d_inputs, inputs)
            #synchronize context
            self.context.execute_v2(bindings)
            # copy outoputs back to host
            cuda.memcpy_dtoh(outputs, d_outputs)

            result = self.postprocess_img(outputs)

            # free the memory
            d_inputs.free()
            d_outputs.free()

            #Results
            results.append(result)

            #Display Results
            self.display_recognized_images(full_img_path, result)

        return results

    '''
    param[0] : image_path
    param[1] : class_label
    target: Display and saving detected images
    '''

    def display_recognized_images(self, image_path, class_label):
        image = Image.open(image_path)
        for class_name in class_label:
            #create one directory for detected images
            path_to_detected_imags = "/Tensorflow_Vision/Resnet18/images_detected"

            #check path existance
            if not os.path.exists(path_to_detected_imags):
                os.makedirs(path_to_detected_imags)
            plt.imshow(image)
            plt.title(f"Recognized Image: {class_name}")
            plt.axis('off')

            # Save the image with detected class label
            save_img = os.path.join(path_to_detected_imags, f"{class_name}.jpg") 
            plt.savefig(save_img)
            plt.close()
            return image, save_img
        
engine_file_path = "/Tensorflow_Vision/Resnet18/Resnet_engine/resnet18_dynamic.engine"
img_file_path = "/Tensorflow_Vision/Resnet18/images_original"
 
class_labels = "/Tensorflow_Vision/Resnet18/imagenet-classes.txt"
path_to_original_images = "/Tensorflow_Vision/Resnet18/images_original"

inference = TRTInference(engine_file_path, class_labels)
inference.inference_detection(path_to_original_images)






