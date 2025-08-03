'''
Target: TensorRT conversion
model: Resnet18
Image Resolution: 1,3,224,224
'''
import tensorrt as trt
class TensorRTConversion:
    '''
    path: to onnx
    path: to engine
    maxworkspace: < 1 GB
    precision: float16
    inference mode: Dynamic Batch [1,10,20]
    '''
    def __init__(self, path_to_onnx, path_to_engine, max_workspace_size=1 << 30, half_precision=False):

        self.TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
        self.path_to_onnx = path_to_onnx
        self.path_to_engine = path_to_engine
        self.max_workspace_size = max_workspace_size
        self.half_precision = half_precision
       
    '''
    INIT BUILD
    INIT ONFIG
    INIT EXPLICIT BATCH
    INIT NETROWK
    '''

    def convert(self):
        builder = trt.Builder(self.TRT_LOGGER)
        config = builder.create_builder_config()
        config.max_workspace_size = self.max_workspace_size

        explicit_batch = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        network = builder.create_network(explicit_batch)
        parser = trt.OnnxParser(network, self.TRT_LOGGER)

        with open(self.path_to_onnx, 'rb') as model_onnx:
            if not parser.parse(model_onnx.read()):
                print('ERROR: Failed to parse the ONNX Model.')
                for error in parser.errors:
                    print(error)
                return None
            
        #set profile for explicit batch
        #profile = builder.create_optimization_profile()
        #profile.set_shape('input_name', min(1,3,224,224), opt=(10,3,224,224), max=(20,3,224,224))
        #config.add_optimization_profile(profile)
        print("Successfully TensorRT Engine Configuired to Maximum Batch")
        print("\n")

        if builder.platform_has_fast_fp16:
            config.set_flag(trt.BuilderFlag.FP16)
        engine = builder.build_engine(network, config)
        with open(self.path_to_engine, 'wb') as f_engine:
            f_engine.write(engine.serialize())
        print("Successfully Converted ONNX to TensorRT Dynamic Engine")
        print(f'Serialized Engine saved in engine path: {self.path_to_engine}')
        
convert = TensorRTConversion('/Tensorflow_Vision/Resnet18/resnet18.onnx', '/Tensorflow_Vision/Resnet18/resnet18.engine')

#call class method
convert.convert()
    
