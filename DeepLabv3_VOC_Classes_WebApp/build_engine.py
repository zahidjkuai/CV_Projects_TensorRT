import tensorrt as trt

onnx_file = "/Tensorflow_Vision/Deeplabv3/Autonomus_Vehicle_Vision/deeplabv3_400_dynamic.onnx"
engine_file = "/Tensorflow_Vision/Deeplabv3/Autonomus_Vehicle_Vision/deeplabv3_400_dynamic.engine"

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
builder = trt.Builder(TRT_LOGGER)
network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
parser = trt.OnnxParser(network, TRT_LOGGER)

with open(onnx_file, "rb") as f:
    if not parser.parse(f.read()):
        for i in range(parser.num_errors):
            print(parser.get_error(i))
        raise RuntimeError("Failed to parse ONNX")

config = builder.create_builder_config()
config.set_flag(trt.BuilderFlag.FP16)

# Create dynamic profile
profile = builder.create_optimization_profile()
profile.set_shape("input", (1,3,200,200), (1,3,400,400), (1,3,512,512))
config.add_optimization_profile(profile)

# Build engine
serialized_engine = builder.build_serialized_network(network, config)

with open(engine_file, "wb") as f:
    f.write(serialized_engine)

print("Engine saved at", engine_file)
