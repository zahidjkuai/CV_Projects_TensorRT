import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
import os

TRT_LOGGER = trt.Logger(trt.Logger.INFO)

def build_engine(onnx_file_path, engine_file_path, max_batch_size=1):
    """Builds a TensorRT engine from an ONNX file"""
    builder = trt.Builder(TRT_LOGGER)
    network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(network_flags)
    parser = trt.OnnxParser(network, TRT_LOGGER)

    # Parse ONNX file
    with open(onnx_file_path, 'rb') as f:
        if not parser.parse(f.read()):
            print('Failed to parse ONNX file')
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            return None

    # Builder config
    config = builder.create_builder_config()
    config.max_workspace_size = 1 << 30  # 1 GB
    if builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)

    # Create optimization profile for explicit batch input
    profile = builder.create_optimization_profile()
    input_name = network.get_input(0).name
    # Assume input shape: (batch, 3, H, W)
    profile.set_shape(input_name, min=(1,3,512,512), opt=(1,3,512,512), max=(1,3,512,512))
    config.add_optimization_profile(profile)

    # Build engine
    print("Building TensorRT engine... This may take a while.")
    engine = builder.build_engine(network, config)
    if engine is None:
        print("Failed to build the engine")
        return None

    # Save engine
    with open(engine_file_path, 'wb') as f:
        f.write(engine.serialize())
    print(f"TensorRT engine saved to {engine_file_path}")
    return engine

# Paths
onnx_model = "/Tensorflow_Vision/Deeplabv3/Mobilenetv3/segformer_ade20k.onnx"
trt_engine_file = "/Tensorflow_Vision/Deeplabv3/Mobilenetv3/segformer_ade20k_fp16.engine"

# Build engine
build_engine(onnx_model, trt_engine_file)
