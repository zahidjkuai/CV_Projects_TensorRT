import numpy as np
import cv2
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit  # initialize CUDA context

class DeepLabTRT:
    """
    DeepLabv3 TensorRT engine for semantic segmentation.
    Optimized for smaller input size (e.g., 360) for real-time mobile usage.
    Uses VOC Classes
    """
    # Generate a unique color for each COCO class
    COCO_COLORS = np.array([
        [  0,  0, 255], [  0,128,255], [  0,255,255], [  0,255,128], [  0,255,  0], [128,255,  0], [255,255,  0], [255,128,  0],
        [255,  0,  0], [255,  0,128], [255,  0,255], [128,  0,255], [  0,  0,128], [  0,128,128], [  0,128,  0], [128,128,  0],
        [128,128,128], [128,  0,128], [ 64,  0,128], [ 64,128,128], [128, 64,128], [128, 64, 64], [ 64,128,  0], [192,128,  0],
        [192, 64,  0], [192, 64,128], [ 64, 64,192], [128,128,192], [192,128,192], [192,128,128], [64,192,128], [128,192,128],
        [192,192,128], [192,192,192], [64,128,192], [128,64,192], [192,64,192], [64,192,192], [128,192,192], [192,128,192],
        [255,192,128], [255,128,192], [192,255,128], [128,255,192], [255,255,128], [128,255,255], [255,128,255], [255,255,255],
        [64,0,64], [128,0,64], [192,0,64], [64,128,0], [128,128,0], [192,128,0], [64,192,0], [128,192,0], [192,192,0],
        [64,0,128], [128,0,128], [192,0,128], [64,128,128], [128,128,128], [192,128,128], [64,192,128], [128,192,128],
        [192,192,128], [64,0,192], [128,0,192], [192,0,192], [64,128,192], [128,128,192], [192,128,192], [64,192,192],
        [128,192,192], [192,192,192], [32,32,32], [64,64,64], [96,96,96], [128,128,128], [160,160,160], [192,192,192], [224,224,224], [255,255,255]
    ], dtype=np.uint8)

    def __init__(self, engine_path, input_size=400, num_classes=80, alpha=0.5):
        self.input_size = int(input_size)
        self.num_classes = int(num_classes)
        self.alpha = float(alpha)
        self.palette = DeepLabTRT.COCO_COLORS[:num_classes]

        # TensorRT runtime & engine
        self.logger = trt.Logger(trt.Logger.ERROR)
        self.runtime = trt.Runtime(self.logger)
        with open(engine_path, "rb") as f:
            engine_bytes = f.read()
        self.engine = self.runtime.deserialize_cuda_engine(engine_bytes)
        if self.engine is None:
            raise RuntimeError(f"Failed to load engine: {engine_path}")
        self.context = self.engine.create_execution_context()

        # Bindings
        self.input_idx = next(i for i in range(self.engine.num_bindings) if self.engine.binding_is_input(i))
        self.output_idx = 1 - self.input_idx

        # Handle dynamic input shapes
        in_shape = list(self.engine.get_binding_shape(self.input_idx))
        if any(d == -1 for d in in_shape):
            in_shape = [1, 3, self.input_size, self.input_size]
            self.context.set_binding_shape(self.input_idx, tuple(in_shape))

        # Final shapes
        self.in_shape  = tuple(self.context.get_binding_shape(self.input_idx))
        self.out_shape = tuple(self.context.get_binding_shape(self.output_idx))

        # Allocate memory
        self.in_bytes  = int(np.prod(self.in_shape)  * np.dtype(np.float32).itemsize)
        self.out_bytes = int(np.prod(self.out_shape) * np.dtype(np.float32).itemsize)
        self.d_in  = cuda.mem_alloc(self.in_bytes)
        self.d_out = cuda.mem_alloc(self.out_bytes)
        self.stream = cuda.Stream()

        # Host pinned memory
        self.h_in  = cuda.pagelocked_empty(self.in_shape,  dtype=np.float32, order='C')
        self.h_out = cuda.pagelocked_empty(self.out_shape, dtype=np.float32, order='C')

        # Bindings list
        self.bindings = [0] * self.engine.num_bindings
        self.bindings[self.input_idx]  = int(self.d_in)
        self.bindings[self.output_idx] = int(self.d_out)

        # ImageNet normalization
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def preprocess(self, frame):
        img = cv2.resize(frame, (self.input_size, self.input_size))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        img = (img - self.mean) / self.std
        img = np.transpose(img, (2, 0, 1))[None, ...].astype(np.float32)
        return np.ascontiguousarray(img, dtype=np.float32)

    def postprocess(self, frame, logits):
        mask = logits.argmax(axis=1)[0].astype(np.uint8)
        mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)
        color_mask = self.palette[mask]
        return cv2.addWeighted(frame, 1 - self.alpha, color_mask, self.alpha, 0)

    def infer_frame(self, frame):
        x = self.preprocess(frame)
        np.copyto(self.h_in, x)
        cuda.memcpy_htod_async(self.d_in, self.h_in, self.stream)
        self.context.execute_async_v2(self.bindings, self.stream.handle)
        cuda.memcpy_dtoh_async(self.h_out, self.d_out, self.stream)
        self.stream.synchronize()
        y = np.asarray(self.h_out).reshape(self.out_shape)
        return self.postprocess(frame, y)

    def release(self):
        try:
            self.d_in.free()
            self.d_out.free()
        except Exception:
            pass
        self.stream = None
