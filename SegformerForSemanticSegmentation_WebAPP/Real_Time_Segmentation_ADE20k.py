import os
import time
import numpy as np
import cv2
try:
    import tensorrt as trt
    import pycuda.driver as cuda
    import pycuda.autoinit
    TRT_AVAILABLE = True
except Exception as e:
    print("[WARN] TensorRT/PyCUDA not available; using dummy mode:", e)
    TRT_AVAILABLE = False

class RealTimeSegmentationTrT:
    """TensorRT wrapper for ADE20K segmentation (512x512) real-time video"""
    def __init__(self, engine_path: str, input_hw=(512, 512), num_classes=150):
        self.input_hw = input_hw
        self.num_classes = num_classes
        self.color_lut = self._make_palette(num_classes)
        self.trt_ok = False

        if TRT_AVAILABLE and os.path.isfile(engine_path):
            try:
                self._logger = trt.Logger(trt.Logger.ERROR)
                with open(engine_path, "rb") as f:
                    runtime = trt.Runtime(self._logger)
                    self.engine = runtime.deserialize_cuda_engine(f.read())
                self.context = self.engine.create_execution_context()

                # Allocate buffers for video stream
                self.inputs, self.outputs, self.bindings, self.stream = [], [], [], cuda.Stream()
                for binding in self.engine:
                    shape = self.engine.get_binding_shape(binding)
                    size = trt.volume(shape)
                    dtype = trt.nptype(self.engine.get_binding_dtype(binding))
                    host_mem = cuda.pagelocked_empty(size, dtype)
                    dev_mem = cuda.mem_alloc(host_mem.nbytes)
                    self.bindings.append(int(dev_mem))
                    if self.engine.binding_is_input(binding):
                        self.inputs.append({'host': host_mem, 'device': dev_mem, 'shape': shape})
                    else:
                        self.outputs.append({'host': host_mem, 'device': dev_mem, 'shape': shape})

                self.trt_ok = True
                print("[INFO] TensorRT engine loaded and buffers allocated.")
            except Exception as e:
                print("[WARN] Failed to load engine; using dummy mode:", e)

    @staticmethod
    def _make_palette(num_classes: int) -> np.ndarray:
        rng = np.random.RandomState(42)
        colors = rng.randint(0, 255, size=(num_classes, 3), dtype=np.uint8)
        colors[0] = np.array([0, 0, 0], dtype=np.uint8) 
        return colors

    def preprocess(self, bgr: np.ndarray) -> np.ndarray:
        h, w = self.input_hw
        img = cv2.resize(bgr, (w, h), interpolation=cv2.INTER_LINEAR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))  # HWC -> CHW
        return np.expand_dims(img, 0).copy()  # add batch dim

    def postprocess(self, logits: np.ndarray) -> np.ndarray:
        # Dynamically reshape logits according to engine output shape
        output_shape = self.outputs[0]['shape']  # (1, 150, 512, 512)
        logits = logits.reshape(output_shape)
        if logits.ndim == 4:  # batch dim
            logits = logits[0]
        seg = np.argmax(logits, axis=0).astype(np.uint8)
        return seg

    def visualize(self, bgr: np.ndarray, seg: np.ndarray, alpha=0.45) -> np.ndarray:
        seg_rgb = self.color_lut[seg]
        seg_rgb = cv2.resize(seg_rgb, (bgr.shape[1], bgr.shape[0]), interpolation=cv2.INTER_NEAREST)
        overlay = cv2.addWeighted(bgr, 1 - alpha, seg_rgb, alpha, 0)
        return overlay

    def infer(self, bgr: np.ndarray) -> tuple:
        t0 = time.time()

        if self.trt_ok:
            # Preprocess
            pre = self.preprocess(bgr).ravel()
            np.copyto(self.inputs[0]['host'], pre)
            cuda.memcpy_htod_async(self.inputs[0]['device'], self.inputs[0]['host'], self.stream)

            # Inference
            self.context.execute_async_v2(bindings=self.bindings, stream_handle=self.stream.handle)
            cuda.memcpy_dtoh_async(self.outputs[0]['host'], self.outputs[0]['device'], self.stream)
            self.stream.synchronize()

            #  Get raw output
            output = self.outputs[0]['host']

            # 4. Infer actual output size
            c = self.num_classes
            hw = output.size // c
            h_out = int(hw ** 0.5)   # e.g. 128
            w_out = h_out
            logits = output.reshape(c, h_out, w_out)

            # 5. Upsample logits back to input size (512×512)
            if (h_out, w_out) != self.input_hw:
                logits = cv2.resize(
                    logits.transpose(1, 2, 0),   # (H, W, C)
                    self.input_hw,               # target (512, 512)
                    interpolation=cv2.INTER_LINEAR
                ).transpose(2, 0, 1)              # back to (C, H, W)

            # Segmentation mask
            seg = np.argmax(logits, axis=0).astype(np.uint8)

            # Overlay
            overlay = self.visualize(bgr, seg)

        else:
            # Fallback if TRT not available
            gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 80, 160)
            seg_rgb = np.zeros_like(bgr)
            seg_rgb[edges > 0] = (0, 255, 0)
            overlay = cv2.addWeighted(bgr, 0.6, seg_rgb, 0.4, 0)

        # Latency + FPS
        latency = time.time() - t0
        fps = 1.0 / max(latency, 1e-6)
        latency_ms = latency * 1000.0

        return overlay, fps, latency_ms
