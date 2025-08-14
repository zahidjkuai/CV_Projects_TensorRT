from ultralytics import YOLO

try:
    model = YOLO("yolov8m-seg.pt")
    model.export(
        format="onnx",     # Export to ONNX
        opset=12,          # More stable for TensorRT
        simplify=True,     # Removes redundant nodes
        dynamic=False,     # Keep static shape to avoid TRT parsing errors
        device=0           # Use GPU instead of CPU
    )
except Exception as e:
    print(f"Model is not converted or exported as ONNX, due to error {e}")
