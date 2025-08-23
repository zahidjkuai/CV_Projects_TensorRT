from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse, StreamingResponse
import cv2
import numpy as np
import io
from YOLOv5_Web_APP_Video_Inference import YOLOv5TrTVideo

app = FastAPI()

# Initialize YOLOv5 TensorRT model
engine_file_path = "/Tensorflow_Vision/TensorRT_YOLOv5/yolov5_tensorrt_engine.engine"
input_shape = (1, 3, 640, 640)
output_shape = (1, 25200, 85)
path_to_classes = "/Tensorflow_Vision/TensorRT_YOLOv5/coco.yaml"
conf_threshold = 0.4
score_threshold = 0.45
nms_threshold = 0.35

model = YOLOv5TrTVideo(
    engine_file_path, input_shape, output_shape, path_to_classes,
    conf_threshold, score_threshold, nms_threshold
)

# HTML page served to mobile
@app.get("/")
async def index():
    return HTMLResponse("""
    <html>
      <body>
        <h2>YOLOv5 Mobile Detection</h2>
        <video id="video" autoplay playsinline width="400"></video>
        <canvas id="canvas" style="display:none;"></canvas>
        <img id="output" width="400"/>
        <script>
          const video = document.getElementById('video');
          const canvas = document.getElementById('canvas');
          const output = document.getElementById('output');
          const ctx = canvas.getContext('2d');

          // Request back (environment) camera first
          navigator.mediaDevices.getUserMedia({ video: { facingMode: { exact: "environment" } } })
            .then(stream => { video.srcObject = stream; })
            .catch(err => {
              console.error("Could not access back camera, falling back to default", err);
              // fallback to any available camera
              navigator.mediaDevices.getUserMedia({ video: true })
                .then(stream => { video.srcObject = stream; });
            });

          async function sendFrame() {
            if(video.videoWidth === 0) return;

            canvas.width = video.videoWidth;
            canvas.height = video.videoHeight;
            ctx.drawImage(video, 0, 0);

            canvas.toBlob(async (blob) => {
              const formData = new FormData();
              formData.append("file", blob, "frame.jpg");
              const response = await fetch("/detect", {
                method: "POST",
                body: formData
              });
              const blobOut = await response.blob();
              output.src = URL.createObjectURL(blobOut);
            }, "image/jpeg");
          }

          setInterval(sendFrame, 200); // 5 FPS
        </script>
      </body>
    </html>
    """)

# Endpoint for detecting objects from mobile camera
@app.post("/detect")
async def detect(file: UploadFile = File(...)):
    contents = await file.read()
    np_arr = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    # Run YOLOv5 inference
    output_frame = model.inference_on_frame(frame)
    output_frame = np.asarray(output_frame)

    success, img_encoded = cv2.imencode('.jpg', output_frame)
    if not success:
        raise RuntimeError("cv2.imencode failed!")

    return StreamingResponse(io.BytesIO(img_encoded.tobytes()), media_type="image/jpeg")
