# carvision_mobile.py
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse, StreamingResponse
from contextlib import asynccontextmanager
import numpy as np
import cv2
import io
import traceback
from autonomus_vehicle_vision import DeepLabTRT

ENGINE_PATH = "/Tensorflow_Vision/Deeplabv3/Autonomus_Vehicle_Vision/deeplabv3_400_dynamic.engine"

# Lifespan context to initialize and release engine
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("[INFO] Initializing DeepLabv3 TRT engine...")
    app.state.seg_engine = DeepLabTRT(engine_path=ENGINE_PATH)
    yield
    print("[INFO] Releasing GPU resources...")
    app.state.seg_engine.release()

app = FastAPI(title="DeepLabv3 Mobile Real-Time Segmentation", lifespan=lifespan)

# Mobile HTML frontend
@app.get("/")
async def index():
    return HTMLResponse("""
    <html>
      <body>
        <h2>DeepLabv3 Mobile Real-Time Segmentation</h2>
        <video id="video" autoplay playsinline width="400"></video>
        <canvas id="canvas" style="display:none;"></canvas>
        <img id="output" width="400"/>
        <div>FPS: <span id="fps">0</span></div>
        <script>
          const video = document.getElementById('video');
          const canvas = document.getElementById('canvas');
          const output = document.getElementById('output');
          const fpsEl = document.getElementById('fps');
          const ctx = canvas.getContext('2d');

          // Try rear camera first
          navigator.mediaDevices.getUserMedia({ video: { facingMode: { exact: "environment" } } })
            .then(stream => { video.srcObject = stream; })
            .catch(err => {
              console.warn("Rear camera unavailable, fallback to default.", err);
              navigator.mediaDevices.getUserMedia({ video: true }).then(stream => { video.srcObject = stream; });
            });

          let lastTime = performance.now();

          async function sendFrameLoop() {
            if(video.videoWidth === 0) {
              requestAnimationFrame(sendFrameLoop);
              return;
            }

            // Resize frame to 400 px width for faster inference
            const targetWidth = 400;
            const targetHeight = Math.round(video.videoHeight * (targetWidth / video.videoWidth));
            canvas.width = targetWidth;
            canvas.height = targetHeight;
            ctx.drawImage(video, 0, 0, targetWidth, targetHeight);

            canvas.toBlob(async (blob) => {
              try {
                const formData = new FormData();
                formData.append("file", blob, "frame.jpg");
                const response = await fetch("/segment", { method: "POST", body: formData });
                const blobOut = await response.blob();
                output.src = URL.createObjectURL(blobOut);

                // Compute FPS
                const now = performance.now();
                const fps = Math.round(1000 / (now - lastTime));
                lastTime = now;
                fpsEl.innerText = fps;

                // Next frame
                requestAnimationFrame(sendFrameLoop);

              } catch(e) {
                console.error("Segmentation error:", e);
                requestAnimationFrame(sendFrameLoop);
              }
            }, "image/jpeg");
          }

          sendFrameLoop();
        </script>
      </body>
    </html>
    """)

# Segmentation endpoint
@app.post("/segment")
async def segment(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        np_arr = np.frombuffer(contents, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if frame is None:
            return {"error": "Invalid image file"}

        # Run inference
        output_frame = app.state.seg_engine.infer_frame(frame)

        # Encode output image
        success, img_encoded = cv2.imencode('.jpg', output_frame)
        if not success:
            return {"error": "Failed to encode output image"}

        return StreamingResponse(io.BytesIO(img_encoded.tobytes()), media_type="image/jpeg")

    except Exception as e:
        traceback.print_exc()
        return {"error": str(e)}
