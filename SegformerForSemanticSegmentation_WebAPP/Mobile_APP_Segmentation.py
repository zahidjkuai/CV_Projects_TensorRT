# ade20k_mobile.py
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
from contextlib import asynccontextmanager
import numpy as np
import cv2
import io
import traceback
from Real_Time_Segmentation_ADE20k import RealTimeSegmentationTrT
import base64

ENGINE_PATH = "/Tensorflow_Vision/Deeplabv3/Mobilenetv3/segformer_ade20k_fp16.engine"

# Lifespan context to initialize and release engine
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("[INFO] Initializing ADE20K SegFormer TRT engine...")
    app.state.seg_engine = RealTimeSegmentationTrT(engine_path=ENGINE_PATH, input_hw=(512, 512))
    yield
    print("[INFO] Releasing GPU resources (if any)...")
    if hasattr(app.state.seg_engine, "release"):
        app.state.seg_engine.release()

app = FastAPI(title="ADE20K Mobile Real-Time Segmentation", lifespan=lifespan)

# Mobile HTML frontend
@app.get("/")
async def index():
    return HTMLResponse("""
    <html>
  <body>
    <h2>ADE20K Mobile Real-Time Segmentation</h2>
    <div style="display:flex; gap:10px;">
      <div>
        <h4>Original Camera</h4>
        <video id="video" autoplay playsinline width="512" style="border:1px solid #000;"></video>
      </div>
      <div>
        <h4>Segmentation Overlay</h4>
        <img id="output" width="512" style="border:1px solid #000;" />
      </div>
    </div>
    <div>FPS: <span id="fps">0</span> | Latency: <span id="latency">0</span> ms 

    <canvas id="canvas" style="display:none;"></canvas>

    <script>
      const video = document.getElementById('video');
      const canvas = document.getElementById('canvas');
      const output = document.getElementById('output');
      const fpsEl = document.getElementById('fps');
      const latencyEl = document.getElementById('latency');
      const confEl = document.getElementById('conf');
      const ctx = canvas.getContext('2d');

      // Try rear camera first
      navigator.mediaDevices.getUserMedia({ video: { facingMode: { ideal: "environment" } } })
        .then(stream => { video.srcObject = stream; video.play(); })
        .catch(err => {
          console.warn("Rear camera unavailable, fallback to default:", err);
          navigator.mediaDevices.getUserMedia({ video: true }).then(stream => { video.srcObject = stream; video.play(); });
        });

      let lastTime = performance.now();

      async function sendFrameLoop() {
        if(video.videoWidth === 0){ requestAnimationFrame(sendFrameLoop); return; }

        // Draw frame to hidden canvas for engine input (512x512)
        canvas.width = 512;
        canvas.height = 512;
        ctx.drawImage(video, 0, 0, 512, 512);

        canvas.toBlob(async (blob) => {
          try {
            const formData = new FormData();
            formData.append("file", blob, "frame.jpg");
            const response = await fetch("/segment", { method: "POST", body: formData });
            const data = await response.json();

            // Set overlay
            output.src = "data:image/jpeg;base64," + data.image;

            // Update metrics
            fpsEl.innerText = data.fps.toFixed(1);
            latencyEl.innerText = data.latency_ms.toFixed(1);


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

        original_frame = frame.copy()

        # Resize for engine input (512x512)
        frame_resized = cv2.resize(frame, (512, 512), interpolation=cv2.INTER_LINEAR)

        # Run inference
        overlay, fps, latency_ms = app.state.seg_engine.infer(frame_resized)

        # Resize overlay to match original camera dimensions
        overlay = cv2.resize(overlay, (original_frame.shape[1], original_frame.shape[0]), interpolation=cv2.INTER_NEAREST)

        # Encode overlay to JPEG and convert to base64
        success, img_encoded = cv2.imencode('.jpg', overlay)
        if not success:
            return {"error": "Failed to encode overlay"}

        img_base64 = base64.b64encode(img_encoded.tobytes()).decode('utf-8')

        return JSONResponse({
            "image": img_base64,
            "fps": fps,
            "latency_ms": latency_ms,

        })

    except Exception as e:
        traceback.print_exc()
        return {"error": str(e)}
