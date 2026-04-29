import cv2
import imagezmq
import numpy as np
from flask import Flask, Response, render_template_string
from ultralytics import YOLO
import threading
import time

app = Flask(__name__)
model = YOLO('yolov8n.pt')

# AprilTag Detector
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
aruco_params = cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)

# Global storage
output_frame = None
lock = threading.Lock()

# --- CONFIGURATION ---
CONFIDENCE_THRESHOLD = 0.5  # Only show objects with > 50% confidence
# ---------------------

def process_stream():
    global output_frame
    image_hub = imagezmq.ImageHub()
    
    while True:
        rpi_name, jpg_buffer = image_hub.recv_jpg()
        image_hub.send_reply(b'OK')

        # Decode JPEG
        frame = cv2.imdecode(np.frombuffer(jpg_buffer, dtype=np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            continue

        # --- FIX COLORS ---
        # If the Pi sends RGB and OpenCV treats it as BGR, swap them here
        # frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR) # Uncomment if colors are inverted

        # --- YOLO WITH CONFIDENCE FILTER ---
        # conf=0.5 filters results at the model level for better performance
        results = model(frame, stream=True, conf=CONFIDENCE_THRESHOLD, verbose=False, device='cpu')
        
        for r in results:
            frame = r.plot() # Annotates the frame with filtered boxes

        # --- AprilTag Logic ---
        corners, ids, _ = detector.detectMarkers(frame)
        if ids is not None:
            cv2.aruco.drawDetectedMarkers(frame, corners, ids)

        with lock:
            # When encoding for the web, ensure it's in BGR so browsers read it as RGB
            _, encoded_image = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
            output_frame = encoded_image.tobytes()

def generate():
    while True:
        # Avoid high-frequency polling when no frame is ready
        time.sleep(0.01) 
        with lock:
            if output_frame is None:
                continue
            frame_to_send = output_frame
        
        yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame_to_send + b'\r\n')

@app.route("/")
def index():
    return render_template_string('<h1>Vision Feed</h1><img src="/video_feed" width="640">')

@app.route("/video_feed")
def video_feed():
    return Response(generate(), mimetype="multipart/x-mixed-replace; boundary=frame")

if __name__ == "__main__":
    # Move processing to a daemon thread so it doesn't block the main Flask thread
    process_thread = threading.Thread(target=process_stream, daemon=True)
    process_thread.start()
    
    # Run Flask with 'threaded=True' to allow concurrent web requests
    app.run(host='0.0.0.0', port=5000, threaded=True, debug=False)