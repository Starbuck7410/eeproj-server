import cv2
import imagezmq
import numpy as np
from flask import Flask, Response, render_template, jsonify
from ultralytics import YOLO
import threading
import time
from processing import detect_colored_balls

app = Flask(__name__)

# Load YOLO-World
model = YOLO('yolov8s.pt')

# AprilTag Detector
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
aruco_params = cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)

# Global state
output_frame = None
lock = threading.Lock()
fps = 0

def process_stream():
    global output_frame, fps
    image_hub = imagezmq.ImageHub()
    prev_time = time.time()

    while True:
        rpi_name, jpg_buffer = image_hub.recv_jpg()
        image_hub.send_reply(b'OK')

        frame = cv2.imdecode(np.frombuffer(jpg_buffer, dtype=np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            continue

        # Inference
        # results = model(frame, stream=True, verbose=False, device = '0' )
        # for r in results:
        #     frame = r.plot() # Draws YOLO boxes
        detections, frame = detect_colored_balls(frame)

        # AprilTags
        corners, ids, rejected = detector.detectMarkers(frame)
        
        
        # FPS calculation
        curr_time = time.time()
        fps = round(1 / (curr_time - prev_time), 1)
        prev_time = curr_time

        with lock:
            _, encoded_image = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
            output_frame = encoded_image.tobytes()

def generate():
    while True:
        time.sleep(0.01)
        with lock:
            if output_frame is None:
                continue
            frame_to_send = output_frame
        yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame_to_send + b'\r\n')

@app.route("/")
def index():
    # Flask looks in the /templates folder for index.html
    return render_template('index.html')

@app.route("/video_feed")
def video_feed():
    return Response(generate(), mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/api/garbage")
def garbage():
    x = 1.0
    y = -0.8
    z = 0
    target = {
        "x": x,
        "y": y,
        "z": z
    }
    return jsonify(target)

@app.route("/api/basket")
def basket():
    x = -1.0
    y = 0.8
    z = 0
    target = {
        "x": x,
        "y": y,
        "z": z
    }
    return jsonify

@app.route("/api/frame_stats")
def frame_stats():
    target = {
        "fps": fps
    }
    return jsonify(target)

if __name__ == "__main__":
    process_thread = threading.Thread(target=process_stream, daemon=True)
    process_thread.start()
    app.run(host='0.0.0.0', port=4999, threaded=True, debug=False)
