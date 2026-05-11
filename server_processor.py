import cv2
import imagezmq
import zmq
import numpy as np
from flask import Flask, Response, render_template, jsonify
from ultralytics import YOLO
import threading
import time
from processing import detect_colored_balls, get_relative_pos

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
resolution = (0, 0)
rpi_name = ""
online = False
basket_x = 0
basket_y = 0
garbage_x = 0
garbage_y = 0

def process_stream():
    global output_frame, fps, lock, resolution, rpi_name, online, basket_x, basket_y, garbage_x, garbage_y
    image_hub = imagezmq.ImageHub()
    image_hub.zmq_socket.setsockopt(zmq.RCVTIMEO, 3000)
    prev_time = time.time()

    while True:
        try:
            rpi_name, jpg_buffer = image_hub.recv_jpg()
        except zmq.error.Again:
            print(f"No data received for {int(time.time() - prev_time)} seconds.")
            online = False
            continue

        image_hub.send_reply(b'OK')
        online = True
        frame = cv2.imdecode(np.frombuffer(jpg_buffer, dtype=np.uint8), cv2.IMREAD_COLOR)
        real_frame = frame.copy()
        resolution = frame.shape
        # YOLO Detection
        # results = model(real_frame, stream=True, verbose=False, device = '0' )
        # for r in results:
        #     frame = r.plot() # Draws YOLO boxes

        # Hough
        detections, frame = detect_colored_balls(real_frame)

        # AprilTags
        corners, ids, rejected = detector.detectMarkers(real_frame)
        if(ids is not None):
            cv2.aruco.drawDetectedMarkers(frame, corners, ids)
            if(len(detections) > 0):
                garbage_x, garbage_y = get_relative_pos(detections[0], corners[0])
        
        # FPS calculation
        curr_time = time.time()
        fps = round(1 / (curr_time - prev_time), 1)
        prev_time = curr_time
        with lock:
            _, encoded_image = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 60])
            output_frame = encoded_image.tobytes()

def generate():
    global lock
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
    target = {
        "x": garbage_x,
        "y": garbage_y,
        "z": 0
    }
    return jsonify(target)

@app.route("/api/basket")
def basket():
    target = {
        "x": basket_x,
        "y": basket_y,
        "z": 0
    }
    return jsonify(target)

@app.route("/api/frame_stats")
def frame_stats():
    stats = {
        "fps": fps,
        "resolution": f"{resolution[0]} ⨯ {resolution[1]}",
        "online": online,
        "device_name": rpi_name
    }
    return jsonify(stats)

if __name__ == "__main__":
    process_thread = threading.Thread(target=process_stream, daemon=True)
    process_thread.start()
    app.run(host='0.0.0.0', port=4999, threaded=True, debug=False)
