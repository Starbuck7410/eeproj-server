import cv2
import imagezmq
import zmq
import numpy as np
from flask import Flask, Response, render_template, jsonify
import threading
import time
from processing import detect_colored_balls, get_relative_pos, get_relative_pos_between_tags

app = Flask(__name__)
ROLL_AVG_N = 10
ROBOT_TAG_ID = 30
BASKET_TAG_ID = 5
BIG = 10000
# JUMP_THRESHOLD_MM = 200
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
garbage_id = 0
garbage_valid = False
basket_valid = False


arr_garbage_x = np.zeros(ROLL_AVG_N)
arr_garbage_y = np.zeros(ROLL_AVG_N)
arr_basket_x = np.zeros(ROLL_AVG_N)
arr_basket_y = np.zeros(ROLL_AVG_N)

basket_i = 0
garbage_i = 0


def dist(x, y):
    return x ** 2 + y ** 2


def avg_garbage(x, y):
    global garbage_i

    arr_garbage_x[garbage_i] = x
    arr_garbage_y[garbage_i] = y
    garbage_i = (garbage_i + 1) % ROLL_AVG_N

    return (arr_garbage_x.mean(), arr_garbage_y.mean())



def avg_basket(x, y):
    global basket_i

    arr_basket_x[basket_i] = x
    arr_basket_y[basket_i] = y

    basket_i = (basket_i + 1) % ROLL_AVG_N

    return (arr_basket_x.mean(), arr_basket_y.mean())


def process_stream():
    global output_frame, fps, lock, resolution, rpi_name, online, basket_x, basket_y, garbage_x, garbage_y, garbage_id, basket_valid, garbage_valid
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
        with lock:
            resolution = frame.shape

        # Detect balls
        detections, frame = detect_colored_balls(frame)

        # AprilTags
        corners, ids, rejected = detector.detectMarkers(real_frame)
        if(ids is not None):
            cv2.aruco.drawDetectedMarkers(frame, corners, ids)
            robot_tag_idx = np.where(ids.flatten() == ROBOT_TAG_ID)[0]
            basket_tag_idx = np.where(ids.flatten() == BASKET_TAG_ID)[0] 
            min_x, min_y, new_x, new_y = BIG, BIG, BIG, BIG
            if(len(detections) > 0 and len(robot_tag_idx) > 0):
                for detection in detections:
                    detection_id = detection["id"]
                    # print(f'corners[robot_tag_idx[0]]: {corners[robot_tag_idx[0]]}')
                    new_x, new_y = get_relative_pos((detection["x"], detection["y"]), corners[robot_tag_idx[0]])
                    if(dist(new_x, new_y) < dist(min_x, min_y)):
                        min_x, min_y = new_x, new_y
                        garbage_id = detection_id

                # This 'if' statment is breaking things, but I will leave it here for future reference as it is not a bad idea
                # if(dist(min_x - garbage_x, min_y - garbage_y) < JUMP_THRESHOLD_MM or garbage_x == 0 or garbage_y == 0): 
                with lock:
                    garbage_valid = True
                    garbage_x, garbage_y = avg_garbage(min_x, min_y)
                # print(f"Garbage of id {garbage_id} at: {garbage_x:.2f}, {garbage_y:.2f}")
            else:
                with lock:
                    garbage_valid = False

                
            if(len(robot_tag_idx) > 0 and len (basket_tag_idx) > 0):
                new_x, new_y = get_relative_pos_between_tags(corners[robot_tag_idx[0]], corners[basket_tag_idx[0]])
                with lock:
                    basket_valid = True
                    basket_x, basket_y = avg_basket(new_x, new_y)
            else: 
                with lock:
                    basket_valid = False
        else:
            with lock:
                basket_valid = False
                garbage_valid = False

                
        
        # FPS calculation
        curr_time = time.time()
        with lock:
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
    with lock:
        local_x = garbage_x
        local_y = garbage_y
        local_valid = garbage_valid
    target = {
        "x": float(local_x / 1000),
        "y": float(local_y / 1000),
        "valid": local_valid,
    }
    return jsonify(target)

@app.route("/api/basket")
def basket():
    with lock:
        local_x = basket_x
        local_y = basket_y
        local_valid = basket_valid
    target = {
        "x": float(local_x / 1000),
        "y": float(local_y / 1000),
        "valid": local_valid,
    }
    return jsonify(target)

@app.route("/api/frame_stats")
def frame_stats():
    # print(f'Garbage ID: {garbage_id}')
    with lock:
        local_fps = fps
        local_res = resolution
        local_online = online
        local_name = rpi_name
        local_g_id = garbage_id

    stats = {
        "fps": local_fps,
        "resolution": f"{local_res[0]} ⨯ {local_res[1]}" if len(local_res) > 1 else "0 ⨯ 0",
        "online": local_online,
        "device_name": local_name,
        "garbage_id": local_g_id
    }
    return jsonify(stats)

if __name__ == "__main__":
    process_thread = threading.Thread(target=process_stream, daemon=True)
    process_thread.start()
    app.run(host='0.0.0.0', port=4999, threaded=True, debug=False)






