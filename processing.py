import cv2
import config
import shared_state as state
import cv2
import imagezmq
import zmq
import time
import sys
import numpy as np
from vision_core import detect_colored_balls, get_relative_pos, get_relative_pos_between_tags


# AprilTag Detector
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
aruco_params = cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)


def dist(x, y):
    return x ** 2 + y ** 2



def process_stream():
    image_hub = imagezmq.ImageHub()
    image_hub.zmq_socket.setsockopt(zmq.RCVTIMEO, 1000)
    prev_time = time.time()

    while True:
        try:
            state.rpi_name, jpg_buffer = image_hub.recv_jpg()
        except zmq.error.Again:
            sys.stdout.write("\033[F") # Cursor up
            sys.stdout.write("\033[K") # Clear line
            sys.stdout.write("\033[31m") # Red
            print(f"Offline: No video data received for {int(time.time() - prev_time)} seconds.")
            sys.stdout.write("\033[0m") # Reset color just in case
            with state.lock:
                state.online = False
            continue
        
        image_hub.send_reply(b'OK')
        if(not state.online):
            sys.stdout.write("\033[F") # Cursor up
            sys.stdout.write("\033[K") # Clear line
            sys.stdout.write("\033[32m") # Green
            print("Online!")
            sys.stdout.write("\033[0m") # Reset color just in case
            with state.lock:
                state.online = True
        frame = cv2.imdecode(np.frombuffer(jpg_buffer, dtype=np.uint8), cv2.IMREAD_COLOR)
        # frame = cv2.flip(frame, 0)
        real_frame = frame.copy()
        with state.lock:
            state.resolution = frame.shape

        # Detect balls
        detections, frame = detect_colored_balls(frame)

        # AprilTags
        corners, ids, rejected = detector.detectMarkers(real_frame)
        if(ids is not None):
            cv2.aruco.drawDetectedMarkers(frame, corners, ids)
            robot_tag_idx = np.where(ids.flatten() == config.ROBOT_TAG_ID)[0]
            basket_tag_idx = np.where(ids.flatten() == config.BASKET_TAG_ID)[0] 
            min_x, min_y, new_x, new_y = config.BIG, config.BIG, config.BIG, config.BIG
            if(len(detections) > 0 and len(robot_tag_idx) > 0):
                for detection in detections:
                    detection_id = detection["id"]
                    # print(f'corners[robot_tag_idx[0]]: {corners[robot_tag_idx[0]]}')
                    new_x, new_y = get_relative_pos((detection["x"], detection["y"]), corners[robot_tag_idx[0]])
                    if(dist(new_x, new_y) < dist(min_x, min_y)):
                        min_x, min_y = new_x, new_y
                        with state.lock:
                            state.garbage_id = detection_id

                # This 'if' statment is breaking things, but I will leave it here for future reference as it is not a bad idea
                # if(dist(min_x - garbage_x, min_y - garbage_y) < JUMP_THRESHOLD_MM or garbage_x == 0 or garbage_y == 0): 
                with state.lock:
                    state.garbage_loc.update((min_x, min_y))
                # print(f"Garbage of id {garbage_id} at: {garbage_x:.2f}, {garbage_y:.2f}")
            else:
                with state.lock:
                    state.garbage_loc.invalidate()

                
            if(len(robot_tag_idx) > 0 and len (basket_tag_idx) > 0):
                new_x, new_y = get_relative_pos_between_tags(corners[robot_tag_idx[0]], corners[basket_tag_idx[0]])
                with state.lock:
                    state.basket_loc.update((new_x, new_y))
            else: 
                with state.lock:
                    state.basket_loc.invalidate()
        else:
            with state.lock:
                state.basket_loc.invalidate()
                state.garbage_loc.invalidate()

                
        
        # FPS calculation
        curr_time = time.time()
        with state.lock:
            state.fps = round(1 / (curr_time - prev_time), 1)
        prev_time = curr_time
        with state.lock:
            _, encoded_image = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
            state.output_frame = encoded_image.tobytes()