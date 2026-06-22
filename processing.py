import cv2
import config
import shared_state as state
import cv2
import imagezmq
import zmq
import time
import sys
import numpy as np
import math
from vision_core import detect_colored_balls, get_relative_pos, get_relative_pos_between_tags


# AprilTag Detector
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
aruco_params = cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)


def dist(x, y):
    return x ** 2 + y ** 2

def get_closest_object_3(detections, target_corners, ref_corners, min_dist_from_ref = config.BASKET_TAG_SIZE_MM):
    min_x, min_y, min_id = config.BIG, config.BIG, -1
    for detection in detections:
        detection_id = detection["id"]
        robot_x, robot_y, _ = get_relative_pos_between_tags(ref_corners, target_corners, ref_size=config.BASKET_TAG_SIZE_MM)
        ball_x, ball_y = get_relative_pos(ref_corners, (detection["x"], detection["y"]), ref_size=config.BASKET_TAG_SIZE_MM)
        if(dist(ball_x, ball_y) < min_dist_from_ref ** 2):
            continue
        new_x, new_y = vec_sub((ball_x, ball_y), (robot_x, robot_y))
        if(dist(new_x - config.ROBOT_SIZE_X, new_y) < dist(min_x - config.ROBOT_SIZE_X, min_y)):
            min_x, min_y = new_x, new_y
            min_id = detection_id

    return min_x, min_y, min_id


def get_closest_object(detections, target_corners):
    min_x, min_y, min_id = config.BIG, config.BIG, -1
    for detection in detections:
        detection_id = detection["id"]
        new_x, new_y = get_relative_pos(target_corners, (detection["x"], detection["y"]), ref_size=config.ROBOT_TAG_SIZE_MM)
        if(dist(new_x - config.ROBOT_SIZE_X, new_y) < dist(min_x - config.ROBOT_SIZE_X, min_y)):
            min_x, min_y = new_x, new_y
            min_id = detection_id
    return min_x, min_y, min_id

def vec_flip(tup):
    return tuple(-x for x in tup)


def vec_sub(vec_a, vec_b):
    return tuple(a - b for a, b in zip(vec_a, vec_b))


def vec_rotate(vec, angle_deg):
    angle_rad = math.radians(angle_deg)
    x, y = vec
    c = math.cos(angle_rad)
    s = math.sin(angle_rad)

    return (
        x * c - y * s,
        x * s + y * c
    )


def process_stream():
    image_hub = imagezmq.ImageHub()
    image_hub.zmq_socket.setsockopt(zmq.RCVTIMEO, 1000)
    prev_time = time.time()
    while True:
        try:
            state.rpi_name, jpg_buffer = image_hub.recv_jpg()
            # state.rpi_name, frame = image_hub.recv_image()
        except zmq.error.Again:
            sys.stdout.write("\033[F") # Cursor up
            sys.stdout.write("\033[K") # Clear line
            sys.stdout.write("\033[31m") # Red
            print(f"Offline: No video data received for {int(time.time() - prev_time)} seconds.")
            sys.stdout.write("\033[0m") 
            with state.lock:
                state.online = False
            continue
        
        image_hub.send_reply(b'OK')
        
        sys.stdout.write("\033[F") 
        sys.stdout.write("\033[K") 
        sys.stdout.write("\033[32m") # Green
        print("Online!")
        sys.stdout.write("\033[0m") 
        frame = cv2.imdecode(np.frombuffer(jpg_buffer, dtype=np.uint8), cv2.IMREAD_COLOR)
        real_frame = frame.copy()
        
        with state.lock:
            state.online = True
            state.resolution = frame.shape

        # Detect balls
        detections, frame = detect_colored_balls(frame)

        # AprilTags
        corners, ids, _ = detector.detectMarkers(real_frame)
        if(ids is not None):
            cv2.aruco.drawDetectedMarkers(frame, corners, ids)
            robot_tag_idx = np.where(ids.flatten() == config.ROBOT_TAG_ID)[0]
            basket_tag_idx = np.where(ids.flatten() == config.BASKET_TAG_ID)[0] 
            garbage_x, garbage_y, garbage_id = config.BIG, config.BIG, -1

            if(len(detections) > 0 and len(robot_tag_idx) > 0 and len(basket_tag_idx) > 0):
                # Detected everything (3-Body Case)
                garbage_x, garbage_y, garbage_id = get_closest_object_3(detections, corners[robot_tag_idx[0]], corners[basket_tag_idx[0]])
                basket_x, basket_y, rotation = get_relative_pos_between_tags(corners[basket_tag_idx[0]], corners[robot_tag_idx[0]], ref_size = config.BASKET_TAG_SIZE_MM)
                
                garbage_x, garbage_y = vec_rotate((garbage_x, garbage_y), -rotation)
                basket_x, basket_y = vec_rotate(vec_flip((basket_x, basket_y)), -rotation)
                
                with state.lock:
                    state.basket_loc.update((basket_x, basket_y))
                    state.garbage_loc.update((garbage_x, garbage_y))
                    state.garbage_id = garbage_id
            else:
                # Detected robot and garbage only
                if(len(detections) > 0 and len(robot_tag_idx) > 0):
                    garbage_x, garbage_y, garbage_id = get_closest_object(detections, corners[robot_tag_idx[0]])
                    with state.lock:
                        state.garbage_loc.update((garbage_x, garbage_y))
                        state.garbage_id = garbage_id
                else: 
                    # No garbage detected
                    with state.lock:
                        state.garbage_loc.invalidate()

                # Detected robot and basket only
                if(len(robot_tag_idx) > 0 and len(basket_tag_idx) > 0):
                    # FIX: Capture rotation and apply vec_rotate to maintain frame of reference consistency
                    basket_x, basket_y, rotation = get_relative_pos_between_tags(corners[basket_tag_idx[0]], corners[robot_tag_idx[0]], ref_size=config.BASKET_TAG_SIZE_MM)
                    basket_x, basket_y = vec_flip(vec_rotate((basket_x, basket_y), -rotation))
                    with state.lock:
                        state.basket_loc.update((basket_x, basket_y))
                else:
                    # No basket detected 
                    with state.lock:
                        state.basket_loc.invalidate()

            if(garbage_id != -1):
                with state.lock:
                    state.garbage_loc.update((garbage_x, garbage_y))
                    state.garbage_id = garbage_id

        # No tags detected        
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



