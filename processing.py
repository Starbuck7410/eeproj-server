import cv2
import numpy as np

import cv2
import numpy as np

TAG_SIZE_MM = 26.2

def detect_colored_balls(frame):


    # Downscale for speed (e.g., to 480p width)
    # scale = 640 / frame.shape[1]
    scale = 1
    # small_frame = cv2.resize(frame, (0, 0), fx=scale, fy=scale)
    
    # Blurring & HSV (Only done once)
    # blurred = cv2.GaussianBlur(frame, (5, 5), 0)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Simplified color ranges for lookup
    color_ranges = {
        "red":    [(np.array([0, 50, 50]), np.array([10, 255, 255])),
                   (np.array([150, 50, 50]), np.array([180, 255, 255]))],
        "orange": [(np.array([10, 70, 70]), np.array([20, 255, 255]))],
        "yellow": [(np.array([20, 50, 50]), np.array([40, 255, 255]))],
        "green":  [(np.array([40, 50, 50]), np.array([80, 255, 255]))],
        "cyan":   [(np.array([80, 50, 50]), np.array([100, 255, 255]))],
        "blue":   [(np.array([100, 50, 50]), np.array([110, 255, 255]))],
        "purple": [(np.array([110, 50, 50]), np.array([150, 255, 255]))],
    }

    detected_circles = []

    # Create one master mask for all colors to find blobs
    full_mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
    for ranges in color_ranges.values():
        for lower, upper in ranges:
            full_mask |= cv2.inRange(hsv, lower, upper)

    # Fast Morphological cleanup
    kernel = np.ones((3, 3), np.uint8)
    full_mask = cv2.morphologyEx(full_mask, cv2.MORPH_OPEN, kernel)

    # Use Contours instead of HoughCircles (Much Faster)
    contours, _ = cv2.findContours(full_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    id = 0
    for cnt in contours:
        
        area = cv2.contourArea(cnt)
        perimeter = cv2.arcLength(cnt, True)
        if area < 400:  # Filter noise
            continue
        
        circularity = (4 * np.pi * area) / (perimeter ** 2)

        if circularity < 0.6: continue # Ignore non-round objects

        ((x, y), radius) = cv2.minEnclosingCircle(cnt)
        
        if radius > 5:
            # Classify color by sampling the center pixel in HSV
            # This replaces the loop-per-color logic
            cx, cy = int(x), int(y)
            sample_pixel = hsv[cy, cx]
            
            detected_color = "unknown"
            for name, ranges in color_ranges.items():
                for lower, upper in ranges:
                    if np.all(sample_pixel >= lower) and np.all(sample_pixel <= upper):
                        detected_color = name
                        break
            
            if detected_color != "unknown":
                # Rescale coordinates back to original size
                orig_x, orig_y, orig_r = int(x/scale), int(y/scale), int(radius/scale)
                
                detected_circles.append({
                    "color": detected_color,
                    "x": orig_x, "y": orig_y, "radius": orig_r, "id": id
                })
                
                id += 1

                # Drawing on original frame
                cv2.circle(frame, (orig_x, orig_y), orig_r, (0, 255, 0), 2)
                cv2.putText(frame, f"{detected_color}, {id}", (orig_x, orig_y - orig_r), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 4)
                cv2.putText(frame, f"{detected_color}, {id}", (orig_x, orig_y - orig_r), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)

    return detected_circles, frame



def get_relative_pos(ball_pixel_coords, tag_corners):
    # 1. Define the real-world coordinates of the tag corners (in mm)
    # Assuming tag is centered at (0,0), corners are half-size away
    s = TAG_SIZE_MM / 2
    # Order: Top-Left, Top-Right, Bottom-Right, Bottom-Left
    world_pts = np.array([
        [-s,  s], [ s,  s], 
        [ s, -s], [-s, -s]
    ], dtype=np.float32)

    # 2. Get the detected pixel corners from AprilTag
    # Ensure tag_corners is shaped (4, 2)
    pixel_pts = tag_corners.reshape(4, 2).astype(np.float32)

    # 3. Calculate the Homography Matrix
    # This matrix 'M' translates any pixel (u, v) on that plane to real (x, y)
    M, _ = cv2.findHomography(pixel_pts, world_pts)

    # 4. Transform the ball pixel coordinate
    ball_pt = np.array([[ball_pixel_coords]], dtype=np.float32)
    real_world_pt = cv2.perspectiveTransform(ball_pt, M)

    # Result is [[x, y]]
    rel_x = real_world_pt[0][0][0]
    rel_y = real_world_pt[0][0][1]
    
    return (round(rel_x, 2), round(rel_y, 2))