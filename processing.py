import cv2
import numpy as np

def detect_colored_balls(frame):

    # Blur to reduce noise
    # frame = cv2.GaussianBlur(frame, (9, 9), 2)

    # Convert to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # HSV ranges
    color_ranges = {
        "green": [
            (np.array([35, 50, 50]), np.array([85, 255, 255]))
        ],

        "blue": [
            (np.array([90, 50, 50]), np.array([130, 255, 255]))
        ],

        "purple": [
            (np.array([130, 50, 50]), np.array([160, 255, 255]))
        ],

        # red wraps around HSV hue space
        "red": [
            (np.array([0, 50, 50]), np.array([10, 255, 255])),
            (np.array([170, 50, 50]), np.array([180, 255, 255]))
        ]
    }

    detected_circles = []

    for color_name, ranges in color_ranges.items():

        # Build combined mask
        mask = np.zeros(hsv.shape[:2], dtype=np.uint8)

        for lower, upper in ranges:
            mask |= cv2.inRange(hsv, lower, upper)

        # Morphological cleanup
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        # Edge detection
        edges = cv2.Canny(mask, 50, 150)

        # Hough Circle Transform
        circles = cv2.HoughCircles(
            edges,
            cv2.HOUGH_GRADIENT,
            dp=1.2,
            minDist=30,
            param1=100,
            param2=15,
            minRadius=5,
            maxRadius=200
        )

        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")

            for (x, y, r) in circles:
                detected_circles.append({
                    "color": color_name,
                    "x": int(x),
                    "y": int(y),
                    "radius": int(r)
                })

                # Optional visualization
                frame = cv2.circle(frame, (x, y), r, (0, 255, 0), 2)
                frame = cv2.circle(frame, (x, y), 2, (0, 0, 255), 3)
    return detected_circles, frame  